import json
import logging
import os
import os.path
import shutil
from itertools import groupby

import numpy as np
from celery import shared_task
from celery.signals import before_task_publish, task_failure, task_prerun, task_success
from django.apps import apps
from django.conf import settings
from django.contrib.auth import get_user_model
from django.db.models import F, Q
from django.utils.text import slugify
from django.utils.translation import gettext as _
from django_redis import get_redis_connection
from easy_thumbnails.files import get_thumbnailer
from kraken.lib import train as kraken_train
# DO NOT REMOVE THIS IMPORT, it will break celery tasks located in this file
from reporting.tasks import create_task_reporting  # noqa F401
from users.consumers import send_event

logger = logging.getLogger(__name__)
User = get_user_model()
redis_ = get_redis_connection()


# tasks for which to keep track of the state and update the front end
STATE_TASKS = [
    'core.tasks.binarize',
    'core.tasks.segment',
    'core.tasks.transcribe'
]


def update_client_state(part_id, task, status, task_id=None, data=None):
    DocumentPart = apps.get_model('core', 'DocumentPart')
    part = DocumentPart.objects.get(pk=part_id)
    task_name = task.split('.')[-1]
    send_event('document', part.document.pk, "part:workflow", {
        "id": part.pk,
        "process": task_name,
        "status": status,
        "task_id": task_id,
        "data": data or {}
    })


@shared_task(autoretry_for=(MemoryError,), default_retry_delay=60)
def generate_part_thumbnails(instance_pk=None, user_pk=None, **kwargs):
    if not getattr(settings, 'THUMBNAIL_ENABLE', True):
        return

    try:
        DocumentPart = apps.get_model('core', 'DocumentPart')
        part = DocumentPart.objects.get(pk=instance_pk)
    except DocumentPart.DoesNotExist:
        logger.error('Trying to compress non-existent DocumentPart : %d', instance_pk)
        return

    aliases = {}
    thbnr = get_thumbnailer(part.image)
    for alias, config in settings.THUMBNAIL_ALIASES[''].items():
        aliases[alias] = thbnr.get_thumbnail(config).url
    return aliases


@shared_task(autoretry_for=(MemoryError,), default_retry_delay=3 * 60)
def convert(instance_pk=None, user_pk=None, **kwargs):
    if user_pk:
        try:
            user = User.objects.get(pk=user_pk)
            # If quotas are enforced, assert that the user still has free CPU minutes
            if not settings.DISABLE_QUOTAS and user.cpu_minutes_limit() is not None:
                assert user.has_free_cpu_minutes(), f"User {user.id} doesn't have any CPU minutes left"
        except User.DoesNotExist:
            user = None

    try:
        DocumentPart = apps.get_model('core', 'DocumentPart')
        part = DocumentPart.objects.get(pk=instance_pk)
    except DocumentPart.DoesNotExist:
        logger.error('Trying to convert non-existent DocumentPart : %d', instance_pk)
        return
    part.convert()


@shared_task(autoretry_for=(MemoryError,), default_retry_delay=5 * 60)
def lossless_compression(instance_pk=None, user_pk=None, **kwargs):
    if user_pk:
        try:
            user = User.objects.get(pk=user_pk)
            # If quotas are enforced, assert that the user still has free CPU minutes
            if not settings.DISABLE_QUOTAS and user.cpu_minutes_limit() is not None:
                assert user.has_free_cpu_minutes(), f"User {user.id} doesn't have any CPU minutes left"
        except User.DoesNotExist:
            user = None

    try:
        DocumentPart = apps.get_model('core', 'DocumentPart')
        part = DocumentPart.objects.get(pk=instance_pk)
    except DocumentPart.DoesNotExist:
        logger.error('Trying to compress non-existent DocumentPart : %d', instance_pk)
        return
    part.compress()


@shared_task(autoretry_for=(MemoryError,), default_retry_delay=10 * 60)
def binarize(instance_pk=None, user_pk=None, binarizer=None, threshold=None, **kwargs):
    try:
        DocumentPart = apps.get_model('core', 'DocumentPart')
        part = DocumentPart.objects.get(pk=instance_pk)
    except DocumentPart.DoesNotExist:
        logger.error('Trying to binarize non-existent DocumentPart : %d', instance_pk)
        return

    if user_pk:
        try:
            user = User.objects.get(pk=user_pk)
            # If quotas are enforced, assert that the user still has free CPU minutes
            if not settings.DISABLE_QUOTAS and user.cpu_minutes_limit() is not None:
                assert user.has_free_cpu_minutes(), f"User {user.id} doesn't have any CPU minutes left"
        except User.DoesNotExist:
            user = None
    else:
        user = None

    try:
        part.binarize(threshold=threshold)
    except Exception as e:
        if user:
            user.notify(_("Something went wrong during the binarization!"),
                        id="binarization-error", level='danger')
        part.workflow_state = part.WORKFLOW_STATE_CREATED
        part.save()
        logger.exception(e)
        raise e
    else:
        if user:
            user.notify(_("Binarization done!"),
                        id="binarization-success", level='success')


def make_segmentation_training_data(part):
    data = {
        'image': part.image.path,
        'baselines': [{'script': line.typology and line.typology.name or 'default',
                       'baseline': line.baseline}
                      for line in part.lines.only('baseline', 'typology')
                      if line.baseline],
        'regions': {typo: list(reg.box for reg in regs)
                    for typo, regs in groupby(
            part.blocks.only('box', 'typology').order_by('typology'),
            key=lambda reg: reg.typology and reg.typology.name or 'default')}
    }
    return data


@shared_task(bind=True, autoretry_for=(MemoryError,), default_retry_delay=60 * 60)
def segtrain(task, model_pk, part_pks, document_pk=None, user_pk=None, **kwargs):
    # # Note hack to circumvent AssertionError: daemonic processes are not allowed to have children
    from multiprocessing import current_process
    current_process().daemon = False

    if user_pk:
        try:
            user = User.objects.get(pk=user_pk)
            # If quotas are enforced, assert that the user still has free CPU minutes, GPU minutes and disk storage
            if not settings.DISABLE_QUOTAS:
                if user.cpu_minutes_limit() is not None:
                    assert user.has_free_cpu_minutes(), f"User {user.id} doesn't have any CPU minutes left"
                if user.gpu_minutes_limit() is not None:
                    assert user.has_free_gpu_minutes(), f"User {user.id} doesn't have any GPU minutes left"
                if user.disk_storage_limit() is not None:
                    assert user.has_free_disk_storage(), f"User {user.id} doesn't have any disk storage left"
        except User.DoesNotExist:
            user = None
    else:
        user = None

    def msg(txt, fg=None, nl=False):
        logger.info(txt)

    redis_.set('segtrain-%d' % model_pk, json.dumps({'task_id': task.request.id}))

    Document = apps.get_model('core', 'Document')
    DocumentPart = apps.get_model('core', 'DocumentPart')
    OcrModel = apps.get_model('core', 'OcrModel')

    model = OcrModel.objects.get(pk=model_pk)

    try:
        load = model.file.path
    except ValueError:  # model is empty
        load = settings.KRAKEN_DEFAULT_SEGMENTATION_MODEL
        model.file = model.file.field.upload_to(model, slugify(model.name) + '.mlmodel')

    model_dir = os.path.join(settings.MEDIA_ROOT, os.path.split(model.file.path)[0])

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    try:
        model.training = True
        model.save()
        send_event('document', document_pk, "training:start", {
            "id": model.pk,
        })
        qs = DocumentPart.objects.filter(pk__in=part_pks).prefetch_related('lines')

        ground_truth = list(qs)
        if ground_truth[0].document.line_offset == Document.LINE_OFFSET_TOPLINE:
            topline = True
        elif ground_truth[0].document.line_offset == Document.LINE_OFFSET_CENTERLINE:
            topline = None
        else:
            topline = False

        np.random.default_rng(241960353267317949653744176059648850006).shuffle(ground_truth)
        partition = max(1, int(len(ground_truth) / 10))

        training_data = []
        evaluation_data = []
        for part in qs[partition:]:
            training_data.append(make_segmentation_training_data(part))
        for part in qs[:partition]:
            evaluation_data.append(make_segmentation_training_data(part))

        DEVICE = getattr(settings, 'KRAKEN_TRAINING_DEVICE', 'cpu')
        LOAD_THREADS = getattr(settings, 'KRAKEN_TRAINING_LOAD_THREADS', 0)
        trainer = kraken_train.KrakenTrainer.segmentation_train_gen(
            message=msg,
            output=os.path.join(model_dir, 'version'),
            format_type=None,
            device=DEVICE,
            load=load,
            training_data=training_data,
            evaluation_data=evaluation_data,
            threads=LOAD_THREADS,
            augment=True,
            resize='both',
            hyper_params={'epochs': 30},
            load_hyper_parameters=True,
            topline=topline
        )

        def _print_eval(epoch=0, accuracy=0, mean_acc=0, mean_iu=0, freq_iu=0,
                        val_metric=0):
            model.refresh_from_db()
            model.training_epoch = epoch
            model.training_accuracy = float(val_metric)
            # model.training_total = chars
            # model.training_errors = error
            relpath = os.path.relpath(model_dir, settings.MEDIA_ROOT)
            model.new_version(file=f'{relpath}/version_{epoch}.mlmodel')
            model.save()

            send_event('document', document_pk, "training:eval", {
                "id": model.pk,
                'versions': model.versions,
                'epoch': epoch,
                'accuracy': float(val_metric)
                # 'chars': chars,
                # 'error': error
            })

        trainer.run(_print_eval)

        best_version = os.path.join(model_dir,
                                    f'version_{trainer.stopper.best_epoch}.mlmodel')

        try:
            shutil.copy(best_version, model.file.path)  # os.path.join(model_dir, filename)
        except FileNotFoundError:
            user.notify(_("Training didn't get better results than base model!"),
                        id="seg-no-gain-error", level='warning')
            shutil.copy(load, model.file.path)

    except Exception as e:
        send_event('document', document_pk, "training:error", {
            "id": model.pk,
        })
        if user:
            user.notify(_("Something went wrong during the segmenter training process!"),
                        id="training-error", level='danger')
        logger.exception(e)
        raise e
    else:
        if user:
            user.notify(_("Training finished!"),
                        id="training-success",
                        level='success')
    finally:
        model.training = False
        model.file_size = model.file.size
        model.save()

        send_event('document', document_pk, "training:done", {
            "id": model.pk,
        })


@shared_task(autoretry_for=(MemoryError,), default_retry_delay=5 * 60)
def segment(instance_pk=None, user_pk=None, model_pk=None,
            steps=None, text_direction=None, override=None,
            **kwargs):
    """
    steps can be either 'regions', 'lines' or 'both'
    """
    try:
        DocumentPart = apps.get_model('core', 'DocumentPart')
        part = DocumentPart.objects.get(pk=instance_pk)
    except DocumentPart.DoesNotExist:
        logger.error('Trying to segment non-existent DocumentPart : %d', instance_pk)
        return

    try:
        OcrModel = apps.get_model('core', 'OcrModel')
        model = OcrModel.objects.get(pk=model_pk)
    except OcrModel.DoesNotExist:
        model = None

    if user_pk:
        try:
            user = User.objects.get(pk=user_pk)
            # If quotas are enforced, assert that the user still has free CPU minutes
            if not settings.DISABLE_QUOTAS and user.cpu_minutes_limit() is not None:
                assert user.has_free_cpu_minutes(), f"User {user.id} doesn't have any CPU minutes left"
        except User.DoesNotExist:
            user = None
    else:
        user = None

    try:
        if steps == 'masks':
            part.make_masks()
        else:
            part.segment(steps=steps,
                         override=override,
                         text_direction=text_direction,
                         model=model)
    except Exception as e:
        if user:
            user.notify(_("Something went wrong during the segmentation!"),
                        id="segmentation-error", level='danger')
        part.workflow_state = part.WORKFLOW_STATE_CONVERTED
        part.save()
        logger.exception(e)
        raise e
    else:
        if user:
            user.notify(_("Segmentation done!"),
                        id="segmentation-success", level='success')


@shared_task(autoretry_for=(MemoryError,), default_retry_delay=60)
def recalculate_masks(instance_pk=None, user_pk=None, only=None, **kwargs):
    if user_pk:
        try:
            user = User.objects.get(pk=user_pk)
            # If quotas are enforced, assert that the user still has free CPU minutes
            if not settings.DISABLE_QUOTAS and user.cpu_minutes_limit() is not None:
                assert user.has_free_cpu_minutes(), f"User {user.id} doesn't have any CPU minutes left"
        except User.DoesNotExist:
            user = None

    try:
        DocumentPart = apps.get_model('core', 'DocumentPart')
        part = DocumentPart.objects.get(pk=instance_pk)
    except DocumentPart.DoesNotExist:
        logger.error('Trying to recalculate masks of non-existent DocumentPart : %d', instance_pk)
        return

    result = part.make_masks(only=only)
    send_event('document', part.document.pk, "part:mask", {
        "id": part.pk,
        "lines": [{'pk': line.pk, 'mask': line.mask} for line in result]
    })


def train_kraken(qs, document, transcription, model=None, user=None):
    # # Note hack to circumvent AssertionError: daemonic processes are not allowed to have children
    from multiprocessing import current_process
    current_process().daemon = False

    # try to minimize what is loaded in memory for large datasets
    ground_truth = list(qs.values('content',
                                  baseline=F('line__baseline'),
                                  mask=F('line__mask'),
                                  image=F('line__document_part__image')))

    np.random.default_rng(241960353267317949653744176059648850006).shuffle(ground_truth)

    partition = int(len(ground_truth) / 10)

    training_data = [{'image': os.path.join(settings.MEDIA_ROOT, lt['image']),
                      'text': lt['content'],
                      'baseline': lt['baseline'],
                      'boundary': lt['mask']} for lt in ground_truth[partition:]]
    evaluation_data = [{'image': os.path.join(settings.MEDIA_ROOT, lt['image']),
                        'text': lt['content'],
                        'baseline': lt['baseline'],
                        'boundary': lt['mask']} for lt in ground_truth[:partition]]

    load = None
    try:
        load = model.file.path
    except ValueError:  # model is empty
        filename = slugify(model.name) + '.mlmodel'
        model.file = model.file.field.upload_to(model, filename)
        model.save()

    model_dir = os.path.join(settings.MEDIA_ROOT, os.path.split(model.file.path)[0])

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    DEVICE = getattr(settings, 'KRAKEN_TRAINING_DEVICE', 'cpu')
    LOAD_THREADS = getattr(settings, 'KRAKEN_TRAINING_LOAD_THREADS', 0)
    if (document.main_script
        and (document.main_script.text_direction == 'horizontal-rl'
             or document.main_script.text_direction == 'vertical-rl')):
        reorder = 'R'
    else:
        reorder = 'L'
    trainer = (kraken_train.KrakenTrainer
               .recognition_train_gen(device=DEVICE,
                                      load=load,
                                      output=os.path.join(model_dir, 'version'),
                                      format_type=None,
                                      training_data=training_data,
                                      evaluation_data=evaluation_data,
                                      resize='add',
                                      threads=LOAD_THREADS,
                                      augment=False,
                                      hyper_params={'batch_size': 1},
                                      load_hyper_parameters=True,
                                      reorder=reorder))

    def _print_eval(epoch=0, accuracy=0, chars=0, error=0, val_metric=0):
        model.refresh_from_db()
        model.training_epoch = epoch
        model.training_accuracy = float(accuracy.item())
        model.training_total = int(chars)
        model.training_errors = int(error)
        relpath = os.path.relpath(model_dir, settings.MEDIA_ROOT)
        model.new_version(file=f'{relpath}/version_{epoch}.mlmodel')
        model.save()

        send_event('document', document.pk, "training:eval", {
            "id": model.pk,
            'versions': model.versions,
            'epoch': epoch,
            'accuracy': float(accuracy.item()),
            'chars': int(chars),
            'error': int(error)})

    trainer.run(_print_eval)

    if trainer.stopper.best_epoch != 0:
        best_version = os.path.join(model_dir, f'version_{trainer.stopper.best_epoch}.mlmodel')
        shutil.copy(best_version, model.file.path)
    else:
        raise ValueError('No model created.')


def crop_image(image, mask, polygon=True, smooth=True, background='dominant'):
    """ Crop polygon or bounding box from image """
    from shapely.geometry import Polygon
    from PIL import Image, ImageDraw, ImageFilter
    bbox = Polygon(mask).bounds
    cutout = image.crop(bbox)
    if polygon:
        # Background
        if background == 'white':
            bg = Image.new("L", cutout.size, 255)
        else:
            # Get dominant color
            img = cutout.copy()
            img = img.convert(cutout.mode)
            img = img.resize((1, 1), resample=0)
            dominant_color = img.getpixel((0, 0))
            bg = Image.new(cutout.mode, cutout.size, dominant_color)
        mask_img = Image.new("L", cutout.size, 255)
        # Mask text region
        draw = ImageDraw.Draw(mask_img)
        draw.polygon([(point[0] - bbox[0], point[1] - bbox[1]) for point in mask], 0, 1)
        if smooth:
            mask_img = mask_img.filter(ImageFilter.GaussianBlur(3))
        # Combine bg and cutout
        cutout.paste(bg, (0, 0), mask_img)
    return cutout

def update_calamarimodel(model):
    """ Updates Calamarimodels from Version to the newest version """
    from pathlib import Path
    from zipfile import ZipFile
    from calamari_ocr.ocr import SavedCalamariModel

    if not model.file.path.endswith('.calamarimodel'):
        return False
    try:
        model_path = Path(model.file.path).parent
        with ZipFile(model.file.path, 'r') as zipped:
            zipped.extractall(str(model_path.absolute()))
        list_json = list(model_path.glob('*.json'))
        Path(model.file.path).unlink()
        for jsonfpath in list_json:
            calamarimodel = SavedCalamariModel(str(jsonfpath.absolute()))
            calamarimodel.update_checkpoint()
        for old_files in model_path.glob('*.json_*'):
            old_files.unlink()
            old_files.parent.joinpath(old_files.name.replace('.json_', '.h5_')).unlink()
        with ZipFile(model.file.path, 'w') as zipped:
            for jsonfpath in list_json:
                zipped.write(str(jsonfpath.absolute()), jsonfpath.name)
                [zipped.write(str(modelfiles.absolute()), str(modelfiles.absolute().relative_to(model_path))) for
                 modelfiles in model_path.joinpath(jsonfpath.name.replace('.json', '')).rglob('*')
                 if modelfiles.is_file()]
    except:
        return False

    return True

def default_expert_settings():
    from collections import namedtuple
    expert_settings = namedtuple('Expert_Settings', 'mask_polygon')
    expert_settings.crop_polygon = True
    expert_settings.crop_smoothing = True
    expert_settings.crop_background = 'dominant'
    return expert_settings

def train_calamari(qs, document, transcription, model=None, user=None, expert_settings=None):
    from dataclasses import dataclass, field
    import json
    from multiprocessing import current_process
    import os
    import sys
    from typing import List, Generator, Type

    from calamari_ocr.ocr.dataset.datareader.base import (
        CalamariDataGenerator,
        CalamariDataGeneratorParams,
        SampleMeta,
        InputSample)
    from calamari_ocr.ocr.scenario import CalamariScenario
    from calamari_ocr.utils.image import to_uint8
    from paiargparse import pai_dataclass
    from PIL import Image
    import tensorflow.keras as keras
    from tfaip.data.pipeline.definitions import PipelineMode
    from tfaip.data.pipeline.datagenerator import DataGenerator
    from tfaip.util.tftyping import sync_to_numpy_or_python_type

    current_process().daemon = False

    # Load expert settings
    expert_settings = default_expert_settings() if expert_settings is None else expert_settings

    # try to minimize what is loaded in memory for large datasets
    ground_truth = list(qs.values('content',
                                  baseline=F('line__baseline'),
                                  external_id=F('line__external_id'),
                                  mask=F('line__mask'),
                                  image=F('line__document_part__image')))
    ground_truth = [lt for lt in ground_truth if lt['mask'] is not None]
    if not ground_truth:
        raise ValueError('No ground truth provided.')
    partition = int(len(ground_truth) / 10)
    traindata = {'images': [], 'texts': [], 'ids': []}
    imagepath = ""
    image = Image.Image()

    for lt in ground_truth:
        if imagepath != os.path.join(settings.MEDIA_ROOT, lt['image']):
            imagepath = os.path.join(settings.MEDIA_ROOT, lt['image'])
            image = Image.open(imagepath)
        traindata['images'].append(crop_image(image, lt['mask'], expert_settings.crop_polygon,
                            expert_settings.crop_smoothing, expert_settings.crop_background))
        traindata['texts'].append(lt['content'])
        traindata['ids'].append(lt['external_id'])
    image.close()

    # Convert to uint8
    traindata['images'] = [to_uint8(np.asarray(img.convert('L'))) for img in traindata['images']]

    # Shuffle lists
    np.random.default_rng(241960353267317949653744176059648850006).shuffle(traindata['images'])
    np.random.default_rng(241960353267317949653744176059648850006).shuffle(traindata['texts'])
    np.random.default_rng(241960353267317949653744176059648850006).shuffle(traindata['ids'])
    p = CalamariScenario.default_trainer_params()

    @pai_dataclass
    @dataclass
    class eScriptoriumDataReaderParams(CalamariDataGeneratorParams):
        images: List[str] = field(default_factory=list)
        texts: List[str] = field(default_factory=list)
        ids: List[str] = field(default_factory=list)
        preload = True
        skip_invalid = False
        def to_prediction(self):
            raise NotImplementedError

        def __len__(self):
            return len(self.images)
        @staticmethod
        def cls() -> Type["DataGenerator"]:
            return eScriptoriumDataReader

    class eScriptoriumDataReader(CalamariDataGenerator[eScriptoriumDataReaderParams]):
        def __init__(self, mode: PipelineMode, params: eScriptoriumDataReaderParams):
            """Create a dataset from memory"""
            super().__init__(mode, params)
            self.chars = 0
            for image, text, id in zip(params.images, params.texts, params.ids):
                meta = SampleMeta(id)
                self.add_sample(
                    {
                        "image": image,
                        "text": text,
                        "id": meta.id,
                        "meta": meta,
                    }
                )
                self.chars += len(text)
            self.loaded = True
        def _load_sample(self, sample, text_only) -> Generator[InputSample, None, None]:
            if text_only:
                yield InputSample(None, sample["text"], sample["meta"])
            yield InputSample(sample["image"], sample["text"], sample["meta"])

    p.gen.train = eScriptoriumDataReaderParams(images=traindata['images'][partition:],
                                      texts=traindata['texts'][partition:],
                                      ids=traindata['ids'][partition:])

    p.gen.val = eScriptoriumDataReaderParams(images=traindata['images'][:partition],
                                      texts=traindata['texts'][:partition],
                                      ids=traindata['ids'][:partition])
    p.preload = True
    p.gen.setup.val.batch_size = 1
    p.gen.setup.val.num_processes = 1
    p.gen.setup.train.batch_size = 1
    p.gen.setup.train.num_processes = 1
    p.epochs = 10
    p.samples_per_epoch = 1
    p.scenario.data.__post_init__()
    p.scenario.__post_init__()
    p.__post_init__()

    model_dir = os.path.join(settings.MEDIA_ROOT, os.path.split(model.file.path)[0])

    # create the trainer and run it
    trainer = p.scenario.cls().create_trainer(p)

    class TrainerCheckpointsCallback(keras.callbacks.ModelCheckpoint):
        """
        Callback to store the current state of the trainer params and the current training model with
        all of its weights which is required for resuming the training.

        This is realized by reimplementing some of the methods of the keras ModelCheckpoint-Callback which is a base class.
        """

        def __init__(self, train_params, save_freq=None, store_weights=True, store_params=True):
            # before the actual ModelCheckpoint base is initialized, determine the ckpt_dir
            logs = None
            log_dir = train_params.output_dir
            if log_dir and train_params.checkpoint_sub_dir:
                log_dir = os.path.join(log_dir, train_params.checkpoint_sub_dir)

            ckpt_dir = os.path.join(log_dir, "variables", "variables") if log_dir else ""

            if ckpt_dir is None or save_freq is None:
                save_freq = sys.maxsize  # never

            # Init of the parent
            super().__init__(
                ckpt_dir,
                save_weights_only=True,
                save_freq=save_freq,
            )

            # Set additional members
            self.log_dir = log_dir
            self.train_params = train_params
            self.store_weights = store_weights
            self.store_params = store_params

        def _print_eval(self, epoch=0, accuracy=0, chars=0, error=0, val_metric=0):
            model.refresh_from_db()
            model.training_epoch = epoch
            model.training_accuracy = float(accuracy)
            model.training_total = int(chars)
            model.training_errors = int(error)
            relpath = os.path.relpath(model_dir, settings.MEDIA_ROOT)
            model.new_version(file=f'{relpath}/version_{epoch}.calamarimodel')
            model.save()

            send_event('document', document.pk, "training:eval", {
                "id": model.pk,
                'versions': model.versions,
                'epoch': epoch,
                'accuracy': float(accuracy),
                'chars': int(chars),
                'error': int(error)})

        def _save_model(self, epoch, logs):
            # Override save model to either store weights or the params
            if self.store_params:
                if isinstance(self.save_freq, int) or self.epochs_since_last_save >= self.period:
                    logs = sync_to_numpy_or_python_type(logs)
                    # Crop variables/variables of the path
                    filepath = os.path.abspath(os.path.join(self._get_file_path(epoch, logs), "..", ".."))
                    self.train_params.current_epoch = epoch + 1
                    self.train_params.saved_checkpoint_sub_dir = os.path.relpath(filepath, self.train_params.output_dir)
                    self._save_params(filepath)

            if self.store_weights:
                super()._save_model(epoch, logs)

        def _save_params(self, filepath):
            # Save the params only
            os.makedirs(filepath, exist_ok=True)
            path = os.path.join(filepath, "trainer_params.json")
            logger.debug(f"Logging current trainer state to '{path}'")
            with open(path, "w") as f:
                json.dump(self.train_params.to_dict(), f, indent=2)

        def on_epoch_end(self, epoch, logs=None):
            # we ended an epoch, store that we ended it
            self.train_params.current_epoch = epoch + 1
            super().on_epoch_end(epoch, logs)
            if self.save_freq == 'epoch':
                self._save_model(epoch=epoch, logs=logs)
                self._print_eval(epoch=epoch+1,
                                 accuracy=int(logs.get('val_CER', 0)),
                                 chars=int(self.train_params.gen.train.chars*(epoch+1)),
                                 error=int(self.train_params.gen.train.chars*(epoch+1)*logs.get('CER', 1)),
                                 val_metric=0)

    checkpoint_callback = TrainerCheckpointsCallback(p,
                                                     save_freq='epoch',
                                                     store_weights=True,
                                                     store_params=True)
    trainer.train(callbacks=[checkpoint_callback])


def train_tesseract(qs, document, transcription, model=None, user=None, expert_settings=None):
    # # Note hack to circumvent AssertionError: daemonic processes are not allowed to have children
    from collections import Counter
    from multiprocessing import current_process
    from hashlib import sha256
    import fcntl
    from datetime import date
    from pathlib import Path
    import subprocess
    import shlex
    import time
    import pickle

    from PIL import Image
    from tesserocr import PyTessBaseAPI
    from bs4 import BeautifulSoup
    from urllib.request import urlopen

    def clean_old_ground_truth(gt_path, expiration_timedelta_days=7):
        # Clean the files if they are X (default: 7) days and not used or a half year (locking files is not perfect)
        locked_files = subprocess.run(shlex.split(f"lsof +D {gt_path.resolve()}"), stdout=subprocess.PIPE,
                             stderr=subprocess.DEVNULL).stdout
        locked_files = ['/' + line.split(b' /')[-1].decode('utf-8') for line in locked_files.split(b'\n') if
                        line.startswith(b'python')]
        for gt_file in gt_path.glob('*.lstmf'):
            if str(gt_file.resolve()) not in locked_files:
                if (time.time() - gt_file.stat().st_atime) > expiration_timedelta_days * 86400:
                    gt_file.unlink()
            elif (time.time() - gt_file.stat().st_atime) > 180 * 86400:
                gt_file.unlink()
        return

    def unlock_files(locked_files):
        while locked_files:
            locked_file = locked_files.pop()
            fcntl.flock(locked_file, fcntl.LOCK_UN)
            locked_file.close()
        return

    # Placeholder for up comming "Expert Settings"
    expert_settings = default_expert_settings() if expert_settings is None else expert_settings

    # Currently just checking if the model is trainable (best model)
    # TODO: Convert model from fast to best instead and than train with that
    if model.file.path:
        model_info = subprocess.check_output(shlex.split(f"combine_tessdata -l {model.file.file.name}"),
                                             stderr=subprocess.STDOUT)
        if b'int_mode=1' in model_info:
            if user:
                user.notify(_("Tesseract model is not trainable! Please use a best model!"),
                            id="training-error", level='danger')
                raise TypeError("The tesseract model has the wrong type. Please use only best model!")

    current_process().daemon = False
    # try to minimize what is loaded in memory for large datasets
    ground_truth = list(qs.values('content',
                                  baseline=F('line__baseline'),
                                  external_id=F('line__external_id'),
                                  script=F('line__document_part__document__main_script__text_direction'),
                                  mask=F('line__mask'),
                                  image=F('line__document_part__image')))
    # TODO: Automatically recalculating the mask if baseline exists
    ground_truth = [lt for lt in ground_truth if lt['mask'] is not None]
    if not ground_truth:
        raise ValueError('No ground truth provided.')

    trainings_dir = Path(settings.MEDIA_ROOT).joinpath(f"training/tesseract/")
    trainingsdata_dir = trainings_dir.joinpath('data')
    trainingsdata_dir.mkdir(parents=True, exist_ok=True)
    trainingsgt_dir = trainings_dir.joinpath('gt')
    trainingsgt_dir.mkdir(parents=True, exist_ok=True)
    locked_files = []
    try:
        imagepath = ""
        image = Image.Image()
        gt_list = []
        rtl_cnt = 0
        with PyTessBaseAPI(path=os.path.dirname(model.file.path),
                           lang=os.path.basename(model.file.path).rsplit('.', 1)[0],
                           psm=13) as api:
            for lt in ground_truth:
                filepath = trainingsgt_dir.joinpath(f"{lt['external_id']}_"
                                                 f"{sha256(lt['content'].encode('utf-8')).hexdigest()}.lstmf")
                gt_list.append(str(filepath.resolve()))
                if filepath.exists():
                    filepath.touch(mode=0o666, exist_ok=True)
                else:
                    if imagepath != os.path.join(settings.MEDIA_ROOT, lt['image']):
                        imagepath = os.path.join(settings.MEDIA_ROOT, lt['image'])
                        image = Image.open(imagepath)
                    vertical = 'vertical' in lt['script']
                    rtl_cnt += 'rl' in lt['script']
                    cutout = crop_image(image, lt['mask'],expert_settings.crop_polygon,
                                        expert_settings.crop_smoothing, expert_settings.crop_background)
                    api.WriteLSTMFLineData(filepath.with_suffix('').name, str(filepath.parent.resolve()), cutout,
                                       lt['content'], vertical)
                locked_files.append(filepath.open('r'))
                fcntl.flock(locked_files[-1], fcntl.LOCK_SH)
        image.close()

        # Delete old gt files
        clean_old_ground_truth(trainingsgt_dir)

        rtl = True if rtl_cnt > (len(gt_list)/2.25) else False
        # Randomize gt
        np.random.default_rng(241960353267317949653744176059648850006).shuffle(gt_list)

        # Calculate splitting point for training and evaluation set
        partition = int(len(ground_truth) / 10) if len(ground_truth) > 9 else 1

        # Get train and eval list
        eval_list = gt_list[:partition]
        train_list = gt_list[partition:]

        # Generate trainings id (startmodel_evallist_trainlist)
        model_dir = trainings_dir.joinpath('jobs').\
            joinpath(f"{slugify(model.name)}_{sha256(pickle.dumps(eval_list+train_list)).hexdigest()}").\
            joinpath(model.name)
        model_dir.mkdir(parents=True, exist_ok=True)

        evaluationlist_fpath = model_dir.joinpath('list.eval')
        evaluationlist_fpath.open('w').write('\n'.join(eval_list))
        trainingslist_fpath = model_dir.joinpath('list.train')
        trainingslist_fpath.open('w').write('\n'.join(train_list))

        # Get chars and occurences for train and eval files
        np.random.default_rng(241960353267317949653744176059648850005).shuffle(ground_truth)
        eval_cc = Counter('\n'.join([lt['content'] for lt in ground_truth[:partition]]))
        train_cc = Counter('\n'.join([lt['content'] for lt in ground_truth[partition:]]))
        avg_char_per_iteration = int(sum(train_cc.values())/(len(ground_truth)-partition))
        all_unicodes_fpath = model_dir.joinpath('all_unicodes')
        all_unicodes_fpath.open('w').write('\n'.join(train_cc.keys()))

        # Find difference in eval and train chars
        miss_trainingschar = set(eval_cc.keys()).difference(set(train_cc.keys()))
        miss_evaluationchar = set(train_cc.keys()).difference(set(eval_cc.keys()))

        if user:
            if any(miss_trainingschar):
                user.notify(_("There are chars in the evaluation set, which are not in the trainings set!"),
                            id="training-tesseract", level='warning')
            if any(miss_evaluationchar):
                user.notify(_("There are chars in the trainings set, which are not in the evaluation set!"),
                            id="training-tesseract", level='warning')

    except RuntimeError as e:
        print(f"An Error occrued and this line is skipped: \n")
        unlock_files(locked_files)
        pass

    def store_checkpoint_as_model(ckpt_path, model_path, start_model_path):
        try:
            logger.info(f"Converting checkpoint to model: {ckpt_path.relative_to(settings.MEDIA_ROOT)} "
                        f"-> {model_path.relative_to(settings.MEDIA_ROOT)}")
            subprocess.run(shlex.split(f"lstmtraining --stop_training "
                                       f"--continue_from {str(ckpt_path.absolute())} "
                                       f"--traineddata {str(start_model_path.absolute())} "
                                       f"--model_output {str(model_path.absolute())}"),
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            # Clean checkpoints
            ckpt_path.unlink()
        except:
            logger.error("Could not store checkpoint as model.")
            pass

    def get_necessary_datafiles(data_dir):

        # Get list of necessary files for training
        url = "https://github.com/tesseract-ocr/langdata_lstm/"
        with urlopen(url) as response:
            soup = BeautifulSoup(response.read(), "html.parser")

        datafiles = []
        for el in soup.find_all("a", class_="js-navigation-open Link--primary"):
            if '.' in el.text:
                datafiles.append(el.text)

        # Download necessary files
        for datafile in datafiles:
            subprocess.run(shlex.split(f"wget -P {str(data_dir.absolute())} "
                                       f"'https://github.com/tesseract-ocr/langdata_lstm/raw/main/{datafile}'"),
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        # Inherited unicharset (not included in the official tesseract trainingsdata)
        data_dir.joinpath('Inherited.unicharset').open('w').write("""53
NULL 0 Common 0
̀ 0 202,248,238,255,1,66,0,71,0,173 Inherited 1168 17 1168 ̀	# ̀ [300 ]
́ 0 174,197,216,232,17,29,0,0,0,0 Inherited 4 17 4 ́	# ́ [301 ]
̂ 0 174,197,214,232,12,23,0,0,0,0 Inherited 5 17 5 ̂	# ̂ [302 ]
̃ 0 179,202,206,227,22,32,0,0,0,0 Inherited 6 17 6 ̃	# ̃ [303 ]
̄ 0 183,205,197,222,15,27,0,0,0,0 Inherited 7 17 7 ̄	# ̄ [304 ]
̅ 0 186,205,200,224,30,33,0,0,0,0 Inherited 8 17 8 ̅	# ̅ [305 ]
̆ 0 174,198,213,229,21,32,0,0,0,0 Inherited 9 17 9 ̆	# ̆ [306 ]
̇ 0 202,202,227,227,8,8,0,0,0,0 Inherited 10 17 10 ̇	# ̇ [307 ]
̈ 0 177,202,202,227,17,27,0,0,0,0 Inherited 11 17 11 ̈	# ̈ [308 ]
̉ 0 176,176,227,227,21,21,0,0,0,0 Inherited 12 17 12 ̉	# ̉ [309 ]
̊ 0 174,195,221,236,5,15,0,0,0,0 Inherited 13 17 13 ̊	# ̊ [30a ]
̋ 0 174,197,216,232,3,39,0,0,0,0 Inherited 14 17 14 ̋	# ̋ [30b ]
̌ 0 174,174,214,214,22,22,0,0,0,0 Inherited 15 17 15 ̌	# ̌ [30c ]
̍ 0 177,177,225,225,14,14,0,0,0,0 Inherited 16 17 16 ̍	# ̍ [30d ]
̎ 0 177,177,225,225,29,29,0,0,0,0 Inherited 17 17 17 ̎	# ̎ [30e ]
̏ 0 177,197,220,232,3,40,0,0,0,0 Inherited 18 17 18 ̏	# ̏ [30f ]
̐ 0 177,177,228,228,38,38,0,0,0,0 Inherited 19 17 19 ̐	# ̐ [310 ]
̑ 0 177,198,216,229,28,32,0,0,0,0 Inherited 20 17 20 ̑	# ̑ [311 ]
̒ 0 181,181,234,234,28,28,0,0,0,0 Inherited 21 17 21 ̒	# ̒ [312 ]
̔ 0 195,195,232,232,30,30,104,104,0,0 Inherited 22 17 22 ̔	# ̔ [314 ]
̚ 0 177,177,234,234,38,38,0,0,0,0 Inherited 23 17 23 ̚	# ̚ [31a ]
̛ 0 137,162,193,212,1,15,0,0,0,0 Inherited 24 17 24 ̛	# ̛ [31b ]
̡ 0 7,7,72,72,12,12,0,0,0,0 Inherited 25 17 25 ̡	# ̡ [321 ]
̢ 0 7,7,72,72,21,21,0,0,0,0 Inherited 26 17 26 ̢	# ̢ [322 ]
̦ 0 0,0,44,44,20,20,53,53,126,126 Inherited 27 17 27 ̦	# ̦ [326 ]
̲ 0 25,25,42,42,30,30,0,0,0,0 Inherited 28 17 28 ̲	# ̲ [332 ]
̳ 0 0,0,42,42,30,30,0,0,0,0 Inherited 29 17 29 ̳	# ̳ [333 ]
̴ 0 125,125,151,151,5,21,0,0,0,0 Inherited 30 17 30 ̴	# ̴ [334 ]
̶ 0 106,106,118,118,7,7,0,0,0,0 Inherited 31 17 31 ̶	# ̶ [336 ]
̷ 0 85,85,139,139,10,10,0,0,0,0 Inherited 32 17 32 ̷	# ̷ [337 ]
̸ 0 71,71,153,153,5,5,0,0,0,0 Inherited 33 17 33 ̸	# ̸ [338 ]
̽ 0 177,177,234,234,35,35,0,0,0,0 Inherited 34 17 34 ̽	# ̽ [33d ]
̾ 0 174,174,235,235,22,22,0,0,0,0 Inherited 35 17 35 ̾	# ̾ [33e ]
̿ 0 186,186,227,227,40,40,0,0,0,0 Inherited 36 17 36 ̿	# ̿ [33f ]
́ 0 222,222,255,255,1,1,0,0,0,0 Inherited 37 17 37 ́	# ́ [341 ]
͂ 0 199,199,223,223,19,19,0,0,0,0 Inherited 38 17 38 ͂	# ͂ [342 ]
͠ 0 176,176,221,225,99,124,0,0,0,0 Inherited 39 17 39 ͠	# ͠ [360 ]
͡ 0 176,214,221,246,92,143,0,0,0,0 Inherited 40 17 40 ͡	# ͡ [361 ]
҅ 0 177,192,209,244,67,85,0,27,85,123 Inherited 41 17 41 ҅	# ҅ [485 ]
҆ 0 177,192,209,244,67,85,0,27,85,123 Inherited 42 17 42 ҆	# ҆ [486 ]
ً 0 178,178,232,232,70,70,29,29,0,0 Inherited 43 17 43 ً	# ً [64b ]
ٌ 0 178,178,235,235,94,94,29,29,0,0 Inherited 44 17 44 ٌ	# ٌ [64c ]
ٍ 0 0,22,38,66,39,70,7,29,0,58 Inherited 45 17 45 ٍ	# ٍ [64d ]
َ 0 178,178,207,207,70,70,29,29,0,0 Inherited 46 17 46 َ	# َ [64e ]
ُ 0 178,178,238,238,74,74,29,29,0,0 Inherited 47 17 47 ُ	# ُ [64f ]
ِ 0 0,22,15,50,39,70,7,29,0,58 Inherited 48 17 48 ِ	# ِ [650 ]
ّ 0 178,178,238,238,74,74,29,29,0,0 Inherited 49 17 49 ّ	# ّ [651 ]
ْ 0 178,178,238,238,53,53,29,29,0,0 Inherited 50 17 50 ْ	# ْ [652 ]
ٰ 0 178,178,227,227,12,12,29,29,0,0 Inherited 51 17 51 ٰ	# ٰ [670 ]
॒ 0 12,25,24,109,1,5,0,0,0,2 Inherited 52 17 52 ॒	# ॒ [952 ]
⃝ 0 57,57,225,225,170,170,7,7,184,184 Inherited 53 17 53 ⃝	# ⃝ [20dd ]
゙ 0 210,221,228,230,43,46,133,157,193,205 Inherited 54 17 54 ゙	# ゙ [3099 ]""")
        return

    def _train_tesseract(*, learning_rate=0.0001, max_iterations=10000, max_model_timedelta=10000,
                         min_target_error_rate=1.0, target_error_rate=0.01, norm_mode=2,
                        net_spec="[1,36,0,1 Ct3,3,16 Mp3,3 Lfys48 Lfx96 Lrx96 Lfx192 O1c\#\#\#]"):
        """runs preprocessing and tesseract training with subprocess"""

        def _print_eval(ckpt_path, epoch="0", accuracy="0", chars=0, error=0, val_metric=0):
            logger.info(f"Wrote checkpoint:   {ckpt_path}\n"
                        f"Current iterations: {epoch}\n"
                        f"Current accuracy:   {accuracy}")

        # Radical stroke (for proto model creation)
        radical_stroke = trainingsdata_dir.joinpath("radical-stroke.txt")
        if not radical_stroke.exists():
            get_necessary_datafiles(trainingsdata_dir)

        proto_model = Path(model_dir.joinpath(model.name + '.traineddata'))
        model_dir.joinpath('checkpoints').mkdir(exist_ok=True)
        training_cmd = f"lstmtraining " \
                       f"--debug_interval 0 " \
                       f"--traineddata {str(proto_model.absolute())} " \
                       f"--learning_rate {learning_rate} " \
                       f"--model_output {str(model_dir.absolute())} " \
                       f"--train_listfile {str(trainingslist_fpath.absolute())} " \
                       f"--eval_listfile {str(evaluationlist_fpath.absolute())} " \
                       f"--max_iterations {max_iterations} " \
                       f"--target_error_rate {target_error_rate} "

        if model.file:
            # TODO: Convert fast model to best if b'int=1' ==  'combine_tessdata - l {start_model}`
            # Finetune a start model
            # unpack start model
            if not model_dir.joinpath(model.name + '.lstm').exists():
                subprocess.run(shlex.split(f"combine_tessdata "
                                               f"-u {model.file.file.name} "
                                               f"{str(model_dir.joinpath(model.name).absolute())}"),
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL )

            # create unicharset
            if not model_dir.joinpath('my.unicharset').exists():
                subprocess.run(shlex.split(f"unicharset_extractor "
                                           f"--output_unicharset "
                                           f"{str(model_dir.joinpath('my.unicharset').absolute())} "
                                           f"--norm_mode "
                                           f"{norm_mode} {str(all_unicodes_fpath.absolute())}"),
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                subprocess.run(shlex.split(f"merge_unicharsets "
                                           f"{str(model_dir.joinpath(model.name + '.lstm-unicharset').absolute())} "
                                           f"{str(model_dir.joinpath('my.unicharset').absolute())} "
                                           f"{str(model_dir.joinpath('unicharset').absolute())}"),
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            # Training cmd
            training_cmd = training_cmd + f"--old_traineddata " \
                                          f"{model.file.file.name} " \
                                          f"--continue_from " \
                                          f"{str(model_dir.joinpath(model.name + '.lstm').absolute())}"

        else:
            # Create a placeholder model
            filename = slugify(model.name) + '.traineddata'
            model.file = model.file.field.upload_to(model, filename)
            model.save()

            # Create new model
            subprocess.run(shlex.split(f"unicharset_extractor "
                                       f"--output_unicharset "
                                       f"{str(model_dir.joinpath('my.unicharset').absolute())} "
                                       f"--norm_mode "
                                       f"{norm_mode} {str(all_unicodes_fpath.absolute())}"),
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            net_spec = net_spec.replace('\#\#\#', str(len(train_cc)))
            training_cmd = training_cmd + f"--net_spec {net_spec})"

        # Create proto model (neural net not included)
        if not proto_model.exists():
            proto_cmd = "combine_lang_model "
            proto_cmd += "--lang_is_rtl " if rtl else ""
            proto_cmd += "--pass_through_recoder " if norm_mode > 2 else ""
            proto_cmd += f"--input_unicharset " \
                         f"{str(model_dir.joinpath('unicharset').absolute())} " \
                         f"--lang {model.name} " \
                         f"--script_dir {str(trainingsdata_dir.absolute())} " \
                         f'--version_str "Trained at {date.today()} with eScriptorium." ' \
                         f"--output_dir {str(model_dir.parent.absolute())}"
            """
            TODO: Optional files (not yet included), maybe use the files of the start model?
                                       f"--words {model_name}.wordlist "
                                       f"--numbers {model_name}.numbers "
                                       f"--puncs {model_name}.punc "))
            """
            subprocess.run(shlex.split(proto_cmd), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        # Start training
        logger.info(training_cmd)
        process = subprocess.Popen(shlex.split(training_cmd), stderr=subprocess.PIPE)

        # Read process output on the fly and create new models automatically
        ckpt_path, current_bestmodel_path = Path(), Path()
        time_last_model = time.time()
        for c in iter(lambda: process.stderr.readline(), b""):
            text = c.decode()
            logger.info(text)
            if "wrote best model" in text:
                # Delete second last checkpoint if it exists
                if ckpt_path.name.endswith('checkpoint') and ckpt_path.exists() and current_bestmodel_path.exists():
                    ckpt_path.resolve().unlink()
                ckpt_path = Path(text.split('best model:')[1].split(' ', 1)[0])
                accuracy = text.split('=')[3].split(',', 1)[0]
                iterations = text.split('/', 2)[2].split(',', 1)[0]
                _print_eval(str(ckpt_path), iterations, accuracy, 0, 0)
                # -> Convert checkpoint to best model and save it
                epoch = int(int(iterations)/100)
                current_bestmodel_path = Path(model.file.path).parent.joinpath(f'version_{epoch}.traineddata')
                current_bestmodel_relpath = current_bestmodel_path.relative_to(settings.MEDIA_ROOT)
                store_checkpoint_as_model(ckpt_path,
                                  current_bestmodel_path,
                                  Path(model.file.file.name))
                # Add model to eScriptorium database
                model.refresh_from_db()
                model.training_epoch = epoch
                model.training_accuracy = (100.0-float(accuracy[:-1]))/100
                model.training_total = avg_char_per_iteration*int(iterations)
                model.training_errors = int((1-model.training_accuracy)*model.training_total)
                model.new_version(file=str(current_bestmodel_relpath))
                model.save()
                send_event('document', document.pk, "training:eval", {
                    'id': model.pk,
                    'versions': model.versions,
                    'epoch': epoch,
                    'accuracy': model.training_accuracy,
                    'chars': model.training_total,
                    'error': model.training_errors})
                if (1-model.training_accuracy)*100 < min_target_error_rate:
                    if max_model_timedelta > 0 and time_last_model < time.time()+max_model_timedelta:
                        process.kill()
                        return
                time_last_model = time.time()
            if "is an integer (fast) model, cannot continue training" in text:
                user.notify(_("Only tesseracts best models can be trained!"),
                            id="training-tesseract", level='warning')
                process.kill()
                # Delete stuff
                Path(model.file.file.name).unlink()
                model.delete()
                raise ValueError
    try:
        _train_tesseract(learning_rate=0.0001, max_iterations=10000, max_model_timedelta=10,
                         min_target_error_rate=0.0, target_error_rate=0.01, norm_mode=2,
                         net_spec="[1,36,0,1 Ct3,3,16 Mp3,3 Lfys48 Lfx96 Lrx96 Lfx192 O1c\#\#\#]")
    except:
        pass
    unlock_files(locked_files)

@shared_task(bind=True, autoretry_for=(MemoryError,), default_retry_delay=60 * 60)
def train(task, transcription_pk, model_pk=None, part_pks=None, user_pk=None, **kwargs):
    if user_pk:
        try:
            user = User.objects.get(pk=user_pk)
            # If quotas are enforced, assert that the user still has free CPU minutes, GPU minutes and disk storage
            if not settings.DISABLE_QUOTAS:
                if user.cpu_minutes_limit() is not None:
                    assert user.has_free_cpu_minutes(), f"User {user.id} doesn't have any CPU minutes left"
                if user.gpu_minutes_limit() is not None:
                    assert user.has_free_gpu_minutes(), f"User {user.id} doesn't have any GPU minutes left"
                if user.disk_storage_limit() is not None:
                    assert user.has_free_disk_storage(), f"User {user.id} doesn't have any disk storage left"
        except User.DoesNotExist:
            user = None
    else:
        user = None

    redis_.set('training-%d' % model_pk, json.dumps({'task_id': task.request.id}))

    Transcription = apps.get_model('core', 'Transcription')
    LineTranscription = apps.get_model('core', 'LineTranscription')
    OcrModel = apps.get_model('core', 'OcrModel')

    try:
        model = OcrModel.objects.get(pk=model_pk)
        model.training = True
        model.save()
        transcription = Transcription.objects.get(pk=transcription_pk)
        document = transcription.document
        send_event('document', document.pk, "training:start", {
            "id": model.pk,
        })
        qs = (LineTranscription.objects
              .filter(transcription=transcription,
                      line__document_part__pk__in=part_pks)
              .exclude(Q(content='') | Q(content=None)))
        {"calamari": train_calamari,
         "kraken": train_kraken,
         "tesseract": train_tesseract}.get(model.engine, train_kraken)(qs, document, transcription,
                                                                       model=model, user=user)
    except Exception as e:
        send_event('document', document.pk, "training:error", {
            "id": model.pk,
        })
        if user:
            user.notify(_("Something went wrong during the training process!"),
                        id="training-error", level='danger')
        logger.exception(e)
    else:
        if user:
            user.notify(_("Training finished!"),
                        id="training-success",
                        level='success')
    finally:
        model.training = False
        model.file_size = model.file.size
        model.save()

        send_event('document', document.pk, "training:done", {
            "id": model.pk,
        })


@shared_task(autoretry_for=(MemoryError,), default_retry_delay=10 * 60)
def transcribe(instance_pk=None, model_pk=None, user_pk=None, text_direction=None, **kwargs):

    try:
        DocumentPart = apps.get_model('core', 'DocumentPart')
        part = DocumentPart.objects.get(pk=instance_pk)
    except DocumentPart.DoesNotExist:

        logger.error('Trying to transcribe non-existent DocumentPart : %d', instance_pk)
        return

    if user_pk:
        try:
            user = User.objects.get(pk=user_pk)
            # If quotas are enforced, assert that the user still has free CPU minutes
            if not settings.DISABLE_QUOTAS and user.cpu_minutes_limit() is not None:
                assert user.has_free_cpu_minutes(), f"User {user.id} doesn't have any CPU minutes left"
        except User.DoesNotExist:
            user = None
    else:
        user = None

    try:
        OcrModel = apps.get_model('core', 'OcrModel')
        model = OcrModel.objects.get(pk=model_pk)
        part.transcribe(model)
    except Exception as e:
        if user:
            user.notify(_("Something went wrong during the transcription!"),
                        id="transcription-error", level='danger')
        part.workflow_state = part.WORKFLOW_STATE_SEGMENTED
        part.save()
        logger.exception(e)
        raise e
    else:
        if user and model:
            user.notify(_("Transcription done!"),
                        id="transcription-success",
                        level='success')


def check_signal_order(old_signal, new_signal):
    SIGNAL_ORDER = ['before_task_publish', 'task_prerun', 'task_failure', 'task_success']
    return SIGNAL_ORDER.index(old_signal) < SIGNAL_ORDER.index(new_signal)


@before_task_publish.connect
def before_publish_state(sender=None, body=None, **kwargs):
    if sender not in STATE_TASKS:
        return
    instance_id = body[1]["instance_pk"]
    data = json.loads(redis_.get('process-%d' % instance_id) or '{}')

    signal_name = kwargs['signal'].name

    try:
        # protects against signal race condition
        if (data[sender]['task_id'] == sender.request.id
                and not check_signal_order(data[sender]['status'], signal_name)):
            return
    except (KeyError, AttributeError):
        pass

    data[sender] = {
        "task_id": kwargs['headers']['id'],
        "status": 'before_task_publish'
    }
    redis_.set('process-%d' % instance_id, json.dumps(data))
    try:
        update_client_state(instance_id, sender, 'pending')
    except NameError:
        pass


@task_prerun.connect
@task_success.connect
@task_failure.connect
def done_state(sender=None, body=None, **kwargs):
    if sender.name not in STATE_TASKS:
        return
    instance_id = sender.request.kwargs["instance_pk"]

    try:
        data = json.loads(redis_.get('process-%d' % instance_id) or '{}')
    except TypeError as e:
        logger.exception(e)
        return

    signal_name = kwargs['signal'].name

    try:
        # protects against signal race condition
        if (data[sender.name]['task_id'] == sender.request.id
                and not check_signal_order(data[sender.name]['status'], signal_name)):
            return
    except KeyError:
        pass

    data[sender.name] = {
        "task_id": sender.request.id,
        "status": signal_name
    }
    status = {
        'task_success': 'done',
        'task_failure': 'error',
        'task_prerun': 'ongoing'
    }[signal_name]
    if status == 'error':
        # remove any pending task down the chain
        data = {k: v for k, v in data.items() if v['status'] != 'pending'}
    redis_.set('process-%d' % instance_id, json.dumps(data))

    if status == 'done':
        result = kwargs.get('result', None)
    else:
        result = None
    update_client_state(instance_id, sender.name, status, task_id=sender.request.id, data=result)
