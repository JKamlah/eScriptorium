import json
import logging
import os.path
import subprocess
import redis
from PIL import Image

from django.apps import apps
from django.conf import settings
from django.contrib.auth import get_user_model
from django.core.files import File
from django.utils.translation import gettext as _

from celery import shared_task
from celery.signals import *
from easy_thumbnails.files import generate_all_aliases
from kraken import binarization, pageseg

from users.consumers import send_event


logger = logging.getLogger(__name__)
User = get_user_model()
redis_ = redis.Redis(host=settings.REDIS_HOST, port=settings.REDIS_PORT, db=0)


@shared_task
def generate_part_thumbnails(instance_pk):
    try:
        DocumentPart = apps.get_model('core', 'DocumentPart')
        part = DocumentPart.objects.get(pk=instance_pk)
    except DocumentPart.DoesNotExist:
        logger.error('Trying to compress innexistant DocumentPart : %d', instance_pk)
        raise
    
    generate_all_aliases(part.image, include_global=True)


def update_client_state(part_id, task, status):
    if task == 'core.tasks.generate_part_thumbnails':
        return
    try:
        DocumentPart = apps.get_model('core', 'DocumentPart')
        part = DocumentPart.objects.get(pk=part_id)
    except DocumentPart.DoesNotExist:
        logger.error('Trying to segment innexistant DocumentPart : %d', instance_pk)
        raise
    task_name = task.split('.')[-1]
    send_event('document', part.document.pk, "part:workflow", {
        "id": part.pk,
        "process": 'binarize' if task_name == 'lossless_compression' else task_name,
        "status": status
    })


@shared_task(bind=True)
def lossless_compression(self, instance_pk):
    try:
        DocumentPart = apps.get_model('core', 'DocumentPart')
        part = DocumentPart.objects.get(pk=instance_pk)
    except DocumentPart.DoesNotExist:
        logger.error('Trying to compress innexistant DocumentPart : %d', instance_pk)
        raise

    if part.workflow_state < part.WORKFLOW_STATE_COMPRESSING:
        part.workflow_state = part.WORKFLOW_STATE_COMPRESSING
        part.save()
        
    convert = False
    old_name = part.image.file.name
    filename, extension = os.path.splitext(old_name)
    if extension != ".png":
        convert = True
        new_name = filename + ".png"
        error = subprocess.check_call(["convert", old_name, new_name])
        if error:
            raise RuntimeError("Error trying to convert file(%s) to png.")
    else:
        new_name = old_name
    opti_name = filename + '_opti.png'
    # Note: leave it fail it's fine
    try:
        subprocess.check_call(["pngcrush", "-q", new_name, opti_name])
    except Exception as e:
        logger.exception("png optimization failed.")
    os.rename(opti_name, new_name)
    if convert:
        part.image = new_name.split(settings.MEDIA_ROOT)[1][1:]
        os.remove(old_name)
    if part.workflow_state < part.WORKFLOW_STATE_COMPRESSED:
        part.workflow_state = part.WORKFLOW_STATE_COMPRESSED
        part.save()


@shared_task
def binarize(instance_pk, user_pk=None, binarizer=None):
    try:
        DocumentPart = apps.get_model('core', 'DocumentPart')
        part = DocumentPart.objects.get(pk=instance_pk)
    except DocumentPart.DoesNotExist:
        logger.error('Trying to binarize innexistant DocumentPart : %d', instance_pk)
        raise
    
    if user_pk:
        try:
            user = User.objects.get(pk=user_pk)
        except User.DoesNotExist:
            user = None
    else:
        user = None
    
    try:
        if part.workflow_state < part.WORKFLOW_STATE_BINARIZING:
            part.workflow_state = part.WORKFLOW_STATE_BINARIZING
            part.save()
        
        fname = os.path.basename(part.image.file.name)
        # should be formated to png already by by lossless_compression but better safe than sorry
        form = None
        f_, ext = os.path.splitext(part.image.file.name)
        if ext[1] in ['.jpg', '.jpeg', '.JPG', '.JPEG', '']:
            if ext:
                logger.warning('jpeg does not support 1bpp images. Forcing to png.')
            form = 'png'
            fname = '%s.%s' % (f_, form)
        bw_file_name = 'bw_' + fname
        bw_file = os.path.join(os.path.dirname(part.image.file.name), bw_file_name)
        with Image.open(part.image.path) as im:
            # threshold, zoom, escale, border, perc, range, low, high
            res = binarization.nlbin(im)
            res.save(bw_file, format=form)
    except:
        if user:
            user.notify(_("Something went wrong during the binarization!"),
                        id="binarization-error", level='danger')
        part.workflow_state = part.WORKFLOW_STATE_CREATED
        part.save()
        raise
    else:
        if user:
            user.notify(_("Binarization done!"),
                        id="binarization-success", level='success')
        
        from core.models import document_images_path  # meh
        part.bw_image = document_images_path(part, bw_file_name)
        if part.workflow_state < part.WORKFLOW_STATE_BINARIZED:
            part.workflow_state = part.WORKFLOW_STATE_BINARIZED
            part.save()


@shared_task
def segment(instance_pk, user_pk=None, text_direction=None):
    try:
        DocumentPart = apps.get_model('core', 'DocumentPart')
        part = DocumentPart.objects.get(pk=instance_pk)
    except DocumentPart.DoesNotExist:
        logger.error('Trying to segment innexistant DocumentPart : %d', instance_pk)
        raise

    if user_pk:
        try:
            user = User.objects.get(pk=user_pk)
        except User.DoesNotExist:
            user = None
    else:
        user = None
    
    try:
        Block = apps.get_model('core', 'Block')
        Line = apps.get_model('core', 'Line')
        # cleanup pre-existing
        part.lines.all().delete()
        Block.objects.filter(document_part=part).delete()
        
        part.workflow_state = part.WORKFLOW_STATE_SEGMENTING
        part.save()
        
        with Image.open(part.bw_image.file.name) as im:
            # text_direction='horizontal-lr', scale=None, maxcolseps=2, black_colseps=False, no_hlines=True, pad=0
            options = {}
            if text_direction:
                options['text_direction'] = text_direction
            res = pageseg.segment(im, **options)
            # if script_detect:
            #     res = pageseg.detect_scripts(im, res, valid_scripts=allowed_scripts)
            for line in res['boxes']:
                # block = Block.objects.create(document_part=part)
                # for line in block_:
                Line.objects.create(
                    document_part=part,
                    #     # block=block,
                    #     # script=selfcript,
                    box=line)
    except:
        if user:
            user.notify(_("Something went wrong during the segmentation!"),
                        id="segmentation-error", level='danger')
        part.workflow_state = part.WORKFLOW_STATE_BINARIZED
        part.save()
        raise
    else:
        if user:
            user.notify(_("Segmentation done!"),
                        id="segmentation-success", level='success')
        part.workflow_state = part.WORKFLOW_STATE_SEGMENTED
        part.save()


@shared_task
def transcribe(instance_pk, user_pk=None):
    try:
        DocumentPart = apps.get_model('core', 'DocumentPart')
        part = DocumentPart.objects.get(pk=instance_pk)
    except DocumentPart.DoesNotExist:
        logger.error('Trying to transcribe innexistant DocumentPart : %d', instance_pk)
        raise

    if user_pk:
        try:
            user = User.objects.get(pk=user_pk)
        except User.DoesNotExist:
            user = None
    else:
        user = None

    Transcription = apps.get_model('core', 'Transcription')
    Transcription.objects.get_or_create(name='default', document=part.document)
    part.workflow_state = part.WORKFLOW_STATE_TRANSCRIBING
    part.save()
    
    # model = part.select_model()
    #if model:
    #    TODO: kraken ocr


@before_task_publish.connect
def before_publish_state(sender=None, body=None, **kwargs):
    instance_id = body[0][0]
    data = json.loads(redis_.get('process-%d' % instance_id) or '{}')
    data[sender] = {
        "task_id": kwargs['headers']['id'],
        "status": 'before_task_publish'
    }
    redis_.set('process-%d' % instance_id, json.dumps(data))
    update_client_state(instance_id, sender, 'pending')


@task_prerun.connect
@task_success.connect
@task_failure.connect
def done_state(sender=None, body=None, **kwargs):
    instance_id = sender.request.args[0]
    data = json.loads(redis_.get('process-%d' % instance_id) or '{}')
    signal_name = kwargs['signal'].name
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
        data = {k:v for k,v in data.items() if v['status'] != 'pending'}
    
    redis_.set('process-%d' % instance_id, json.dumps(data))    
    update_client_state(instance_id, sender.name, status)
