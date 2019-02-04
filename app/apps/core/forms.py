import json

from django import forms
from django.db.models import Q
from django.forms.models import inlineformset_factory
from django.utils.functional import cached_property
from django.utils.translation import gettext as _

from bootstrap.forms import BootstrapFormMixin
from core.models import *


class DocumentForm(BootstrapFormMixin, forms.ModelForm):
    def __init__(self, *args, **kwargs):
        self.request = kwargs.pop('request')
        super().__init__(*args, **kwargs)
        
    class Meta:
        model = Document
        fields = ['name', 'typology']


class DocumentShareForm(BootstrapFormMixin, forms.ModelForm):
    def __init__(self, *args, **kwargs):
        self.request = kwargs.pop('request')
        super().__init__(*args, **kwargs)
        self.fields['shared_with_groups'].widget = forms.CheckboxSelectMultiple()
        self.fields['shared_with_groups'].queryset = self.request.user.groups
    
    class Meta:
        model = Document
        fields = ['shared_with_groups']  # shared_with_users


class MetadataForm(BootstrapFormMixin, forms.ModelForm):
    class Meta:
        model = DocumentMetadata
        fields = '__all__'
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        attrs = self.fields['key'].widget.attrs
        # attrs.update({'autocomplete':"off", 'list': "metadataKeys"})
        attrs['class'] += ' input-group-text px-5'
        self.fields['key'].empty_label = '-'
        self.fields['key'].widget.need_label = False


MetadataFormSet = inlineformset_factory(Document, DocumentMetadata, form=MetadataForm,
                                        extra=1, can_delete=True)

class DocumentPartUpdateForm(forms.ModelForm):
    index = forms.IntegerField(required=False, min_value=0)
    blocks = forms.CharField(required=False)
    lines = forms.CharField(required=False)
    
    class Meta:
        model = DocumentPart
        fields = ('name', 'typology', 'index')
        
    def save(self, *args, **kwargs):
        self.created = None
        if 'index' in self.cleaned_data and self.cleaned_data['index'] is not None:
            self.instance.to(self.cleaned_data['index'])
        
        # TODO: reassign lines
        if 'blocks' in self.cleaned_data and self.cleaned_data['blocks']:
            blocks = json.loads(self.cleaned_data['blocks'])
            for block_ in blocks:
                if block_['pk'] is None:
                    block = Block.objects.create(document_part=self.instance,
                                         box = block_['box'])
                    self.created = block
                else:
                    block = Block.objects.get(pk=block_['pk'])
                    if 'delete' in block_ and block_['delete'] is True:
                        block.delete()
                    else:
                        block.box = block_['box']
                        block.save()
        
        # TODO: find block + recalculate ordering of lines
        if 'lines' in self.cleaned_data and self.cleaned_data['lines']:
            lines = json.loads(self.cleaned_data['lines'])
            for line_ in lines:
                if line_['pk'] is None:
                    if 'block' in line_ and line_['block']:
                        block = Block.objects.get(pk=line_['block'])
                    else:
                        block = None
                    Line.objects.create(document_part=self.instance,
                                        block=block,
                                        box = line_['box'])
                    self.created = block
                else:
                    line = Line.objects.get(pk=line_['pk'])
                    if 'delete' in line_ and line_['delete'] is True:
                        line.delete()
                    else:
                        line.box = line_['box']
                        line.save()
        
        self.instance.recalculate_ordering()
        
        return super().save(*args, **kwargs)


class DocumentProcessForm(BootstrapFormMixin, forms.ModelForm):
    task = forms.ChoiceField(choices=(
        ('binarize', 1),
        ('segment', 2),
        ('train', 3),
        ('transcribe', 4)))
    parts = forms.CharField()
    bw_image = forms.ImageField(required=False)
    segmentation_steps = forms.ChoiceField(choices=(
        ('regions', _('Regions')),
        ('lines', _('Lines')),
        ('both', _('Lines and regions'))
    ), initial='both', required=False)
    new_model = forms.CharField(required=False, label=_('Name'))
    upload_model = forms.FileField(required=False)
    
    class Meta:
        model = DocumentProcessSettings
        fields = '__all__'
    
    def __init__(self, document, user, *args, **kwargs):
        self.user = user
        self.document = document
        super().__init__(*args, **kwargs)
        # self.fields['typology'].widget = forms.HiddenInput()  # for now
        # self.fields['typology'].initial = Typology.objects.get(name="Page")
        # self.fields['typology'].widget.attrs['title'] = _("Default Typology")
        self.fields['binarizer'].widget.attrs['disabled'] = True
        self.fields['binarizer'].required = False
        self.fields['text_direction'].required = False
        self.fields['train_model'].queryset = OcrModel.objects.filter(document=self.document)
        self.fields['ocr_model'].queryset = OcrModel.objects.filter(
            Q(document=None) | Q(document=self.document), trained=True)
    
    @cached_property
    def parts(self):
        pks = json.loads(self.data.get('parts'))
        parts = DocumentPart.objects.filter(document=self.document, pk__in=pks)
        return parts
    
    def clean_bw_image(self):
        img = self.cleaned_data.get('bw_image')
        if not img:
            return
        if len(self.parts) != 1:
            raise forms.ValidationError(_("Uploaded image with more than one selected image."))
        # Beware: don't close the file here !
        fh = Image.open(img)
        if fh.mode not in ['1', 'L']:
            raise forms.ValidationError(_("Uploaded image should be black and white."))
        isize = (self.parts[0].image.width, self.parts[0].image.height) 
        if fh.size != isize:
            raise forms.ValidationError(_("Uploaded image should be the same size as original image {}.").format(isize))
        return img
    
    def process(self):
        self.save()  # save settings
        task = self.cleaned_data.get('task')
        if task == 'binarize':
            if len(self.parts) == 1 and self.cleaned_data.get('bw_image'):
                self.parts[0].bw_image = self.cleaned_data['bw_image']
                self.parts[0].save()
            else:
                for part in self.parts:
                    part.binarize(user_pk=self.user.pk)
        elif task == 'segment':
            for part in self.parts:
                part.segment(user_pk=self.user.pk,
                             text_direction=self.cleaned_data['text_direction'])
        elif task == 'train':
            if self.cleaned_data.get('upload_model'):
                # create corresponding OcrModel
                pass
            elif self.cleaned_data.get('new_model'):
                # create model and corresponding OcrModel
                pass
            elif self.cleaned_data.get('train_model'):
                pass
            # part.train(user_pk=self.user.pk, model=None)
        elif task == 'transcribe':
            for part in self.parts:
                part.transcribe(user_pk=self.user.pk)


class UploadImageForm(BootstrapFormMixin, forms.ModelForm):
    class Meta:
        model = DocumentPart
        fields = ('image',)
    
    def __init__(self, *args, **kwargs):
        self.document = kwargs.pop('document')
        super().__init__(*args, **kwargs)
    
    def save(self, commit=True):
        part = super().save(commit=False)
        part.document = self.document
        if commit:
            part.save()
        return part
