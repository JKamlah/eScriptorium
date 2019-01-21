from django import forms
from django.forms.renderers import TemplatesSetting


class BootstrapFormMixin():
    default_renderer = TemplatesSetting
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for name, field in self.fields.items():
            if not field.widget.is_hidden:
                field.widget.attrs.update({
                    'placeholder': field.label,
                    'title': field.label
                })
                class_ = field.widget.attrs.get('class', '')
                if issubclass(field.widget.__class__, forms.CheckboxInput):
                    class_ += ' form-check-input'
                elif issubclass(field.widget.__class__, forms.FileInput):
                    class_ += ' form-control-file'
                else:
                    class_ += ' form-control'
                    if issubclass(field.widget.__class__, forms.Select):
                        class_  += ' custom-select'
                if field.widget.input_type == 'select':
                    field.widget.need_label = True

                field.widget.attrs['class'] = class_
    
    def full_clean(self):
        super().full_clean()
        if self._errors:
            for name, error in self._errors.items():
                if not self.fields[name].widget.is_hidden:
                    self.fields[name].widget.attrs['class'] += ' is-invalid'
