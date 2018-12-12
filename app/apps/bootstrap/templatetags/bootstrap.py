from django import template


register = template.Library()

@register.filter
def level_to_color(tags):
    """
    map django.contrib.messages level to bootstraps color code 
    """
    level_map = {
        'debug': 'dark',
        'info': 'info',
        'success': 'success',
        'warning': 'warning',
        'error': 'danger'
    }
    return level_map[tags]


@register.simple_tag
def render_field(field, group=False, **kwargs):
    tplt = template.loader.get_template('django/forms/widgets/field.html')
    field.field.widget.attrs.update(**kwargs)
    context = {'field': field, 'group': group}
    return tplt.render(context)  #.mark_safe()