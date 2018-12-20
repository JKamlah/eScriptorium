from imagekit import ImageSpec, register
from imagekit.processors import ResizeToFill, ResizeToFit


class ListThumbnail(ImageSpec):
    processors = [ResizeToFill(50, 50)]
    format = 'JPEG'
    options = {'quality': 85}


class CardThumbnail(ImageSpec):
    processors = [ResizeToFill(180, 180)]
    format = 'JPEG'
    options = {'quality': 85}

    
class Large(ImageSpec):
    processors = [ResizeToFit(1110, upscale=False)]
    options = {'quality': 90}

    
register.generator('core:list.thumbnail', ListThumbnail)
register.generator('core:card.thumbnail', CardThumbnail)
register.generator('core:large', Large)
