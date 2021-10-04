# Generated by Django 2.2.23 on 2021-09-15 08:34

from django.db import migrations, models


def batch_qs(qs, batch_size=10):
    total = qs.count()
    for start in range(0, total, batch_size):
        yield qs[start:start+batch_size]


def set_image_file_size(apps, schema_editor):
    DocumentPart = apps.get_model('core', 'DocumentPart')

    for parts in batch_qs(DocumentPart.objects.all()):
        for part in parts:
            try:
                part.image_file_size = part.image.size
            except FileNotFoundError as e:
                print(f"Couldn't update image_file_size field on {part.id}, the file wasn't found: {e}")

        DocumentPart.objects.bulk_update(parts, ['image_file_size'])


class Migration(migrations.Migration):

    dependencies = [
        ('core', '0053_auto_20210720_0932'),
    ]

    operations = [
        migrations.AddField(
            model_name='documentpart',
            name='image_file_size',
            field=models.BigIntegerField(default=0),
        ),
        migrations.RunPython(
            set_image_file_size,
            reverse_code=migrations.RunPython.noop,
        ),
        migrations.AlterField(
            model_name='documentpart',
            name='image_file_size',
            field=models.BigIntegerField(),
        ),
    ]
