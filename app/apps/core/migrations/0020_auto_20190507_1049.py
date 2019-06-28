# Generated by Django 2.1.4 on 2019-05-07 10:49

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('core', '0019_load_scripts'),
    ]

    operations = [
        migrations.AlterModelOptions(
            name='script',
            options={'ordering': ('name',)},
        ),
        migrations.AddField(
            model_name='block',
            name='external_id',
            field=models.CharField(blank=True, max_length=128),
        ),
        migrations.AddField(
            model_name='line',
            name='external_id',
            field=models.CharField(blank=True, max_length=128),
        ),
        migrations.AlterField(
            model_name='document',
            name='read_direction',
            field=models.CharField(choices=[('ltr', 'Left to right'), ('rtl', 'Right to left')], default='ltr', max_length=3),
        ),
        migrations.AlterField(
            model_name='script',
            name='text_direction',
            field=models.CharField(choices=[('horizontal-lr', 'Horizontal l2r'), ('horizontal-rl', 'Horizontal r2l'), ('vertical-lr', 'Vertical l2r'), ('vertical-rl', 'Vertical r2l'), ('ttb', 'Top to bottom')], default='horizontal-lr', max_length=64),
        ),
    ]