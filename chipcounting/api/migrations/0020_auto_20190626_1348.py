# Generated by Django 2.0.7 on 2019-06-26 17:48

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('api', '0019_auto_20190502_1603'),
    ]

    operations = [
        migrations.AlterModelOptions(
            name='cvimagemasks',
            options={'verbose_name': 'Computer Vision Model Image Mask'},
        ),
        migrations.AddField(
            model_name='chipevents',
            name='chip_upload',
            field=models.ForeignKey(default=None, editable=False, null=True, on_delete=django.db.models.deletion.SET_NULL, to='api.ChipUploads'),
        ),
    ]
