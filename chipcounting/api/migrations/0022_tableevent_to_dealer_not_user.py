# Generated by Django 2.0.7 on 2019-07-26 20:04

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('api', '0021_auto_20190720_0108'),
    ]

    operations = [
        migrations.AlterField(
            model_name='tableevents',
            name='dealer',
            field=models.ForeignKey(blank=True, default=None, null=True, on_delete=django.db.models.deletion.PROTECT, to='api.Dealers'),
        ),
    ]
