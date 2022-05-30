# Generated by Django 2.0.7 on 2019-07-20 05:08

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('api', '0020_auto_20190626_1348'),
    ]

    operations = [
        migrations.CreateModel(
            name='Dealers',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=255)),
                ('create_tstamp', models.DateTimeField(auto_now_add=True)),
                ('update_tstamp', models.DateTimeField(auto_now=True)),
            ],
            options={
                'verbose_name': 'Casino Dealer',
                'verbose_name_plural': 'Casino Dealers',
                'db_table': 'dealers',
            },
        ),
        migrations.AlterField(
            model_name='rfid',
            name='dealer',
            field=models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.SET_NULL, to='api.Dealers'),
        ),
    ]