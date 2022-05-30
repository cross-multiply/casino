# Generated by Django 2.0.3 on 2018-04-02 17:31

from django.conf import settings
import django.contrib.postgres.fields.jsonb
from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    initial = True

    dependencies = [
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
    ]

    operations = [
        migrations.CreateModel(
            name='ChipEvents',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('results', django.contrib.postgres.fields.jsonb.JSONField()),
                ('create_tstamp', models.DateTimeField(auto_now_add=True)),
                ('update_tstamp', models.DateTimeField(auto_now=True)),
            ],
            options={
                'db_table': 'chip_events',
            },
        ),
        migrations.CreateModel(
            name='ChipEventsDebug',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('log_data', models.TextField()),
                ('debug_data', models.BinaryField()),
                ('results', django.contrib.postgres.fields.jsonb.JSONField()),
                ('create_tstamp', models.DateTimeField(auto_now_add=True)),
                ('chip_event_id', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='api.ChipEvents')),
            ],
            options={
                'db_table': 'chip_events_debug',
            },
        ),
        migrations.CreateModel(
            name='Devices',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('mender_id', models.CharField(max_length=255)),
                ('create_tstamp', models.DateTimeField(auto_now_add=True)),
                ('update_tstamp', models.DateTimeField(auto_now=True)),
            ],
            options={
                'db_table': 'devices',
            },
        ),
        migrations.CreateModel(
            name='Games',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=255)),
                ('description', models.CharField(max_length=255)),
                ('create_tstamp', models.DateTimeField(auto_now_add=True)),
                ('update_tstamp', models.DateTimeField(auto_now=True)),
            ],
            options={
                'db_table': 'games',
            },
        ),
        migrations.CreateModel(
            name='RFID',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('token_id', models.CharField(max_length=255)),
                ('token_type', models.CharField(choices=[('DEALER', 'Dealer'), ('GAME', 'Game')], max_length=255)),
                ('active', models.BooleanField(default=True)),
                ('create_tstamp', models.DateTimeField(auto_now_add=True)),
                ('update_tstamp', models.DateTimeField(auto_now=True)),
                ('dealer_id', models.ForeignKey(null=True, on_delete=django.db.models.deletion.SET_NULL, to=settings.AUTH_USER_MODEL)),
                ('game_id', models.ForeignKey(null=True, on_delete=django.db.models.deletion.SET_NULL, to='api.Games')),
            ],
            options={
                'db_table': 'rfid',
            },
        ),
        migrations.CreateModel(
            name='RFIDEvents',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('create_tstamp', models.DateTimeField(auto_now_add=True)),
                ('rfid_id', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='api.RFID')),
            ],
            options={
                'db_table': 'rfid_events',
            },
        ),
        migrations.CreateModel(
            name='Tables',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=255)),
                ('description', models.CharField(max_length=255)),
                ('location', models.CharField(max_length=255)),
                ('active', models.BooleanField(default=True)),
                ('create_tstamp', models.DateTimeField(auto_now_add=True)),
                ('update_tstamp', models.DateTimeField(auto_now=True)),
            ],
            options={
                'db_table': 'tables',
            },
        ),
        migrations.AddField(
            model_name='rfidevents',
            name='table_id',
            field=models.ForeignKey(null=True, on_delete=django.db.models.deletion.SET_NULL, to='api.Tables'),
        ),
        migrations.AddField(
            model_name='devices',
            name='table_id',
            field=models.ForeignKey(null=True, on_delete=django.db.models.deletion.SET_NULL, to='api.Tables'),
        ),
        migrations.AddField(
            model_name='chipevents',
            name='table_id',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='api.Tables'),
        ),
    ]
