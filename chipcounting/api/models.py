import uuid
import logging
import json
import numpy as np

from PIL import Image
from django.db import models
from auditlog.registry import auditlog
from django.contrib.postgres import fields
from django.utils import timezone
from datetime import datetime
from chipcounting.users.models import User
from django.core.validators import ValidationError

log = logging.getLogger(__name__)


class Games(models.Model):
    name = models.CharField(max_length=255)
    description = models.CharField(max_length=255, default='', blank=True)
    create_tstamp = models.DateTimeField(auto_now_add=True)
    update_tstamp = models.DateTimeField(auto_now=True)

    def __str__(self):
        return self.name

    class Meta:
        db_table = 'games'
        verbose_name = 'Casino Game Type'
        verbose_name_plural = 'Casino Game Types'


class Dealers(models.Model):
    name = models.CharField(max_length=255)
    create_tstamp = models.DateTimeField(auto_now_add=True)
    update_tstamp = models.DateTimeField(auto_now=True)

    def __str__(self):
        return self.name

    class Meta:
        db_table = 'dealers'
        verbose_name = 'Casino Dealer'
        verbose_name_plural = 'Casino Dealers'

class RFID(models.Model):

    RFID_TYPES = (
        ('DEALER', 'Dealer'),
        ('GAME', 'Game')
    )

    token_id = models.CharField(max_length=255, unique=True)
    token_type = models.CharField(max_length=255, choices=RFID_TYPES)

    dealer = models.ForeignKey(Dealers, null=True, blank=True, on_delete=models.SET_NULL)
    game = models.ForeignKey(Games, null=True, blank=True, on_delete=models.SET_NULL)
    active = models.BooleanField(default=True)

    create_tstamp = models.DateTimeField(auto_now_add=True)
    update_tstamp = models.DateTimeField(auto_now=True)

    def __str__(self):
        ret = 'RFID: {} Type: {}'.format(self.token_id, self.token_type)

        if self.dealer:
            ret += ' {}'.format(self.dealer.name)
        if self.game:
            ret += ' {}'.format(self.game.name)

        return ret

    def validation_unique(self, *args, **kwargs):
        super(RFID, self).validate_unique(*args, **kwargs)

        if self.token_type == 'DEALER' and self.game is not None:
            raise ValidationError({
                'game': ['Game must be NULL if RFID Token is Dealer',]
            })

        if self.token_type == 'GAME' and self.dealer is not None:
            raise ValidationError({
                'dealer': ['Dealer must be NULL if RFID Token is Game',]
            })

    class Meta:
        db_table = 'rfid'
        verbose_name = 'RFID Token'
        verbose_name_plural = 'RFID Tokens'


class Tables(models.Model):
    name = models.CharField(max_length=255)
    description = models.CharField(max_length=255, blank=True)
    location = models.CharField(max_length=255, blank=True)

    active = models.BooleanField(default=True)

    create_tstamp = models.DateTimeField(auto_now_add=True)
    update_tstamp = models.DateTimeField(auto_now=True)

    def __str__(self):
        return '{} at {}'.format(self.name, self.location)

    class Meta:
        db_table = 'tables'
        verbose_name = 'Casino Table'
        verbose_name_plural = 'Casino Tables'


class RFIDEvents(models.Model):
    table = models.ForeignKey(Tables, on_delete=models.SET_NULL, null=True, blank=True)
    rfid = models.ForeignKey(RFID, on_delete=models.CASCADE, null=False, blank=True)
    tstamp = models.DateTimeField(verbose_name='Timestamp', db_index=True, default=timezone.now, editable=False)

    create_tstamp = models.DateTimeField(db_index=True, auto_now_add=True)
    update_tstamp = models.DateTimeField(auto_now=True)

    def __str__(self):
        return 'ID: {} at {}, Table: {} RFID: {}'.format(
            self.id, self.tstamp, self.table, self.rfid
        )

    class Meta:
        db_table = 'rfid_events'
        verbose_name = 'RFID Scan Event'
        verbose_name_plural = 'RFID Scan Events'


def get_device_last_image_upload_path(instance, filename):
    device_id = instance.id
    return 'device_images/last_{}.jpg'.format(device_id)


def get_device_trained_image_upload_path(instance, filename):
    device_id = instance.id
    return 'device_images/trained_{}.jpg'.format(device_id)


class Devices(models.Model):
    name = models.CharField(max_length=255)

    token = models.UUIDField(default=uuid.uuid4)
    table = models.OneToOneField(Tables, on_delete=models.SET_NULL, null=True)

    last_image = models.ImageField(
        verbose_name='Last Update Image',
        upload_to=get_device_last_image_upload_path,
        editable=False,
        null=True,
        blank=True
    )
    trained_image = models.ImageField(
        verbose_name='Trained Idle Image',
        upload_to=get_device_trained_image_upload_path,
        null=True,
        blank=True
    )
    ping_tstamp = models.DateTimeField(verbose_name='Last Ping Time', editable=False, null=True)

    computer_vision_config = models.TextField(default='')

    create_tstamp = models.DateTimeField(auto_now_add=True)
    update_tstamp = models.DateTimeField(auto_now=True)

    def __str__(self):
        ret = 'Device {}'.format(self.id)
        if self.table:
            ret += ' at Table: {}'.format(self.table)

        return ret

    def get_computer_vision_json_or_none(self):
        try:
            ret = json.loads(self.computer_vision_config)
            if not isinstance(ret, dict):
                log.warn("CV Config not in a valid dictionary format for ID {}".format(self.id))
                ret = None
        except Exception as e:
            log.info("Failed to load CV config {}".format(e))
            ret = None

        return ret

    class Meta:
        db_table = 'devices'
        verbose_name = 'Device'
        verbose_name_plural = 'Devices'


class Chips(models.Model):
    name = models.CharField(max_length=255)
    description = models.CharField(max_length=255)
    image_top = models.ImageField()
    image_side = models.ImageField()
    value = models.DecimalField(max_digits=12, decimal_places=2, default=0)

    properties = fields.JSONField()
    active = models.BooleanField(default=True)

    create_tstamp = models.DateTimeField(auto_now_add=True)
    update_tstamp = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name = 'Chip'
        verbose_name_plural = 'Chips'


class TableEvents(models.Model):
    EVENT_TYPES = (
        ('DEALER', 'Dealer Clock In'),
        ('GAME', 'Game Clock In'),
        ('TABLE_CLOSE', 'Table Close')
    )

    table = models.ForeignKey(Tables, on_delete=models.PROTECT)
    tstamp = models.DateTimeField(db_index=True, verbose_name='Timestamp', default=timezone.now)
    event_type = models.CharField(max_length=255, choices=EVENT_TYPES)

    game = models.ForeignKey(Games, blank=True, default=None, on_delete=models.PROTECT, null=True)
    dealer = models.ForeignKey(Dealers, blank=True, default=None, on_delete=models.PROTECT, null=True)

    source = models.ForeignKey(RFIDEvents, blank=True, default=None,
                               editable=False, on_delete=models.PROTECT, null=True
    )

    create_tstamp = models.DateTimeField(auto_now_add=True)
    update_tstamp = models.DateTimeField(auto_now=True)

    def __str__(self):
        ret = 'Table: {} {} {}'.format(self.table, self.tstamp, self.event_type)
        if self.game:
            ret += ' {}'.format(self.game)
        if self.dealer:
            ret += ' {}'.format(self.dealer)

        return ret

    class Meta:
        db_table = 'table_events'
        verbose_name = 'Table Event'
        verbose_name_plural = 'Table Events'


def get_chip_debug_path(instance, filename):
    date_fragment = datetime.now().strftime('%Y/%m/%d')

    if instance.chip_upload is not None:
        unique_id = instance.chip_upload.unique_id
    else:
        unique_id = str(uuid.uuid4())
    table_id = instance.table.id

    return 'debug/{}/{}_{}'.format(date_fragment, table_id, unique_id)


def get_chip_upload_path(instance, filename):
    date_fragment = datetime.now().strftime('%Y/%m/%d')

    unique_id = instance.unique_id
    table_id = instance.table.id

    return 'chips_raw/{}/{}_{}'.format(date_fragment, table_id, unique_id)


class ChipUploads(models.Model):

    table = models.ForeignKey(Tables, on_delete=models.CASCADE)
    device = models.ForeignKey(Devices, null=False, on_delete=models.CASCADE)
    tstamp = models.DateTimeField(db_index=True, verbose_name='Timestamp', default=timezone.now, editable=False)
    unique_id = models.UUIDField(default=uuid.uuid4, unique=True, editable=False)

    data = models.FileField(upload_to=get_chip_upload_path, editable=False)
    data_size = models.PositiveIntegerField(default=0, null=False)

    create_tstamp = models.DateTimeField(db_index=True, auto_now_add=True)
    update_tstamp = models.DateTimeField(auto_now=True)

    def __str__(self):
        return 'Upload {} at {} {} bytes'.format(self.id, self.tstamp, self.data_size)

    def save(self, *args, **kwargs):
        self.data_size = self.data.size
        super(ChipUploads, self).save(*args, **kwargs)

    def get_deserialized_data(self):
        pass

    class Meta:
        db_table = 'chip_uploads'
        verbose_name = 'Device Chip Upload'
        verbose_name_plural = 'Device Chip Uploads'

class ChipEvents(models.Model):
    table = models.ForeignKey(Tables, on_delete=models.CASCADE)
    tstamp = models.DateTimeField(db_index=True, verbose_name='timestamp', default=timezone.now)
    results = fields.JSONField()
    amount = models.DecimalField(max_digits=12, decimal_places=2, default=0)
    unique_id = models.UUIDField(default=uuid.uuid4, editable=False)
    chip_upload = models.ForeignKey(ChipUploads, null=True, default=None, editable=False, on_delete=models.SET_NULL)

    create_tstamp = models.DateTimeField(db_index=True, auto_now_add=True)
    update_tstamp = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = 'chip_events'
        verbose_name = 'Table Chip Drop Event'
        verbose_name_plural = 'Table Chip Drop Events'
class ChipEventsDebug(models.Model):
    RESULT_TYPES = (
        ('WAIT', 'Waiting to be processed'),
        ('PROCESSED', 'Processed'),
        ('EMPTY', 'No Slide Drop Detected'),
        ('ERROR', 'Error')
    )
    chip_upload = models.ForeignKey(ChipUploads, null=True, default=None, editable=False, on_delete=models.SET_NULL)
    table = models.ForeignKey(Tables, null=False, editable=False, on_delete=models.CASCADE)

    result = models.CharField(
        verbose_name='Processing Result',
        max_length=255, default='WAIT',
        choices=RESULT_TYPES,
        editable=False
    )

    upload_tstamp = models.DateTimeField(verbose_name='Timestamp', default=timezone.now, db_index=True, editable=False)
    unique_id = models.UUIDField(verbose_name='Unique ID', null=False, db_index=True, editable=False)

    data = models.FileField(upload_to=get_chip_debug_path, null=True, editable=False)
    data_size = models.PositiveIntegerField(default=0, null=False)

    message = models.TextField(editable=False, blank=True, default='', null=False)

    create_tstamp = models.DateTimeField(db_index=True, auto_now_add=True)
    # update_tstamp = models.DateTimeField(auto_now=True)

    def save(self, *args, **kwargs):
        # You need to query .name to see if the field is NULL
        if self.data.name is None:
            self.data_size = 0
        else:
            self.data_size = self.data.size
        super(ChipEventsDebug, self).save(*args, **kwargs)

    class Meta:
        db_table = 'chip_events_debug'
        verbose_name = 'Table Chip Drop Events Debug'
        verbose_name_plural = 'Table Chip Drop Debug Events'


class CVModels(models.Model):

    description = models.TextField(verbose_name='Description')
    data = models.FileField(verbose_name='Model Binary File')
    labels = models.TextField(verbose_name='Labels in JSON array format')

    create_tstamp = models.DateTimeField(db_index=True, auto_now_add=True)
    update_tstamp = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = 'cv_models'
        verbose_name = 'Computer Vision Model'
        verbose_name_plural = 'Computer Vision Models'


class CVImageMasks(models.Model):

    CAMERAS = (
        ('top', 'Top'),
        ('bottom', 'Bottom')
    )
    set_id = models.IntegerField(verbose_name='Mask Set ID', db_index=True)
    camera = models.CharField(choices=CAMERAS, max_length=255)
    chip_position = models.IntegerField(verbose_name='Chip Position')

    mask = models.ImageField()

    create_tstamp = models.DateTimeField(db_index=True, auto_now_add=True)
    update_tstamp = models.DateTimeField(auto_now=True)

    def mask_numpy_format(self):
        pil_image = Image.open(self.mask)
        cv_format = np.array(pil_image)
        if len(cv_format.shape) == 3:
            cv_format = cv_format[:, :, 0]
        return cv_format

    class Meta:
        db_table = 'cv_image_masks'
        verbose_name = 'Computer Vision Model Image Mask'


auditlog.register(Games)
auditlog.register(RFID)
auditlog.register(Tables)
auditlog.register(Chips)
