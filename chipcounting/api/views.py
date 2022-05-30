import logging
import uuid
import base64
import six
import uuid
import zlib
import json

from django.core.files.base import ContentFile
from django.shortcuts import render
from django.db import models
from django.db.utils import IntegrityError
from django.utils.crypto import constant_time_compare

#from rest_framework.renders import JSONRenderer, JSONParser
from rest_framework.views import APIView
from rest_framework.permissions import IsAuthenticated, IsAdminUser
from rest_framework import status, serializers, renderers
from rest_framework.response import Response
from rest_framework.exceptions import AuthenticationFailed

from chipcounting.api.models import ChipUploads, ChipEventsDebug, Tables, TableEvents, Devices, RFID, RFIDEvents
from chipcounting.image_processing.utils import ChipUploadHelper, DebugChipDataHelper
from chipcounting.api.tasks import run_cv_code

log = logging.getLogger(__name__)

class Base64ImageField(serializers.ImageField):
    """
    A Django REST framework field for handling image-uploads through raw post data.
    It uses base64 for encoding and decoding the contents of the file.

    Heavily based on
    https://github.com/tomchristie/django-rest-framework/pull/1268

    Updated for Django REST framework 3.
    """

    def to_internal_value(self, data):

        # Check if this is a base64 string
        if isinstance(data, six.string_types):
            # Check if the base64 string is in the "data:" format
            if 'data:' in data and ';base64,' in data:
                # Break out the header from the base64 content
                header, data = data.split(';base64,')

            # Try to decode the file. Return validation error if it fails.
            try:
                decoded_file = base64.b64decode(data)
            except TypeError:
                self.fail('invalid_image')

            # Generate file name:
            file_name = str(uuid.uuid4())[:12] # 12 characters are more than enough.
            # Get the file name extension:
            file_extension = self.get_file_extension(file_name, decoded_file)

            complete_file_name = "%s.%s" % (file_name, file_extension, )

            data = ContentFile(decoded_file, name=complete_file_name)

        return super(Base64ImageField, self).to_internal_value(data)

    def get_file_extension(self, file_name, decoded_file):
        import imghdr

        extension = imghdr.what(file_name, decoded_file)
        extension = "jpg" if extension == "jpeg" else extension

        return extension

def validate_device_token(device_obj, request):
    device_token = str(device_obj.token)
    given_token = request.META.get('HTTP_AUTHORIZATION')

    if not constant_time_compare(device_token, given_token):
        log.error("Device {} Invalid Token".format(device_obj.id))
        raise AuthenticationFailed()

    return True

def get_device_and_validate_token(request, device_id):
    try:
        device = Devices.objects.get(pk=device_id)
    except Devices.DoesNotExist:
        log.error("Device ID: {} does not exist".format(device_id))
        return Response({'error': 'Device Does Not Exist'}, status=status.HTTP_404_NOT_FOUND)

    validate_device_token(device, request)

    return device

class DeviceDataSerializer(serializers.ModelSerializer):
    data = serializers.FileField()
    data_size = serializers.IntegerField(read_only=True)
    unique_id = serializers.UUIDField()
    tstamp = serializers.DateTimeField()

    def validate_data(self, value):
        # store data compressed to save room on disk
        json_data = value.read()
        try:
            _ = json.loads(json_data)
        except ValueError as e:
            log.exception("Invalid JSON Data Uplaoded")
            raise e

        compressed = zlib.compress(json_data)
        value.seek(0)
        value.truncate()
        value.write(compressed)
        value.seek(0)
        value.size = len(compressed)

        return value

    class Meta:
        model = ChipUploads
        fields = ('table', 'device', 'tstamp', 'unique_id', 'data', 'data_size')

class RFIDEventSerializer(serializers.ModelSerializer):
    tstamp = serializers.DateTimeField()

    class Meta:
        model = RFIDEvents
        fields = ('table', 'tstamp', 'rfid')

class DevicePingSerializer(serializers.Serializer):
    tstamp = serializers.DateTimeField()
    image = Base64ImageField(allow_null=True, allow_empty_file=True, required=False)

class RFIDEventsUpload(APIView):

    def post(self, request, device_id):
        #/api/device/<device_id>/rfid
        # {'token_id': 'XXX', 'tstamp'}
        token_id = request.data['token_id']

        device = get_device_and_validate_token(request, device_id)
        try:
            rfid = RFID.objects.get(token_id=token_id)
        except RFID.DoesNotExist:
            log.error("RFID ID: {} does not exist".format(token_id))
            return Response({'error': 'Token Does Not Exist'}, status=status.HTTP_409_CONFLICT)

        data = {
            'rfid': rfid.id,
            'table': device.table.id,
            'tstamp': request.data['tstamp']
        }

        ser = RFIDEventSerializer(data=data)
        ser.is_valid(raise_exception=True)
        raw_event = ser.save()

        game = rfid.game
        dealer = rfid.dealer
        if game is not None or dealer is not None:
            event = TableEvents(
                table=device.table,
                tstamp=raw_event.tstamp,
                event_type=raw_event.rfid.token_type,
                game=game,
                dealer=dealer,
                source=raw_event
            )

            event.save()
        else:
            log.warn("Token Scan without attached game or dealer: {}".format(rfid.id))

        return Response(status=status.HTTP_200_OK)


class DeviceDataUpload(APIView):

    def put(self, request, device_id, upload_uuid, format=None):
        # /api/device/<device_id>/data/<upload_uuid>
        # {'data': 'BLOB', 'tstamp': 'ISO8601'}
        device = get_device_and_validate_token(request, device_id)

        data = request.data

        data['table'] = device.table.id
        data['device'] = device.id
        data['unique_id'] = upload_uuid
        dds = DeviceDataSerializer(data=data)
        log.debug(dds)
        dds.is_valid(raise_exception=True)

        try:
            dds.save()
        except IntegrityError as e:
            return Response(status=status.HTTP_202_ACCEPTED)

        return Response(status=status.HTTP_200_OK)

class DevicePing(APIView):
    def post(self, request, device_id, format=None):
        # /api/device/<device_id>/ping
        # {'image': 'base64', 'tstamp': 'ISO8601'}

        device = get_device_and_validate_token(request, device_id)
        ping = DevicePingSerializer(data=request.data)
        ping.is_valid(raise_exception=True)

        device.ping_tstamp = ping.validated_data['tstamp']
        device.last_image = ping.validated_data.get('image', None)

        device.save()

        return Response(status=status.HTTP_200_OK)

class ZipFileRenderer(renderers.BaseRenderer):
    media_type = 'application/zip'
    format = 'zip'
    charset = None
    render_style = 'binary'

    def render(self, data, media_type=None, renderer_context=None):
        return data

class DownloadImageZip(APIView):

    renderer_classes = (ZipFileRenderer, )
    permission_classes = (IsAdminUser, )

    def get(self, request, row_id, format=None):
        # /api/data/raw/<row_id>/images
        ddu = ChipUploads.objects.get(pk=row_id)
        data = ddu.data.open('rb').read()
        helper = ChipUploadHelper.init_from_serialized(data)

        zipfile = helper.generate_image_zipfile()

        headers = {'content-disposition': 'attachment; filename=images_{}.zip'.format(row_id)}
        return Response(zipfile, headers=headers)

class DownloadDebugImageZip(APIView):

    renderer_classes = (ZipFileRenderer, )
    permission_classes = (IsAdminUser, )

    def get(self, request, row_id, format=None):
        # /api/data/debug/<row_id>/images
        ddu = ChipEventsDebug.objects.get(pk=row_id)
        data = ddu.data.open('rb').read()
        helper = DebugChipDataHelper.init_from_serialized(data)

        zipfile = helper.generate_image_zipfile()

        headers = {'content-disposition': 'attachment; filename=images_{}.zip'.format(row_id)}
        return Response(zipfile, headers=headers)

class RunCVCodeForUpload(APIView):

    permission_classes = (IsAdminUser, )

    def post(self, request, row_id):
        # POST /api/data/raw/<row_id>/process
        run_cv_code.delay()

        return Response(status=status.HTTP_200_OK)
