import base64
import logging

from django.contrib import admin
from django.urls import reverse
from django.utils.safestring import mark_safe
from django.utils.html import format_html
from django.http import HttpResponseRedirect

from .models import Games, RFID, Tables, \
    RFIDEvents, Devices, ChipEvents, ChipEventsDebug, \
    Chips, TableEvents, ChipUploads, CVImageMasks, CVModels
from .tasks import run_cv_code

from chipcounting.image_processing.utils import ChipUploadHelperV2

log = logging.getLogger(__name__)


class ActionInChangeFormMixin(object):
    """Lifted From
    https://stackoverflow.com/questions/2805701/is-there-a-way-to-get-custom-django-admin-actions-to-appear-on-the-change-view
    """

    def response_action(self, request, queryset):
        """
        Prefer http referer for redirect
        """
        response = super(ActionInChangeFormMixin, self).response_action(request, queryset)
        if isinstance(response, HttpResponseRedirect):
            response['Location'] = request.META.get('HTTP_REFERER', response.url)
        return response

    def change_view(self, request, object_id, extra_context=None):
        actions = self.get_actions(request)
        if actions:
            action_form = self.action_form(auto_id=None)
            action_form.fields['action'].choices = self.get_action_choices(request)
        else:
            action_form = None
        extra_context = extra_context or {}
        extra_context['action_form'] = action_form
        return super(ActionInChangeFormMixin, self).change_view(request, object_id, extra_context=extra_context)


@admin.register(RFIDEvents)
class RFIDEventsAdmin(admin.ModelAdmin):
    readonly_fields = ('table', 'rfid', 'tstamp', 'create_tstamp', 'update_tstamp')
    list_display = ('id', 'rfid', 'table', 'tstamp')
    list_filter = ('tstamp', 'rfid', 'table')
    list_select_related = ('rfid', 'table')
    date_hierarchy = 'tstamp'

    def has_add_permission(self, request):
        """ We normally don't want to let users create this model"""
        return False


@admin.register(TableEvents)
class TableEventsAdmin(admin.ModelAdmin):
    readonly_fields = ('id', 'source', 'create_tstamp', 'update_tstamp')
    list_display = ('id', 'table', 'tstamp', 'event_type', 'game', 'dealer')
    list_filter = ('table', 'tstamp', 'event_type', 'game', 'dealer')
    list_select_related = ('table', 'game', 'dealer')
    date_hierarchy = 'tstamp'


@admin.register(Games)
class GamesAdmin(admin.ModelAdmin):
    readonly_fields = ('id', 'create_tstamp', 'update_tstamp')
    list_display = ('id', 'name', 'description')


@admin.register(RFID)
class RFIDAdmin(admin.ModelAdmin):
    readonly_fields = ('id', 'create_tstamp', 'update_tstamp')
    list_display = ('id', 'token_id', 'token_type', 'dealer', 'game', 'active')
    list_filter = ('active', )
    list_select_related = ('dealer', 'game')


@admin.register(Tables)
class TablesAdmin(admin.ModelAdmin):
    readonly_fields = ('id', 'create_tstamp', 'update_tstamp')
    list_display = ('id', 'name', 'description', 'location', 'active')


@admin.register(Devices)
class DevicesAdmin(admin.ModelAdmin):
    readonly_fields = ('id', 'token', 'last_image_tag', 'last_image', 'trained_image_tag', 'ping_tstamp', 'create_tstamp', 'update_tstamp')
    list_display = ('id', 'table', 'ping_tstamp')

    def last_image_tag(self, obj):
        #template = '<img alt="embedded" src="data:image/jpeg;base64,{}"/>'
        #data = obj.last_url.open('rb')
        return mark_safe('<img src="{}" />'.format(obj.last_image.url))

    def trained_image_tag(self, obj):
        return mark_safe('<img src="{}" />'.format(obj.trained_image.url))


@admin.register(Chips)
class ChipsAdmin(admin.ModelAdmin):
    readonly_fields = ('id', 'create_tstamp', 'update_tstamp')
    list_display = ('id', 'name', 'description', 'value')


@admin.register(ChipEvents)
class ChipEventsAdmin(admin.ModelAdmin):
    readonly_fields = ('id', 'unique_id', 'create_tstamp', 'update_tstamp')
    list_display = ('id', 'table', 'tstamp', 'amount')
    list_filter = ('tstamp', 'table', 'amount')
    list_select_related = ('table',)
    date_hierarchy = 'tstamp'


@admin.register(ChipEventsDebug)
class ChipEventsDebugAdmin(admin.ModelAdmin):
    readonly_fields = (
        'id', 'link_to_chip_upload', 'table', 'unique_id', 'drop_images', 'download_images_link',
        'upload_tstamp', 'result', 'data', 'data_size', 'message', 'create_tstamp'
    )
    list_display = ('id', 'table', 'result', 'link_to_chip_upload', 'unique_id', 'upload_tstamp', 'create_tstamp')
    list_filter = ('upload_tstamp', 'table', 'result', 'create_tstamp')
    list_select_related = ('chip_upload', )
    date_hierarchy = ('upload_tstamp')
    search_fields = ('^unique_id', )

    def has_add_permission(self, request):
        """ We normally don't want to let users create this model"""
        return False

    def link_to_chip_upload(self, instance):
        if instance.chip_upload is not None:
            link = reverse('admin:api_chipuploads_change', args=(instance.chip_upload.pk,))
            return format_html('<a href="{}">View {}</a>', link, instance.chip_upload)
        else:
            return 'No Matching ChipUpload Found. Data might have been deleted.'

    def drop_video(self, instance):
        pass

    def drop_images(self, instance):
        template = '<img alt="embedded" src="data:image/jpeg;base64,{}"/>'
        fg_img = None
        bg_img = None
        success = False
        try:
            data = instance.chip_upload.data.open('rb').read()
            upload_data = ChipUploadHelperV2.init_from_serialized(data)
            primed, recovery = upload_data.get_drop_images()

            fg_img = base64.b64encode(primed).decode('ascii')
            bg_img = base64.b64encode(recovery).decode('ascii')
            success = True
            #upload_data = DebugChipDataHelper.init_from_serialized(data)
            ##fg_img = upload_data._calibrated_fg_img
            #bg_img = upload_data._calibrated_slide_img
            #if fg_img is not None and bg_img is not None:
            #    fg_img = base64.b64encode(fg_img).decode('ascii')
            #    bg_img = base64.b64encode(bg_img).decode('ascii')
            #    success = True
        except Exception as e:
            log.exception('Error generating drop images for {}'.format(instance))

        if success:
            ret = mark_safe(template.format(fg_img) + template.format(bg_img))
        else:
            ret = mark_safe('<p>Chip Upload Contains Invalid Data</p>')

        return ret

    def download_images_link(self, instance):
        url = reverse('download-debug-images', kwargs={'row_id': instance.id})

        return format_html('<a href="{}">Download Images</a>', url)


@admin.register(ChipUploads)
class ChipUploadsAdmin(ActionInChangeFormMixin, admin.ModelAdmin):
    readonly_fields = ('id', 'table', 'tstamp', 'unique_id', 'drop_images', 'download_images_link',
                       'data', 'data_size', 'device', 'create_tstamp', 'update_tstamp')
    list_display = ('id', 'table', 'tstamp', 'unique_id', 'data_size')
    list_filter = ('tstamp', 'table')
    list_select_related = ('table', )
    date_hierarchy = 'tstamp'
    actions = ['run_cv_code_action']
    search_fields = ('^unique_id', )

    def has_add_permission(self, request):
        """ We normally don't want to let users create this model"""
        return False

    def drop_images(self, instance):
        template = '<img alt="embedded" src="data:image/jpeg;base64,{}"/>'
        primed = None
        recovery = None
        success = False
        try:
            data = instance.data.open('rb').read()
            upload_data = ChipUploadHelperV2.init_from_serialized(data)
            primed, recovery = upload_data.get_drop_images()

            primed = base64.b64encode(primed).decode('ascii')
            recovery = base64.b64encode(recovery).decode('ascii')
            success = True
        except Exception as e:
            log.exception('Error generating images')

        if success:
            ret = mark_safe(template.format(primed) + template.format(recovery))
        else:
            ret = mark_safe('<p>Chip Upload Contains Invalid Data</p>')

        return ret

    def download_images_link(self, instance):
        if instance.id is None:
            return ''

        url = reverse('download-raw-images', kwargs={'row_id': instance.id})

        return format_html('<a href="{}">Download Images</a>', url)

    def run_cv_code_action(modeladmin, request, queryset):
        # Run CV Code on Selected Items
        count = 0
        for n in queryset:
            count += 1
            run_cv_code.delay(n.id)
        modeladmin.message_user(request, 'Running CV Code for {} entries'.format(count))

    run_cv_code_action.short_description = 'Run CV code on selected entries'


@admin.register(CVModels)
class CVModelsAdmin(admin.ModelAdmin):
    pass


@admin.register(CVImageMasks)
class CVImageMasksAdmin(admin.ModelAdmin):
    list_display = ('id', 'set_id', 'camera', 'chip_position', 'mask', 'create_tstamp', 'update_tstamp')
