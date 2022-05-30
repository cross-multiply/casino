import logging
import io
import json

from threading import local
from datetime import datetime, timedelta, timezone

from django.db import connection
from django.core.files.base import ContentFile

from celery import shared_task
from celery.utils.log import get_task_logger

from chipcounting.api.models import ChipUploads, ChipEventsDebug, ChipEvents, CVImageMasks, CVModels, Chips
from chipcounting.image_processing.utils import ChipUploadHelper, DebugChipDataHelper, ChipUploadHelperV2
from chipcounting.image_processing.video_analysis import SlidePullManager, VideoFrameProcessor
from chipcounting.image_processing.config import get_local_config, reinit_local

from chipcounting.image_processing.masked_model import MaskedModel, h5py_binary_to_keras

log = get_task_logger(__name__)


@shared_task
def cleanup_old_uploads(days=60):
    start_time = datetime.now(timezone.utc) - timedelta(days=days)
    log.info("Deleting Objects Older Than {}".format(start_time.isoformat()))
    objs = ChipUploads.objects.filter(create_tstamp__lte=start_time)
    num = 0
    for n in objs:
        objs.delete()
        num += 1
    # Apparently a bulk delete may not trigger the delete hooks
    # num, _ = objs.delete()
    log.info("Deleted {} objects".format(num))


@shared_task
def cleanup_old_chip_debug_uploads(days=60):
    start_time = datetime.now(timezone.utc) - timedelta(days=days)
    log.info("Deleting Chip Debug Objects Older Than {}".format(start_time.isoformat()))
    objs = ChipEventsDebug.objects.filter(create_tstamp__lte=start_time)
    num = 0
    for n in objs:
        objs.delete()
        num += 1
    # Apparently a bulk delete may not trigger the delete hooks
    # num, _ = objs.delete()
    log.info("Deleted {} objects".format(num))


@shared_task
def check_for_unprocessed(min_age=20*60, max_age=24*60*60):
    start_time = datetime.now(timezone.utc) - timedelta(seconds=min_age)
    end_time = datetime.now(timezone.utc) - timedelta(seconds=max_age)

    sql = """SELECT cu.id FROM chip_uploads cu
    LEFT JOIN chip_events_debug ced ON cu.id = ced.chip_upload_id
    WHERE cu.create_tstamp >= %s
    AND cu.create_tstamp <= %s
    AND ced.id is NULL
    """

    ids_ = []
    with connection.cursor() as cursor:
        cursor.execute(sql, [end_time, start_time])
        ids_ = [x[0] for x in cursor.fetchall()]

    log.info("Found {} Uploads That Were Never Processed: {}".format(len(ids_), ids_))

    for i in ids_:
        run_cv_code.delay(i)


@shared_task
def run_cv_code_old(db_id):
    log.info("Running CV Code for ID: {}".format(db_id))
    obj = ChipUploads.objects.get(pk=db_id)
    try:
        reinit_local()
        cv_params = obj.device.get_computer_vision_json_or_none()
        if cv_params:
            log.info("Found CV Config: {}".format(cv_params))
            thread_local = get_local_config()
            thread_local.cv_params = cv_params

        cuh = ChipUploadHelper.init_from_serialized(obj.data.read())
        image_groupings = cuh.image_groupings

        # Rerun Slide Pull
        spm = SlidePullManager()
        valid, new, slide_idx = spm.single_pass_run(image_groupings)

        if valid:
            result = 'PROCESSED'
            dcdh = DebugChipDataHelper(new, slide_idx)
            _ = dcdh.color_calibrate_drop()
            data_tmp = dcdh.generate_data_field()
            data = ContentFile(data_tmp)
        else:
            result = 'EMPTY'
            data = None

        ChipEventsDebug.objects.filter(chip_upload=obj).delete()
        debug = ChipEventsDebug(
            chip_upload=obj,
            unique_id=obj.unique_id,
            table=obj.table,
            result=result,
            upload_tstamp=obj.tstamp,
        )
        if data is not None:
            debug.data.save('data.zlib', data)
        debug.save()
    except Exception as e:
        debug = ChipEventsDebug(
            chip_upload=obj,
            table=obj.table,
            unique_id=obj.unique_id,
            result='ERROR',
            upload_tstamp=obj.tstamp,
            data=None,
            message=str(e)
        )
        debug.save()


@shared_task
def run_cv_code(db_id):
    log.info("Running CV Code for ID: {}".format(db_id))
    obj = ChipUploads.objects.get(pk=db_id)
    try:
        reinit_local()
        cv_params = obj.device.get_computer_vision_json_or_none()

        cv_model_id = 1
        cv_mask_set_id = 1

        if cv_params:
            log.info("Found CV Config: {}".format(cv_params))
            thread_local = get_local_config()
            thread_local.cv_params = cv_params
            cv_model_id = int(cv_params.get('cv_model_id', 1))
            cv_mask_set_id = int(cv_params.get('cv_mask_set_id', 1))

        cuh = ChipUploadHelperV2.init_from_serialized(obj.data.read())
        top_image_groupings = cuh.top_image_groupings

        # Rerun Slide Pull
        spm = SlidePullManager()
        valid, new, slide_idx = spm.single_pass_run(top_image_groupings)

        message = ''

        # Commenting out the slide check and Color Correction for now, until we
        # have a better definition of configuration formats
        if True:  #valid:
            primed_imgs, _ = cuh.get_cv_images()

            model = CVModels.objects.get(pk=cv_model_id)
            masks = CVImageMasks.objects.filter(set_id=cv_mask_set_id)
            masks_top = {}
            masks_bot = {}
            for m in masks:
                pos_id = m.chip_position
                mask = m.mask_numpy_format()
                if m.camera == 'top':
                    masks_top[pos_id] = mask
                elif m.camera == 'bottom':
                    masks_bot[pos_id] = mask
                else:
                    log.error("Unknown mask", m.camera, m.id)

            model_keras = h5py_binary_to_keras(model.data.read())
            model_labels = sorted(json.loads(model.labels))
            cv_masked_model = MaskedModel(model_keras, model_labels, masks_bot, masks_top)

            predictions, raw_prob = cv_masked_model.process_cv(primed_imgs[0], primed_imgs[1])
            num_chips, labels = cv_masked_model.post_process_output(predictions)
            result = 'PROCESSED'

            message = '{} chips\n{}\n\n'.format(num_chips, labels)
            for n in raw_prob:
                message += '{} => {}\n\n'.format(n, raw_prob[n])
            chips = Chips.objects.all()
            log.info('\nNM ------------ Chips {}\n'.format(chips))
            ChipEvents.objects.filter(chip_upload=obj).delete()
            for i in labels:
                label = labels[i][0]
                chip = chips.filter(name=label)
                if len(chip) > 0:
                    log.info('\nNM -----{}------- Chip {}\n'.format(label, chip.get()))
                    event = ChipEvents(
                        unique_id=obj.unique_id,
                        chip_upload=obj,
                        table=obj.table,
                        tstamp=obj.tstamp,
                        results=label,
                        amount=chip.get().value
                    )
                    event.save()
        else:
            result = 'EMPTY'

        ChipEventsDebug.objects.filter(chip_upload=obj).delete()
        debug = ChipEventsDebug(
            chip_upload=obj,
            unique_id=obj.unique_id,
            table=obj.table,
            result=result,
            upload_tstamp=obj.tstamp,
            message=message
        )
        debug.save()
    except Exception as e:
        debug = ChipEventsDebug(
            chip_upload=obj,
            table=obj.table,
            unique_id=obj.unique_id,
            result='ERROR',
            upload_tstamp=obj.tstamp,
            data=None,
            message=str(e)
        )
        debug.save()

