import logging
from django.conf import settings
from django.db.models.signals import post_save
from django.db import transaction
from django.dispatch import receiver

from .models import ChipUploads
from .tasks import run_cv_code

log = logging.getLogger(__name__)


@receiver(post_save, sender=ChipUploads)
def signal_run_cv_code(sender, instance, **kwargs):
    if settings.AUTORUN_CV_CODE:
        log.info("Running CV Code for {} after commit".format(instance.pk))
        transaction.on_commit(run_cv_code.s(db_id=instance.pk).delay)
    else:
        log.info("CV Code is disable, not running for {}".format(instance.pk))
