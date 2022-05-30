from django.apps import AppConfig


class APIConfig(AppConfig):
    name = 'chipcounting.api'
    verbose_name = 'Casino Administration'

    def ready(self):
        import chipcounting.api.signals
