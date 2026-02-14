from django.apps import AppConfig

class BanditConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "bandit"

class MlServiceConfig(AppConfig):
    name = 'ml_service'

    def ready(self):
        pass