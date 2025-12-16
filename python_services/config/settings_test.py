from .settings import *

DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.postgresql",
        "NAME": "records",  # your existing DB
        "USER": "app",
        "PASSWORD": "dairyman",
        "HOST": "localhost",
        "PORT": 5432,
        "TEST": {
            "MIRROR": "default",
        },
    }
}