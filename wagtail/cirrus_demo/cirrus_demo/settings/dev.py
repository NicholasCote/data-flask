from .base import *

# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = True

# SECURITY WARNING: keep the secret key used in production secret!
SECRET_KEY = "django-insecure-s$=b^55lxo807)8ssu6s#gq%^bie^23-sc4&90%5%el6w%o0$y"

# SECURITY WARNING: define the correct hosts in production!
ALLOWED_HOSTS = ["*"]

EMAIL_BACKEND = "django.core.mail.backends.console.EmailBackend"

CELERY_BROKER_URL = 'redis://localhost:6379/0'  # You'll need Redis installed

try:
    from .local import *
except ImportError:
    pass
