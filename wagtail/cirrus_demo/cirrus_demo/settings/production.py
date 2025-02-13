from .base import *

DEBUG = True

ALLOWED_HOSTS = ['*', 'localhost', '127.0.0.1']

SECRET_KEY="maybe this will work one time"

CELERY_BROKER_URL = 'redis://redis:6379/0'  # You'll need Redis installed

LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'level': 'DEBUG',
        },
    },
    'root': {
        'handlers': ['console'],
        'level': 'DEBUG',
    },
    'loggers': {
        'django': {
            'handlers': ['console'],
            'level': 'DEBUG',
            'propagate': True,
        },
        'gunicorn.error': {
            'level': 'DEBUG',
            'handlers': ['console'],
            'propagate': True,
        },
    },
}

try:
    from .local import *
except ImportError:
    pass
