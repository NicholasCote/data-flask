import os
from celery import Celery

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'cirrus_demo.settings.base')

app = Celery('cirrus_demo')
app.config_from_object('cirrus_demo.settings.base', namespace='CELERY')
app.autodiscover_tasks()