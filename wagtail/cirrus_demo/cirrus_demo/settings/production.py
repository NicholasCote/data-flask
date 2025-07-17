from .base import *

DEBUG = True

ALLOWED_HOSTS = ['*', 'localhost', '127.0.0.1']

SECRET_KEY="maybe this will work one time"

CELERY_BROKER_URL = 'redis://redis:6379/0'

# Wagtail site configuration
WAGTAIL_SITE_NAME = 'NCOTE Celery Demo'

# Auto-create sites for multiple hostnames
def setup_wagtail_sites():
    """Auto-create Wagtail sites if they don't exist"""
    try:
        from wagtail.models import Site, Page
        from django.db import connection
        
        # Check if database is ready
        with connection.cursor() as cursor:
            cursor.execute("SELECT 1")
        
        # Get or create root page
        try:
            root_page = Page.objects.get(depth=2)  # Usually the home page
        except Page.DoesNotExist:
            # If no pages exist, let Wagtail handle initial setup
            return
        
        # Define your hostnames
        hostnames = [
            'ncote-celery-demo.k8s.ucar.edu',
            'ncote-celery-demo-preview.k8s.ucar.edu',
            'localhost',
            '127.0.0.1'
        ]
        
        for hostname in hostnames:
            site, created = Site.objects.get_or_create(
                hostname=hostname,
                defaults={
                    'site_name': f'{WAGTAIL_SITE_NAME} - {hostname}',
                    'root_page': root_page,
                    'is_default_site': hostname == 'ncote-celery-demo.k8s.ucar.edu'
                }
            )
            if created:
                print(f"Created Wagtail site for {hostname}")
    
    except Exception as e:
        print(f"Could not setup Wagtail sites: {e}")

# Run site setup when Django starts
import django
if django.VERSION >= (3, 2):
    try:
        django.setup()
        setup_wagtail_sites()
    except:
        pass

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