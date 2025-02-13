import os
import logging

from django.core.wsgi import get_wsgi_application

logger = logging.getLogger('django.request')

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "cirrus_demo.settings.production")

def debug_application(environ, start_response):
    logger.debug(f"Request received: {environ['REQUEST_METHOD']} {environ['PATH_INFO']}")
    logger.debug(f"HTTP headers: {dict((k,v) for k,v in environ.items() if k.startswith('HTTP_'))}")
    
    return get_wsgi_application()(environ, start_response)

application = debug_application