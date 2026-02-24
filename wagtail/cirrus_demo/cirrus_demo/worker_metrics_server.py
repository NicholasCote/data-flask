"""HTTP server to expose Celery worker metrics."""
from prometheus_client import start_http_server
import logging
import time

logger = logging.getLogger(__name__)

def start_metrics_server(port=9090):
    """Start Prometheus metrics HTTP server."""
    try:
        start_http_server(port)
        logger.info(f"Metrics server started on port {port}")
        while True:
            time.sleep(1)
    except Exception as e:
        logger.error(f"Failed to start metrics server: {e}")