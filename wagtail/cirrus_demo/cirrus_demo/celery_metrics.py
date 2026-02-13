"""Prometheus metrics for Celery workers."""
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from celery.signals import (
    task_prerun, task_postrun, task_failure, task_success,
    worker_ready, worker_shutdown
)

# Define metrics
CELERY_TASK_STARTED = Counter(
    'celery_task_started_total',
    'Total number of tasks started',
    ['task_name']
)

CELERY_TASK_COMPLETED = Counter(
    'celery_task_completed_total',
    'Total number of tasks completed successfully',
    ['task_name']
)

CELERY_TASK_FAILED = Counter(
    'celery_task_failed_total',
    'Total number of tasks that failed',
    ['task_name', 'exception']
)

CELERY_TASK_DURATION = Histogram(
    'celery_task_duration_seconds',
    'Task execution time in seconds',
    ['task_name']
)

CELERY_WORKERS_ACTIVE = Gauge(
    'celery_workers_active',
    'Number of active Celery workers'
)


@task_prerun.connect
def task_prerun_handler(sender=None, task_id=None, task=None, **kwargs):
    """Increment counter when task starts."""
    CELERY_TASK_STARTED.labels(task_name=task.name).inc()


@task_success.connect
def task_success_handler(sender=None, result=None, **kwargs):
    """Increment counter when task completes successfully."""
    CELERY_TASK_COMPLETED.labels(task_name=sender.name).inc()


@task_failure.connect
def task_failure_handler(sender=None, task_id=None, exception=None, **kwargs):
    """Increment counter when task fails."""
    exception_name = type(exception).__name__ if exception else 'Unknown'
    CELERY_TASK_FAILED.labels(
        task_name=sender.name,
        exception=exception_name
    ).inc()


@task_postrun.connect
def task_postrun_handler(sender=None, task_id=None, retval=None, state=None, **kwargs):
    """Record task execution time."""
    # Runtime is available in sender.request
    if hasattr(sender.request, 'runtime'):
        CELERY_TASK_DURATION.labels(task_name=sender.name).observe(sender.request.runtime)


@worker_ready.connect
def worker_ready_handler(sender=None, **kwargs):
    """Increment gauge when worker starts."""
    CELERY_WORKERS_ACTIVE.inc()


@worker_shutdown.connect
def worker_shutdown_handler(sender=None, **kwargs):
    """Decrement gauge when worker stops."""
    CELERY_WORKERS_ACTIVE.dec()