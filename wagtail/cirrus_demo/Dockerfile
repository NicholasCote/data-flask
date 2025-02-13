# Use an official Python runtime based on Debian 12 "bookworm" as a parent image.
FROM python:3.12-slim-bookworm

# Add user that will be used in the container.
RUN useradd wagtail

# Port used by this container to serve HTTP.
EXPOSE 8000

# Set environment variables.
ENV PYTHONUNBUFFERED=1 \
    PORT=8000 \
    DJANGO_SETTINGS_MODULE=cirrus_demo.settings.production \
    PYTHONPATH=/app

# Install system packages required by Wagtail and Django.
RUN apt-get update --yes --quiet && apt-get install --yes --quiet --no-install-recommends \
    build-essential \
    libpq-dev \
    libmariadb-dev \
    libjpeg62-turbo-dev \
    zlib1g-dev \
    libwebp-dev \
 && rm -rf /var/lib/apt/lists/*

# Use /app folder as a directory where the source code is stored.
WORKDIR /app

# Copy the requirements first to leverage Docker cache
COPY requirements.txt /app/
RUN pip install -r requirements.txt
RUN pip install "gunicorn==20.1.0"

# Copy the source code of the project into the container.
COPY --chown=wagtail:wagtail . /app/

# Set directory ownership
RUN chown -R wagtail:wagtail /app

# Create the entrypoint script
COPY --chown=wagtail:wagtail <<'EOF' /app/docker-entrypoint.sh
#!/bin/bash
set -e

# Function to run the web server
run_web() {
    echo "Starting web server..."
    python manage.py migrate --noinput
    exec gunicorn cirrus_demo.wsgi:application \
        --bind 0.0.0.0:8000 \
        --workers 3 \
        --timeout 120 \
        --log-level debug \
        --access-logfile - \
        --error-logfile - \
        --capture-output \
        --enable-stdio-inheritance
}

# Function to run the Celery worker
run_worker() {
    echo "Starting Celery worker..."
    python manage.py migrate --noinput
    exec celery -A cirrus_demo worker -l info
}

# Function to run Flower dashboard
run_flower() {
    echo "Starting Flower dashboard..."
    exec celery -A cirrus_demo flower --port=5555 --url_prefix=flower
}

# Check command argument
case "$1" in
    "web")
        run_web
        ;;
    "worker")
        run_worker
        ;;
    "flower")
        run_flower
        ;;
    *)
        echo "Usage: $0 {web|worker|flower}"
        exit 1
        ;;
esac
EOF

RUN chmod +x /app/docker-entrypoint.sh

# Use user "wagtail" to run the commands
USER wagtail

# Set the entrypoint
ENTRYPOINT ["/app/docker-entrypoint.sh"]

# Default to web mode if no argument is provided
CMD ["web"]