FROM python:3.9

WORKDIR /app

RUN git clone --no-checkout --branch dev https://github.com/NicholasCote/data-flask.git && \
    cd data-flask && \
    git sparse-checkout init && \
    git sparse-checkout set flask-code && \
    git checkout

WORKDIR /app/data-flask/flask-code

RUN pip install -r requirements.txt

CMD ["python3", "./wsgi.py"]