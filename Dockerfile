FROM python:3.9

RUN git clone https://github.com/NicholasCote/data-flask.git

WORKDIR data-flask/flask-code

RUN pip install -r requirements.txt

CMD ["python3", "./wsgi.py"]