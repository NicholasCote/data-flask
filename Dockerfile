FROM python:3.9

RUN apt-get update

RUN git clone https://github.com/NicholasCote/Stratus-Python.git

WORKDIR data-flask/app

RUN pip install -r requirements.txt

CMD [ "python3", "./wsgi.py" ]