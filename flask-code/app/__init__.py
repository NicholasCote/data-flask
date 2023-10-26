import secrets
from pathlib import Path

from flask import Flask, url_for
from flask_session import Session
from flask_restful import Api
from flask_swagger_ui import get_swaggerui_blueprint

from celery import Celery

app = Flask(__name__)
app.app_context().push()

SECRET_FILE_PATH = Path(".flask_secret")
try:
    with SECRET_FILE_PATH.open("r") as secret_file:
        app_secret_key = secret_file.read()
except FileNotFoundError:
    # Let's create a cryptographically secure code in that file
    with SECRET_FILE_PATH.open("w") as secret_file:
        app_secret_key = secrets.token_hex(32)
        secret_file.write(app_secret_key)

app.config['SECRET_KEY'] = app_secret_key

app.config["CELERY_BROKER_URL"] = 'redis://localhost:6379'
celery = Celery(app.name, broker=app.config["CELERY_BROKER_URL"])
celery.conf.update(app.config)

app.config["SESSION_PERMANENT"] = False
app.config['SESSION_TYPE'] = 'filesystem'
Session(app)

SWAGGER_URL = '/api/docs'
#API_URL = 'https://ncote-test.k8s.ucar.edu/v1/swagger.json'
API_URL = 'http://localhost:8000/v1/swagger.json'

swaggerui_blueprint = get_swaggerui_blueprint(
    SWAGGER_URL,
    API_URL,
    config={
        'app_name': "Stratus API Example"
    }
)

app.register_blueprint(swaggerui_blueprint)

api = Api(app)

from app.main import views
from app.swagger import views, api_routes
