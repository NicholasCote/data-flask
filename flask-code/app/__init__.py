import secrets
from pathlib import Path

from flask import Flask, url_for
from flask_session import Session
from flask_github import GitHub

from flask_sqlalchemy import SQLAlchemy

import os

app = Flask(__name__)
app.app_context().push()

DATABASE_URI = 'sqlite:///github-flask.db'
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

app.config["SESSION_PERMANENT"] = False
app.config['SESSION_TYPE'] = 'filesystem'
Session(app)

app.config['GITHUB_CLIENT_ID'] = os.environ['NCOTE_GITHUB_OAUTH_ID']
app.config['GITHUB_CLIENT_SECRET'] = os.environ['NCOTE_GITHUB_OAUTH_SECRET']

github = GitHub(app)

app.config['SQLALCHEMY_DATABASE_URI'] = DATABASE_URI

db = SQLAlchemy(app)

@app.before_request
def create_tables():
    db.create_all()

class User(db.Model):
    __tablename__ = 'users'

    id = db.Column(db.Integer, primary_key=True)
    github_access_token = db.Column(db.String(255))
    github_id = db.Column(db.Integer)
    github_login = db.Column(db.String(255))

    def __init__(self, github_access_token):
        self.github_access_token = github_access_token

from app.main import views
