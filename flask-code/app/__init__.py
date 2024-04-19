import os
import secrets
from pathlib import Path

from flask import Flask
from flask_login import LoginManager, UserMixin
from flask_session import Session
from flask_sqlalchemy import SQLAlchemy

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
app.config['SQLALCHEMY_DATABASE_URI'] = DATABASE_URI
app.config["SESSION_PERMANENT"] = False
app.config['SESSION_TYPE'] = 'filesystem'
app.config['OAUTH2_PROVIDERS'] = {
    # GitHub OAuth 2.0 documentation:
    # https://docs.github.com/en/apps/oauth-apps/building-oauth-apps/authorizing-oauth-apps
    'github': {
        'client_id': os.environ.get('NCOTE_GITHUB_OAUTH_ID'),
        'client_secret': os.environ.get('NCOTE_GITHUB_OAUTH_SECRET'),
        'authorize_url': 'https://github.com/login/oauth/authorize',
        'token_url': 'https://github.com/login/oauth/access_token',
        'redirect_uri': 'https://ncote-test.k8s.ucar.edu/callback/github',
        'userinfo': {
            'url': 'https://api.github.com/user/emails',
            'email': lambda json: json[0]['email'],
        },
        'scopes': ['user:email'],
    },
}

Session(app)
db = SQLAlchemy(app)
login = LoginManager(app)
login.login_view = 'home'

@app.before_request
def create_tables():
    db.create_all()

class User(UserMixin, db.Model):
    __tablename__ = 'users'

    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(64), nullable=False)
    email = db.Column(db.String(64), nullable=True)

from app.main import views
from app.auth import views
from app.nacordex import views
from app.stratus import views