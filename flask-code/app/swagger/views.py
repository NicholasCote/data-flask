from flask import jsonify
from app import app
import os
import json

@app.route('/v1/swagger.json')
def swagger():
    with open('stratus/swagger/v1/swagger.json','r') as json_swag:
        return jsonify(json.load(json_swag))