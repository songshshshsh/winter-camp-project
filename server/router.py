#coding=utf-8
from flask import jsonify, current_app, request, render_template, send_from_directory
from main import app
import os
import random
import math
import json

@app.route('/<path:path>')
def catch_all(path):
    return send_from_directory('dist/', path)

@app.route('/')
def index():
    return render_template("index.html")