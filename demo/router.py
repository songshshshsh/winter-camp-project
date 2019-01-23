#coding=utf-8
from flask import jsonify, current_app, request
from main import app
import os
import random
import math
import json
from predict import getToken

@app.route('/api/predict', methods=['POST'])
def captcha():
    current_app.lock.acquire()
    result = {}
    try:
        print(json.loads((request.data).decode('utf-8')))
        d = json.loads((request.data).decode('utf-8'))
        # Just for demo
        res = getToken(d['text'], d['character'])
        result = {'info': 'success', 'res': res}  
    except Exception as e:
        print('ERROR:', e)
        return jsonify(result)
    finally:
        current_app.lock.release()
        return jsonify(result)