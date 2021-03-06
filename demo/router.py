#coding=utf-8
from flask import jsonify, current_app, request
from main import app
import os
import random
import math
import json
from predict import getToken, getClass

@app.route('/api/predict', methods=['GET'])
def captcha():
    current_app.lock.acquire()
    result = {}
    requests = dict(request.args)
    try:
        text = requests['text']
        label0 = requests['label0']
        label1 = requests['label1']
        print(text, label0, label1)
        # Just for demo
        res = getToken(text, label0, label1)
        result = {'info': 'success', 'res': res}  
    except Exception as e:
        print('ERROR:', e)
        return jsonify(result)
    finally:
        current_app.lock.release()
        return jsonify(result)


@app.route('/api/classification', methods=['GET'])
def captchaa():
    current_app.lock.acquire()
    result = {}
    print(dict(request.args))
    requests  = dict(request.args)
    print(requests)
    try:
        text = requests['text']
        print(text)
        # Just for demo
        res = getClass(text)
        result = {'info': 'success', 'label': res}  
    except Exception as e:
        print('ERROR:', e)
        return jsonify(result)
    finally:
        current_app.lock.release()
        return jsonify(result)
