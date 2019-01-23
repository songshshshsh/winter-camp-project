from flask import Flask
import os

app = Flask(__name__, static_folder='dist', template_folder='dist', static_url_path='/dist')

from router import *

if __name__ == '__main__':
    app.run()
