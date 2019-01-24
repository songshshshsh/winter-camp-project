from flask import Flask
from fileLock import Lock
import os

app = Flask(__name__, static_folder='uploads', template_folder='uploads', static_url_path='/api/v1/uploads')

app.lock = Lock.get_file_lock()

from router import *

if __name__ == '__main__':
    app.run()
