# run
gunicorn -w 1 -k gevent wsgi:app -b 0.0.0.0:5000