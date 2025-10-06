import os

bind = f"0.0.0.0:{os.environ.get('PORT', '10000')}"
workers = 2
worker_class = 'sync'
timeout = 180
graceful_timeout = 180
keepalive = 5
accesslog = '-'
errorlog = '-'
loglevel = 'info'