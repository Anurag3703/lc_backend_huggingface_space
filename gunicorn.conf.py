import os

bind = f"0.0.0.0:{os.environ.get('PORT', '10000')}"
workers = 1  # Reduced to 1 for simpler state management
worker_class = 'sync'
timeout = 300  # 5 minutes - long enough for full warmup
graceful_timeout = 300
keepalive = 5
preload_app = False
accesslog = '-'
errorlog = '-'
loglevel = 'info'