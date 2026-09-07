import os

# Automatically bind to 0.0.0.0:$PORT assigned by Render
port = os.environ.get("PORT", "10000")
bind = f"0.0.0.0:{port}"
workers = 1
threads = 2
timeout = 120
max_requests = 250
max_requests_jitter = 25
accesslog = "-"
errorlog = "-"
