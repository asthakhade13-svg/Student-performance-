import os

# Automatically bind to 0.0.0.0:$PORT assigned by Render
port = os.environ.get("PORT", "10000")
bind = f"0.0.0.0:{port}"
workers = 1
threads = 2
timeout = 120
accesslog = "-"
errorlog = "-"

def post_fork(server, worker):
    """
    Runs safely inside the child worker process after Linux fork().
    Prevents thread loss and mutex deadlocks.
    """
    try:
        from app import initialize_app
        initialize_app()
    except Exception as e:
        server.log.error(f"Error in post_fork initialization: {e}")
