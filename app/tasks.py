import os
from celery import Celery

from .app import flask_app

def make_celery(app):
    celery = Celery(
        app.import_name,
        backend=app.config['CELERY_RESULT_BACKEND'],
        broker=app.config['CELERY_BROKER_URL']
    )
    celery.conf.update(app.config)

    class ContextTask(celery.Task):
        def __call__(self, *args, **kwargs):
            with app.app_context():
                return self.run(*args, **kwargs)

    celery.Task = ContextTask
    return celery

## celery setup
# app celery uris
local_uri = "redis://127.0.0.1:6379"
flask_app.config.update(CELERY_BROKER_URL=os.environ.get("REDIS_URL", local_uri),
                  CELERY_RESULT_BACKEND=os.environ.get("REDIS_URL", local_uri))
celery = make_celery(flask_app)

@celery.task()
def add_together(a, b):
    return a + b
