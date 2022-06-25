import os

from flask import Flask


app = Flask(__name__)

if os.environ.get('MODE') == 'PRODUCTION':
    app.config.from_object('modules.web.config.ProductionConfig')
else:
    app.config.from_object('modules.web.config.DevelopmentConfig')
