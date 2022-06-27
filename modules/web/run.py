from modules.web import app
from modules.web import views

PORT = 3000
HOST = '0.0.0.0'  # all network interfaces


# use_reloader set to false because flask can't run on a separate
# Thread with hot_reload enabled
def run(model):
    app.run(host=HOST, port=PORT, use_reloader=False)
