from flask import render_template

from modules.web import app


@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')
