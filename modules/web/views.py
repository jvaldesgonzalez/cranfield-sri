from flask import render_template, request

import modules.data_loader as dl
from modules.models.vectorial_model import VectorialModel
from modules.web import app

items = dl.cranfield.parse_raw()
vectorial_model = VectorialModel()

try:
    vectorial_model.load("./models_saves")
except:
    raise Exception('Models not loaded')


@app.route('/', methods=['GET'])
def home():
    q = request.args.get('search', '')
    result_indices = vectorial_model.make_query(q)
    results = list(filter(lambda x: x.id in result_indices, items))
    print(results[:3])
    amount = len(result_indices)

    return render_template('index.html.jinja', context={
        'q': q,
        'results': results,
        "amount": amount
    })
