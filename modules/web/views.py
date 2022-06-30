from itertools import chain

from flask import render_template, request

import modules.data_loader as dl
from modules.models.vectorial_model import VectorialModel
from modules.models.bm25_model import SmartModel
from modules.web import app

items = dl.cranfield.parse_raw()
vectorial_model = VectorialModel()

try:
    vectorial_model.load("./models_saves")
except:
    vectorial_model.add_data(items)
    vectorial_model.save("./models_saves")

smart_model = SmartModel()
try:
    smart_model.load()
except:
    smart_model.fit(items)
    smart_model.train()
    smart_model.save()


@app.route('/', methods=['GET'])
def home():
    q = request.args.get('search', 'ariplane')
    model = request.args.get('model', 'vec')

    v_result_indices = vectorial_model.make_query(q)
    v_results = list(chain(*[list(filter(lambda x: x.id == id, items))
                             for id in v_result_indices]))
    v_amount = len(v_result_indices)

    bm_result_indices = smart_model.make_query(q)
    bm_results = [items[i] for i in bm_result_indices[0]]
    bm_amount = len(bm_results)

    return render_template('index.html.jinja', context={
        'q': q,
        'results': bm_results if model == 'bm25' else v_results,
        "amount": bm_amount if model == 'bm25' else v_amount,
        "model_used": 'bm25' if model == 'bm25' else 'vectorial'
    })
