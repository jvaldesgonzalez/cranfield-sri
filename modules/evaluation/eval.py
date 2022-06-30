from itertools import chain
import numpy as np

import modules.data_loader as dl
from modules.models.bm25_model import SmartModel
from modules.models.vectorial_model import VectorialModel


def parse_cranrel():
    relevance = {}
    with open('./datasets/cran/cranqrel') as f:
        while True:
            line = f.readline()
            if not line:
                break
            query, doc, *_ = line.split(' ')
            try:
                relevance[int(query)].append(int(doc))
            except:
                relevance[int(query)] = [int(doc)]
    return relevance


def precission(result_indices, qry_index):
    try:
        relevants = relevance[int(qry_index)]
    except KeyError:
        return 0.5
    c = 0
    for r in result_indices:
        if int(r) in relevants:
            c += 1
    return c/10


def recall(result_indices, qry_index):
    try:
        relevants = relevance[int(qry_index)]
    except KeyError:
        return 0.5
    c = 0
    for r in result_indices:
        if int(r) in relevants:
            c += 1
    return c/len(relevants)


relevance = parse_cranrel()

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

queries_with_id = dl.cranfield.parse_raw_query()

precissions_vect = []
precissions_bm = []
recalls_vect = []
recalls_bm = []

for (idq, textq) in queries_with_id:
    v_result_indices = vectorial_model.make_query(textq)
    bm_result_indices = smart_model.make_query(textq)
    precission_vectorial = precission(v_result_indices, idq)
    precission_bm = precission(bm_result_indices[0], idq)

    recall_vectorial = recall(v_result_indices, idq)
    recall_bm = recall(bm_result_indices[0], idq)

    precissions_vect.append(precission_vectorial)
    precissions_bm.append(precission_bm)

    recalls_vect.append(recall_vectorial)
    recalls_bm.append(recall_vectorial)


print("P vec", np.mean(precissions_vect))
print("P bm25", np.mean(precissions_bm))
print("R vec", np.mean(recalls_vect))
print("R bm25", np.mean(recalls_bm))
