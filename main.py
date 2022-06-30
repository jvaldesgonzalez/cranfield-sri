import threading

import modules.data_loader as dl
from modules.models.bm25_model import SmartModel
from modules.models.vectorial_model import VectorialModel
from modules.web import run as run_web
import sys

# parse dataset
items = dl.cranfield.parse_raw()
querys = dl.cranfield.parse_raw_query()
print(querys)

# better_titles = list(
#     items
#     | select(lambda x: x.title)
#     | select(tokenizer.tokenize)
#     | select(normalizer.normalize)
# )

vectorial_model = VectorialModel()
smart_model = SmartModel()

if (len(sys.argv) == 3):
    if(sys.argv[1] == "1"):
        vectorial_model.add_data(items)
        vectorial_model.save("./models_saves")
    if(sys.argv[2] == "1"):
        smart_model.fit(items)
        smart_model.train()
        smart_model.save()

else:
    raise ValueError(
        'You most enter two arguments. Value 1 in each will re-calculate the vectorial and bm25 models respectively')


# smart_model = SmartModel()
# try:
#     smart_model.load()
# except:
#     smart_model.fit(items)
#     smart_model.train()
#     smart_model.save()

# smart_model.make_query(
#     "what similarity laws must be obeyed when constructing aeroelastic models of heated high speed aircraft .")

# with plt.xkcd():
#     pd.DataFrame(
#         smart_model.ft_model.wv.most_similar('flow theory',
#                                              topn=10,
#                                              restrict_vocab=5000
#                                              ),
#         columns=['Word', 'Score']
#     ).plot.barh(
#         x='Word',
#         figsize=(6, 6),
#         color=(0.3, 0.7, 0.7)
#     )
#     plt.show()


print("--------------------------------")
# assert len(better_titles) == 1400


# starting flask service as another thread
threading.Thread(target=run_web.run, args=[vectorial_model]).start()
