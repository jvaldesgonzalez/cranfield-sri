import threading

import modules.data_loader as dl
from modules.models.vectorial_model import VectorialModel
from modules.web import run as run_web

# parse dataset
items = dl.cranfield.parse_raw()

# better_titles = list(
#     items
#     | select(lambda x: x.title)
#     | select(tokenizer.tokenize)
#     | select(normalizer.normalize)
# )

vectorial_model = VectorialModel()

try:
    vectorial_model.load("./models_saves")
except:
    vectorial_model.add_data(items)
    vectorial_model.save("./models_saves")

# vectorial_model.add_data(items)
# vectorial_model.save("./models_saves")

vectorial_model.make_query(
    "what similarity laws must be obeyed when constructing aeroelastic models of heated high speed aircraft .")


print("--------------------------------")
# assert len(better_titles) == 1400


# starting flask service as another thread
threading.Thread(target=run_web.run, args=[vectorial_model]).start()
