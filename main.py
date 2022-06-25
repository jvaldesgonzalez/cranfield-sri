import threading

from pipe import select

import modules.data_loader as dl
import modules.data_processor.normalizer as normalizer
import modules.data_processor.tokenizer as tokenizer
from modules.web import run as run_web

# parse dataset
items = dl.cranfield.parse_raw()

better_titles = list(
    items
    | select(lambda x: x.title)
    | select(tokenizer.tokenize)
    | select(normalizer.normalize)
)

print(better_titles[:5])
assert len(better_titles) == 1400


# starting flask service as another thread
threading.Thread(target=run_web.run).start()
