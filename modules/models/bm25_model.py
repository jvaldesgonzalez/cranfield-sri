from gensim.models.fasttext import FastText
from gensim.models.phrases import ENGLISH_CONNECTOR_WORDS, Phrases
import pickle
import numpy as np
from rank_bm25 import BM25Okapi
from tqdm import tqdm

from modules.data_loader.cranfield import CranItem
from modules.data_processor.normalizer import normalize
from modules.data_processor.tokenizer import tokenize
import nmslib


class SmartModel:
    def __init__(self):
        self.tokenized_text = []
        self.weighted_doc_vects = []

    def fit(self, items: list[CranItem]):
        tok_text = [normalize(tokenize(item.fulltext)) for item in items]
        self.tokenized_text = tok_text

    def train(self, use_phraser=True, auto_save=True):
        if use_phraser:
            phrase_model = Phrases(
                self.tokenized_text, min_count=3, connector_words=ENGLISH_CONNECTOR_WORDS)
            self.phrase_model = phrase_model.freeze()

        ft_model = FastText(
            sg=1,  # skip-gram
            window=10,
            min_count=5,
            negative=15,
            min_n=2,
            max_n=5,
            vector_size=100,
        )

        ft_model.build_vocab(self.tokenized_text)
        ft_model.train(
            self.tokenized_text,
            epochs=6,
            total_examples=ft_model.corpus_count,
            total_words=ft_model.corpus_total_words,
        )
        self.ft_model = ft_model

        bm25 = BM25Okapi(self.tokenized_text)
        for i, doc in tqdm(enumerate(self.tokenized_text)):
            doc_vector = []
            for word in doc:
                vector = self.ft_model.wv[word]
                weight = (bm25.idf[word] * ((bm25.k1 + 1.0)*bm25.doc_freqs[i][word])) / (bm25.k1 * (
                    1.0 - bm25.b + bm25.b * (bm25.doc_len[i]/bm25.avgdl))+bm25.doc_freqs[i][word])
                weighted_vector = vector * weight
                doc_vector.append(weighted_vector)
            doc_vector_mean = np.mean(doc_vector, axis=0)
            self.weighted_doc_vects.append(doc_vector_mean)

        if(auto_save):
            self.save()

        self.build_index()

    def make_phrases(self):
        bigram = self.phrase_model
        self.tokenized_text = bigram[self.tokenized_text]

    def build_index(self):
        data = np.vstack(self.weighted_doc_vects)
        index = nmslib.init(method='hnsw', space='cosinesimil')
        index.addDataPointBatch(data)
        index.createIndex({'post': 2}, print_progress=True)
        self.index = index

    def save(self):
        self.ft_model.save('./models_saves/_fasttext.model')
        self.phrase_model.save('./models_saves/phrase_model.pkl')
        pickle.dump(self.weighted_doc_vects, open(
            "./models_saves/weighted_doc_vects.p", 'wb'))

    def load(self):
        print("loading fastext...")
        self.ft_model = FastText.load(
            './models_saves/_fasttext.model')
        print("loading phraser...")
        self.phrase_model = Phrases.load('./models_saves/phrase_model.pkl')
        print("loading doc vects...")
        with open("./models_saves/weighted_doc_vects.p", 'rb') as f:
            self.weighted_doc_vects = pickle.load(f)
            print("building index...")
            self.build_index()
            print('done!')

    def make_query(self, raw_query):
        input = normalize(tokenize(raw_query))
        query = [self.ft_model.wv[vec] for vec in input]
        query = np.mean(query, axis=0)
        print('querying...')
        ids, distances = self.index.knnQuery(query, k=10)
        print(ids, distances)
        return ids, distances
