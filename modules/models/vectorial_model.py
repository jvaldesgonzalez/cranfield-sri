import json
import math

import modules.data_processor.normalizer as normalizer
import modules.data_processor.tokenizer as tokenizer


class VectorialModel:
    def __init__(self, alpha=0.5, recover_amount=10):
        self.alpha = alpha
        self.recover_amount = recover_amount
        self.terms = {}
        self.data_size = 0

    def add_data(self, items: list):
        self.data_size = len(items)
        total_terms_by_id = {}
        for item in items:
            normalized_title = normalizer.normalize(
                tokenizer.tokenize(item.title))
            normalized_text = normalizer.normalize(
                tokenizer.tokenize(item.text))
            normalized_author = normalizer.normalize(
                tokenizer.tokenize(item.author))
            normalized_bib = normalizer.normalize(tokenizer.tokenize(item.bib))

            total_terms_by_id[item.id] = (len(normalized_title) +
                                          len(normalized_text) +
                                          len(normalized_author) +
                                          len(normalized_bib))

            self.add(item.id, normalized_title)
            self.add(item.id, normalized_text)
            self.add(item.id, normalized_author)
            self.add(item.id, normalized_bib)

        # total = len(self.terms)
        # print(total)

        for term, docs in self.terms.items():
            for doc, freq in docs.items():
                tf = freq / total_terms_by_id[doc]
                idf = math.log(len(items)/len(docs))
                docs[doc] = (tf * idf)
            # print(term)
            # print(docs)
            # print("+++++++++++++++++++++++++++++++++")

    def add(self, id, words):
        for w in words:
            if w in (self.terms):
                try:
                    self.terms[w][id] += 1
                except KeyError:
                    self.terms[w][id] = 1
            else:
                self.terms[w] = {id: 1}

    def make_query(self, query):
        normalized_query = normalizer.normalize(tokenizer.tokenize(query))

        query_vector = {}
        for w in normalized_query:
            if w not in self.terms:
                continue
            try:
                query_vector[w] += 1
            except KeyError:
                query_vector[w] = 1

        for w, freq in query_vector.items():
            if w not in self.terms:
                continue
            tf = freq / len(normalized_query)
            idf = math.log(self.data_size/len(self.terms[w]))
            query_vector[w] = (self.alpha + (1 - self.alpha)) * (tf * idf)

        rank = {}
        for term, q_weight in query_vector.items():
            for doc, d_weight in self.terms[term].items():
                try:
                    rank[doc] += q_weight * d_weight
                except KeyError:
                    rank[doc] = q_weight * d_weight

        rank = [(doc, w) for doc, w in rank.items()]
        rank.sort(key=lambda x: x[1], reverse=True)

        rank = rank[: self.recover_amount]

        print(rank)
        return list(map(lambda x: x[0], rank))

    def save(self, path):
        with open(f'{path}/data_from_vectorial_model.json', 'w+') as f:
            f.write(json.dumps({
                'alpha': self.alpha,
                'recover_amount': self.recover_amount,
                'terms': self.terms,
                'data_size': self.data_size
            }))

            f.close()

    def load(self, path):
        with open(f'{path}/data_from_vectorial_model.json', 'r') as f:
            data = json.load(f)
            self.alpha = data['alpha']
            self.recover_amount = data['recover_amount']
            self.terms = data['terms']
            self.data_size = data['data_size']
