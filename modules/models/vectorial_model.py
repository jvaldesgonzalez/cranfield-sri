import modules.data_processor.normalizer as normalizer
import modules.data_processor.tokenizer as tokenizer
from modules.utils.levenshtein import levenshtein
import math
import json


class VectorialModel:
    def __init__(self, alpha=0.5, recover_amount=10):
        self.alpha = alpha
        self.recover_amount = recover_amount
        self.terms = {}
        self.data_size = 0
        self.items = {}
        self.feedback = {}

    def add_data(self, items: list):
        self.data_size = len(items)
        total_terms_by_id = {}
        for item in items:
            self.items[item.id] = item
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

        for _, docs in self.terms.items():
            for doc, freq in docs.items():
                tf = freq / total_terms_by_id[doc]
                idf = math.log(len(items)/len(docs))
                docs[doc] = (tf * idf)

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

        # temp = []
        # for w in normalized_query:
        #     temp += self.get_nearest_words(w)

        # normalized_query = temp

        # temp = []
        # for w in normalized_query:
        #     _, term = self.get_nearest_word(w)
        #     if term != "":
        #         temp.append(term)

        # normalized_query = temp

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
            print('data_size', self.data_size)
            try:
                idf = math.log(self.data_size[0]/len(self.terms[w]))
            except:
                idf = math.log((self.data_size)/len(self.terms[w]))
            query_vector[w] = (self.alpha + ((1 - self.alpha)) * tf) * idf

        rank = {}
        for term, q_weight in query_vector.items():
            for doc, d_weight in self.terms[term].items():
                try:
                    rank[doc] += q_weight * d_weight
                except KeyError:
                    rank[doc] = q_weight * d_weight

                if (term in self.feedback and doc in self.feedback[term]):
                    rank[doc] += self.feedback[term][doc] * 0.3

        rank = [(doc, w) for doc, w in rank.items()]
        rank.sort(key=lambda x: x[1], reverse=True)

        rank = rank[: self.recover_amount]

        print(rank)
        return list(map(lambda x: x[0], rank))

    def get_nearest_words(self, word):
        result = []
        for term, _ in self.terms.items():
            dist = levenshtein(term, word)
            if dist < 3:
                result.append(term)
        return result

    def get_nearest_word(self, word):
        if word in self.terms:
            return (0, word)
        min = 3
        result = ""
        for term, _ in self.terms.items():
            dist = levenshtein(term, word)
            if dist < min:
                dist = min
                result = term
        return (dist, result)

    def save(self, path):
        with open(f'{path}/data_from_vectorial_model.json', 'w+') as f:
            f.write(json.dumps({
                'alpha': self.alpha,
                'recover_amount': self.recover_amount,
                'terms': self.terms,
                'data_size': self.data_size,
                'feedback': self.feedback
            }))

            f.close()

    def load(self, path):
        with open(f'{path}/data_from_vectorial_model.json', 'r') as f:
            data = json.load(f)
            self.alpha = data['alpha']
            self.recover_amount = data['recover_amount']
            self.terms = data['terms']
            self.data_size = data['data_size'],
            self.feedback = data['feedback']

    def incress_score(self, query, doc_id):
        normalized_query = normalizer.normalize(tokenizer.tokenize(query))

        for w in normalized_query:
            if w in self.feedback:
                try:
                    self.feedback[w][doc_id] += 1
                except KeyError:
                    self.feedback[w][doc_id] = 1
            else:
                self.feedback[w] = {doc_id: 1}
