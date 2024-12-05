import numpy as np
from collections import Counter

class BM25Retriever:
    """
    Implements the BM25 retrieval algorithm for ranking documents.
    """

    def __init__(self, k1=1.5, b=0.75):
        """
        Initializes the BM25 retriever.

        Args:
            k1 (float): Term frequency saturation parameter. Default is 1.5.
            b (float): Length normalization parameter. Default is 0.75.
        """
        self.k1 = k1
        self.b = b
        self.doc_lengths = {}
        self.avg_doc_length = 0
        self.inverted_index = {}
        self.idf_values = {}

    def fit(self, corpus):
        """
        Prepares the BM25 model by computing IDF values and document statistics.

        Args:
            corpus (list of list of str): Tokenized documents.
        """
        total_length = 0
        doc_count = len(corpus)
        term_doc_counts = Counter()

        for doc_id, tokens in enumerate(corpus):
            self.doc_lengths[doc_id] = len(tokens)
            total_length += len(tokens)
            unique_tokens = set(tokens)
            for token in unique_tokens:
                term_doc_counts[token] += 1

        self.avg_doc_length = total_length / doc_count
        self.idf_values = {
            term: np.log((doc_count - count + 0.5) / (count + 0.5) + 1)
            for term, count in term_doc_counts.items()
        }

        # Create the inverted index
        for doc_id, tokens in enumerate(corpus):
            for token in tokens:
                if token not in self.inverted_index:
                    self.inverted_index[token] = []
                self.inverted_index[token].append(doc_id)

    def score(self, query_tokens, doc_tokens):
        """
        Computes the BM25 score for a query and a document.

        Args:
            query_tokens (list of str): Tokenized query.
            doc_tokens (list of str): Tokenized document.

        Returns:
            float: BM25 score.
        """
        doc_length = len(doc_tokens)
        term_frequencies = Counter(doc_tokens)
        score = 0

        for term in query_tokens:
            if term not in self.idf_values:
                continue
            tf = term_frequencies[term]
            idf = self.idf_values[term]
            tf_normalized = ((tf * (self.k1 + 1)) /
                             (tf + self.k1 * (1 - self.b + self.b * (doc_length / self.avg_doc_length))))
            score += idf * tf_normalized

        return score

    def retrieve(self, queries, candidate_docs):
        """
        Retrieves the best matching documents for each query.

        Args:
            queries (dict): Dictionary where keys are query IDs and values are tokenized queries.
            candidate_docs (dict): Dictionary where keys are doc IDs and values are tokenized documents.

        Returns:
            dict: Dictionary of ranked documents for each query.
        """
        results = {}
        for query_id, query_tokens in queries.items():
            scores = {
                doc_id: self.score(query_tokens, doc_tokens)
                for doc_id, doc_tokens in candidate_docs.items()
            }
            ranked_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            results[query_id] = ranked_docs
        return results
