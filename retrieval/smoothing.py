import numpy as np
from collections import Counter

class LaplaceSmoothing:
    """
    Implements Laplace Smoothing for query likelihood language models.
    """

    def __init__(self, vocabulary_size):
        """
        Initializes the Laplace Smoothing model.

        Args:
            vocabulary_size (int): Total number of unique terms in the corpus.
        """
        self.vocabulary_size = vocabulary_size

    def score(self, query_tokens, doc_tokens):
        """
        Computes the Laplace smoothed score for a query and a document.

        Args:
            query_tokens (list of str): Tokenized query.
            doc_tokens (list of str): Tokenized document.

        Returns:
            float: Smoothed log-likelihood score.
        """
        doc_length = len(doc_tokens)
        term_frequencies = Counter(doc_tokens)
        score = 0.0

        for term in query_tokens:
            term_count = term_frequencies.get(term, 0)
            term_probability = (term_count + 1) / (doc_length + self.vocabulary_size)
            score += np.log(term_probability)

        return score


class DirichletSmoothing:
    """
    Implements Dirichlet Smoothing for query likelihood language models.
    """

    def __init__(self, collection_tokens, mu=2000):
        """
        Initializes the Dirichlet Smoothing model.

        Args:
            collection_tokens (list of str): All tokens in the corpus.
            mu (float): Smoothing parameter. Default is 2000.
        """
        self.collection_length = len(collection_tokens)
        self.collection_frequencies = Counter(collection_tokens)
        self.mu = mu

    def score(self, query_tokens, doc_tokens):
        """
        Computes the Dirichlet smoothed score for a query and a document.

        Args:
            query_tokens (list of str): Tokenized query.
            doc_tokens (list of str): Tokenized document.

        Returns:
            float: Smoothed log-likelihood score.
        """
        doc_length = len(doc_tokens)
        term_frequencies = Counter(doc_tokens)
        score = 0.0

        for term in query_tokens:
            term_count = term_frequencies.get(term, 0)
            collection_count = self.collection_frequencies.get(term, 0)
            term_probability = (term_count + self.mu * (collection_count / self.collection_length)) / (doc_length + self.mu)
            score += np.log(term_probability)

        return score
