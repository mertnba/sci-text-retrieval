import numpy as np
from gensim.models import KeyedVectors
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string

# Initialize tools
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

def custom_tokenizer(text):
    """
    Tokenizes, lemmatizes, and removes stopwords and punctuation from a given text.

    Args:
        text (str): Input text.

    Returns:
        list: List of processed tokens.
    """
    tokens = word_tokenize(text.lower())
    return [
        lemmatizer.lemmatize(token)
        for token in tokens
        if token not in stop_words and token not in string.punctuation
    ]

def average_sentence_vector(sentence, model):
    """
    Computes the average word embedding for a given sentence using a pre-trained model.

    Args:
        sentence (str): Input sentence.
        model (KeyedVectors): Pre-trained word embedding model.

    Returns:
        np.ndarray: Average word embedding.
    """
    tokens = custom_tokenizer(sentence)
    word_vectors = [model[word] for word in tokens if word in model.key_to_index]
    if word_vectors:
        return np.mean(word_vectors, axis=0)
    return np.zeros(model.vector_size)

def generate_embeddings(data, model_path):
    """
    Generates embeddings for all queries and passages in the dataset.

    Args:
        data (DataFrame): Input dataset with 'query' and 'passage' columns.
        model_path (str): Path to the pre-trained word embedding model.

    Returns:
        dict: Dictionary containing query and passage embeddings.
    """
    model = KeyedVectors.load(model_path, mmap="r")
    data["query_embeddings"] = data["query"].apply(lambda x: average_sentence_vector(x, model))
    data["passage_embeddings"] = data["passage"].apply(lambda x: average_sentence_vector(x, model))
    return data

def compute_features(data):
    """
    Computes cosine similarity and element-wise product features for ranking models.

    Args:
        data (DataFrame): Dataset with query and passage embeddings.

    Returns:
        DataFrame: Dataset with additional computed features.
    """
    data["cosine_similarity"] = data.apply(
        lambda row: calculate_cosine_similarity(row["query_embeddings"], row["passage_embeddings"]),
        axis=1
    )
    data["elementwise_product"] = data.apply(
        lambda row: row["query_embeddings"] * row["passage_embeddings"],
        axis=1
    )
    return data

def calculate_cosine_similarity(vector_a, vector_b):
    """
    Calculates the cosine similarity between two vectors.

    Args:
        vector_a (np.ndarray): First vector.
        vector_b (np.ndarray): Second vector.

    Returns:
        float: Cosine similarity score.
    """
    dot_product = np.dot(vector_a, vector_b)
    norm_a = np.linalg.norm(vector_a)
    norm_b = np.linalg.norm(vector_b)
    return dot_product / (norm_a * norm_b) if norm_a and norm_b else 0.0
