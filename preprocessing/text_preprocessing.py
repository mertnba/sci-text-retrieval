import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download necessary NLTK data
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")

# Initialize tools
tokenizer = RegexpTokenizer(r'\w+')
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

def preprocess_text(text):
    """
    Tokenizes, lemmatizes, and removes stopwords from a given text.

    Args:
        text (str): Input text.

    Returns:
        list: List of processed tokens.
    """
    tokens = tokenizer.tokenize(text)
    processed_tokens = [
        lemmatizer.lemmatize(token.lower())
        for token in tokens
        if token.lower() not in stop_words
    ]
    return processed_tokens
