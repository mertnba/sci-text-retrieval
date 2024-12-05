import unittest
from text_preprocessing import preprocess_text, tokenize_text, remove_stopwords
from nltk.corpus import stopwords

class TestTextProcessing(unittest.TestCase):
    def setUp(self):
        """
        Set up common variables for the tests.
        """
        self.text = "This is a simple example sentence, showcasing preprocessing."
        self.tokenized_text = ["this", "is", "a", "simple", "example", "sentence", "showcasing", "preprocessing"]
        self.expected_cleaned_text = ["simple", "example", "sentence", "showcasing", "preprocessing"]
        self.stop_words = set(stopwords.words('english'))

    def test_tokenize_text(self):
        """
        Test the tokenize_text function.
        """
        tokenized = tokenize_text(self.text)
        self.assertEqual(tokenized, self.tokenized_text)

    def test_remove_stopwords(self):
        """
        Test the remove_stopwords function.
        """
        filtered_text = remove_stopwords(self.tokenized_text, self.stop_words)
        self.assertEqual(filtered_text, self.expected_cleaned_text)

    def test_preprocess_text(self):
        """
        Test the preprocess_text function for full preprocessing.
        """
        processed_text = preprocess_text(self.text)
        self.assertEqual(processed_text, self.expected_cleaned_text)

if __name__ == "__main__":
    unittest.main()
