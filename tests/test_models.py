import unittest
import numpy as np
import torch
from logistic_regression import CustomLogisticRegression
from passage_ranking_nn import PassageRankingNN
from lambdamart import train_lambdamart_model

class TestLogisticRegression(unittest.TestCase):
    def setUp(self):
        """
        Set up data for testing the logistic regression model.
        """
        self.X_train = np.array([[1, 2], [2, 3], [3, 4]])
        self.y_train = np.array([0, 1, 1])
        self.X_test = np.array([[4, 5], [5, 6]])
        self.expected_predictions = np.array([1, 1])
        self.model = CustomLogisticRegression(learning_rate=0.1, max_iter=100, verbose=False)

    def test_fit_and_predict(self):
        """
        Test the fit and predict methods of CustomLogisticRegression.
        """
        self.model.fit(self.X_train, self.y_train)
        predictions = self.model.predict(self.X_test)
        np.testing.assert_array_equal(predictions, self.expected_predictions)

class TestPassageRankingNN(unittest.TestCase):
    def setUp(self):
        """
        Set up data for testing the neural network model.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = PassageRankingNN(embedding_dim=100).to(self.device)
        self.query_embed = torch.rand((1, 100), device=self.device)
        self.passage_embed = torch.rand((1, 100), device=self.device)

    def test_forward(self):
        """
        Test the forward method of the neural network.
        """
        output = self.model(self.query_embed, self.passage_embed)
        self.assertEqual(output.shape, torch.Size([]))
        self.assertTrue(0 <= output.item() <= 1)

class TestLambdaMART(unittest.TestCase):
    def setUp(self):
        """
        Set up data for testing the LambdaMART model.
        """
        self.X_train = np.random.rand(10, 5)
        self.y_train = np.random.randint(0, 2, size=10)
        self.groups = [5, 5]
        self.X_test = np.random.rand(5, 5)

    def test_train_lambdamart_model(self):
        """
        Test the LambdaMART training function.
        """
        model = train_lambdamart_model(self.X_train, self.y_train, self.groups)
        predictions = model.predict(self.X_test)
        self.assertEqual(len(predictions), len(self.X_test))

if __name__ == "__main__":
    unittest.main()
