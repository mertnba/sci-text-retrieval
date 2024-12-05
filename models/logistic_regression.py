import numpy as np

class LogisticRegressionRanker:
    """
    Implements a custom logistic regression model for ranking.
    """

    def __init__(self, learning_rate=0.01, max_iter=1000, tol=1e-6, verbose=False):
        """
        Initializes the logistic regression ranker.

        Args:
            learning_rate (float): Learning rate for gradient descent.
            max_iter (int): Maximum number of iterations for training.
            tol (float): Tolerance for convergence.
            verbose (bool): Whether to print progress during training.
        """
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self.weights = None

    def sigmoid(self, z):
        """
        Computes the sigmoid function.

        Args:
            z (np.ndarray): Input array.

        Returns:
            np.ndarray: Sigmoid of input.
        """
        z = np.clip(z, -500, 500)  # Prevent overflow
        return 1 / (1 + np.exp(-z))

    def compute_loss(self, y, h):
        """
        Computes the binary cross-entropy loss.

        Args:
            y (np.ndarray): True labels.
            h (np.ndarray): Predicted probabilities.

        Returns:
            float: Loss value.
        """
        epsilon = 1e-7  # Avoid log(0)
        return -np.mean(y * np.log(h + epsilon) + (1 - y) * np.log(1 - h + epsilon))

    def train(self, X, y):
        """
        Trains the logistic regression model.

        Args:
            X (np.ndarray): Feature matrix.
            y (np.ndarray): Target labels.
        """
        m, n = X.shape
        self.weights = np.zeros(n)
        for iteration in range(self.max_iter):
            z = np.dot(X, self.weights)
            h = self.sigmoid(z)
            gradient = np.dot(X.T, (h - y)) / m
            self.weights -= self.learning_rate * gradient

            if self.verbose and iteration % 100 == 0:
                loss = self.compute_loss(y, h)
                print(f"Iteration {iteration}, Loss: {loss}")

            if np.linalg.norm(gradient) < self.tol:
                break

    def predict_proba(self, X):
        """
        Predicts probabilities for the input data.

        Args:
            X (np.ndarray): Feature matrix.

        Returns:
            np.ndarray: Predicted probabilities.
        """
        return self.sigmoid(np.dot(X, self.weights))

    def rank(self, X, data):
        """
        Ranks the documents for a given query using the trained model.

        Args:
            X (np.ndarray): Feature matrix.
            data (DataFrame): Input data.

        Returns:
            dict: Dictionary of ranked documents for each query.
        """
        data["score"] = self.predict_proba(X)
        ranked_results = (
            data.sort_values(by="score", ascending=False)
            .groupby("qid")[["pid", "score"]]
            .apply(lambda group: group.to_dict("records"))
            .to_dict()
        )
        return ranked_results
