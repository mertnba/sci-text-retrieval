import numpy as np
import xgboost as xgb
from sklearn.model_selection import GroupShuffleSplit

class LambdaMART:
    """
    Implements a LambdaMART ranking model using XGBoost.
    """

    def __init__(self, params=None):
        """
        Initializes the LambdaMART model.

        Args:
            params (dict): Hyperparameters for the XGBoost model. Defaults to None.
        """
        if params is None:
            params = {
                "objective": "rank:pairwise",
                "eta": 0.1,
                "max_depth": 6,
                "eval_metric": ["map", "ndcg"],
            }
        self.params = params
        self.model = None

    def fit(self, X, y, groups):
        """
        Trains the LambdaMART model.

        Args:
            X (np.ndarray): Feature matrix.
            y (np.ndarray): Target labels (relevancy scores).
            groups (np.ndarray): Group sizes corresponding to queries.
        """
        dtrain = xgb.DMatrix(X, label=y)
        dtrain.set_group(groups)
        self.model = xgb.train(
            self.params,
            dtrain,
            num_boost_round=100,
            verbose_eval=False,
        )

    def predict(self, X):
        """
        Predicts relevancy scores for the given data.

        Args:
            X (np.ndarray): Feature matrix.

        Returns:
            np.ndarray: Predicted relevancy scores.
        """
        if not self.model:
            raise ValueError("Model has not been trained yet.")
        dtest = xgb.DMatrix(X)
        return self.model.predict(dtest)

    def rank(self, X, data):
        """
        Ranks the documents for a given query using the trained model.

        Args:
            X (np.ndarray): Feature matrix.
            data (DataFrame): Input data.

        Returns:
            dict: Dictionary of ranked documents for each query.
        """
        data["score"] = self.predict(X)
        ranked_results = (
            data.sort_values(by="score", ascending=False)
            .groupby("qid")[["pid", "score"]]
            .apply(lambda group: group.to_dict("records"))
            .to_dict()
        )
        return ranked_results

def prepare_data_for_lambdamart(train_df, feature_columns, y_column, test_size=0.2):
    """
    Prepares training and validation datasets for LambdaMART.

    Args:
        train_df (DataFrame): Input training DataFrame.
        feature_columns (list): List of feature column names.
        y_column (str): Target column name (relevancy scores).
        test_size (float): Proportion of the data to use as the validation set.

    Returns:
        tuple: Training and validation data (X_train, y_train, groups_train, X_validate, y_validate, groups_validate).
    """
    gss = GroupShuffleSplit(test_size=test_size, n_splits=1, random_state=42)
    for train_idx, test_idx in gss.split(train_df, groups=train_df["qid"]):
        train_indices, validate_indices = train_idx, test_idx
        break

    X_train = train_df.iloc[train_indices][feature_columns].values
    y_train = train_df.iloc[train_indices][y_column].values
    groups_train = train_df.iloc[train_indices].groupby("qid").size().to_numpy()

    X_validate = train_df.iloc[validate_indices][feature_columns].values
    y_validate = train_df.iloc[validate_indices][y_column].values
    groups_validate = train_df.iloc[validate_indices].groupby("qid").size().to_numpy()

    return X_train, y_train, groups_train, X_validate, y_validate, groups_validate
