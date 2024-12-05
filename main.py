import os
from preprocessing.text_processing import preprocess_text
from preprocessing.feature_engineering import generate_embeddings, compute_features
from retrieval.bm25 import BM25Retriever
from retrieval.smoothing import LaplaceSmoothing, DirichletSmoothing
from models.logistic_regression import LogisticRegressionRanker
from models.lambdamart import LambdaMARTRanker
from models.passage_ranking_nn import NeuralNetworkRanker
from evaluation.metrics import evaluate_mean_average_precision, evaluate_ndcg
from utils.io_operations import load_data, save_results

def main():
    # Paths to data files
    train_data_path = "data/train_data.tsv"
    validation_data_path = "data/validation_data.tsv"
    test_queries_path = "data/test-queries.tsv"
    candidate_passages_path = "data/candidate-passages.tsv"
    fasttext_embeddings_path = "embeddings/fasttext_100d_reduced.kv"

    # Load and preprocess data
    print("Loading and preprocessing data...")
    train_data = load_data(train_data_path)
    validation_data = load_data(validation_data_path)
    test_queries = load_data(test_queries_path)
    candidate_passages = load_data(candidate_passages_path)

    # Generate embeddings
    print("Generating embeddings...")
    train_embeddings, validation_embeddings = generate_embeddings(
        train_data, validation_data, fasttext_embeddings_path
    )

    # Compute features
    print("Computing features...")
    train_features = compute_features(train_embeddings)
    validation_features = compute_features(validation_embeddings)

    # BM25 Retrieval
    print("Running BM25 retrieval...")
    bm25_retriever = BM25Retriever()
    bm25_results = bm25_retriever.retrieve(test_queries, candidate_passages)

    # Logistic Regression
    print("Training Logistic Regression ranker...")
    logistic_ranker = LogisticRegressionRanker()
    logistic_ranker.train(train_features, train_data)
    logistic_results = logistic_ranker.rank(validation_features, validation_data)

    # LambdaMART
    print("Training LambdaMART ranker...")
    lambdamart_ranker = LambdaMARTRanker()
    lambdamart_ranker.train(train_features, train_data)
    lambdamart_results = lambdamart_ranker.rank(validation_features, validation_data)

    # Neural Network
    print("Training Neural Network ranker...")
    nn_ranker = NeuralNetworkRanker()
    nn_ranker.train(train_features, train_data)
    nn_results = nn_ranker.rank(validation_features, validation_data)

    # Evaluate results
    print("Evaluating results...")
    metrics = {
        "BM25": {
            "MAP": evaluate_mean_average_precision(bm25_results, validation_data),
            "NDCG": evaluate_ndcg(bm25_results, validation_data),
        },
        "Logistic Regression": {
            "MAP": evaluate_mean_average_precision(logistic_results, validation_data),
            "NDCG": evaluate_ndcg(logistic_results, validation_data),
        },
        "LambdaMART": {
            "MAP": evaluate_mean_average_precision(lambdamart_results, validation_data),
            "NDCG": evaluate_ndcg(lambdamart_results, validation_data),
        },
        "Neural Network": {
            "MAP": evaluate_mean_average_precision(nn_results, validation_data),
            "NDCG": evaluate_ndcg(nn_results, validation_data),
        },
    }

    # Save results
    print("Saving results...")
    save_results(metrics, "data/results.json")

    print("Pipeline completed successfully!")

if __name__ == "__main__":
    main()
