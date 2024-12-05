# Passage Retrieval and Ranking System

This repository implements a modular passage retrieval and ranking system using a variety of retrieval models and machine learning approaches. The system processes text data, builds embeddings, retrieves relevant passages for given queries, and ranks them using models like BM25, Logistic Regression, LambdaMART, and a Neural Network.

---

## Features

1. **Preprocessing**: 
   - Text tokenization, normalization, stopword removal, and lemmatization.
   - Embedding generation using pre-trained models like FastText.

2. **Retrieval Models**:
   - BM25, Laplace Smoothing, and Dirichlet Smoothing for passage ranking.

3. **Ranking Models**:
   - Logistic Regression for linear ranking.
   - LambdaMART (XGBoost Ranker) for tree-based ranking.
   - Neural Network (PyTorch) for deep learning-based ranking.

4. **Evaluation**:
   - Metrics include Mean Average Precision (MAP) and Normalized Discounted Cumulative Gain (NDCG).

5. **Feature Engineering**:
   - Cosine similarity and Word Mover’s Distance (WMD) between embeddings.
   - Element-wise product features for ranking models.

---

## Project Structure

```plaintext
project/
│
├── README.md                  # Project overview and usage instructions
├── requirements.txt           # Dependencies
├── main.py                    # Entry-point for executing the pipeline
│
├── preprocessing/             # Preprocessing-related modules
│   ├── text_processing.py     # Text cleaning, tokenization, lemmatization
│   ├── feature_engineering.py # Embedding and feature computation
│
├── retrieval/                 # Retrieval algorithms
│   ├── bm25.py                # BM25 ranking implementation
│   ├── smoothing.py           # Laplace and Dirichlet smoothing
│
├── models/                    # Ranking models
│   ├── logistic_regression.py # Custom logistic regression model
│   ├── lambdamart.py          # LambdaMART (XGBoost Ranker) model
│   ├── passage_ranking_nn.py  # PyTorch-based ranking neural network
│
├── evaluation/                # Evaluation metrics
│   ├── metrics.py             # MAP and NDCG calculations
│
├── utils/                     # Helper utilities
│   ├── io_operations.py       # File handling and data loading
│
├── tests/                     # Unit tests
    ├── test_metrics.py        # Tests for evaluation metrics
    ├── test_text_processing.py# Tests for preprocessing functions
    ├── test_models.py         # Tests for models
