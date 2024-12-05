import numpy as np

def calculate_average_precision(scores, true_labels):
    """
    Calculates the average precision (AP) for a single query.

    Args:
        scores (ndarray): Predicted scores.
        true_labels (ndarray): True labels.

    Returns:
        float: Average precision.
    """
    sorted_indices = np.argsort(-scores)
    sorted_true_labels = true_labels[sorted_indices]

    precisions = np.cumsum(sorted_true_labels) / (np.arange(len(sorted_true_labels)) + 1)
    average_precision = (precisions * sorted_true_labels).sum() / sorted_true_labels.sum() if sorted_true_labels.sum() > 0 else 0

    return average_precision

def calculate_ndcg(scores, true_labels):
    """
    Calculates the normalized discounted cumulative gain (NDCG) for a single query.

    Args:
        scores (ndarray): Predicted scores.
        true_labels (ndarray): True labels.

    Returns:
        float: NDCG.
    """
    sorted_indices = np.argsort(-scores)
    sorted_true_labels = true_labels[sorted_indices]

    gains = 2 ** sorted_true_labels - 1
    discounts = np.log2(np.arange(len(sorted_true_labels)) + 2)
    dcg = np.sum(gains / discounts)

    ideal_sorted_labels = np.sort(true_labels)[::-1]
    ideal_gains = 2 ** ideal_sorted_labels - 1
    idcg = np.sum(ideal_gains / discounts)

    ndcg = dcg / idcg if idcg > 0 else 0

    return ndcg

def calculate_metrics(scores, true_labels, qids):
    """
    Calculates MAP and NDCG metrics.

    Args:
        scores (ndarray): Predicted scores.
        true_labels (ndarray): True labels.
        qids (ndarray): Query IDs.

    Returns:
        tuple: Mean average precision (MAP) and normalized discounted cumulative gain (NDCG).
    """
    unique_qids = np.unique(qids)
    mean_ap_list = []
    mean_ndcg_list = []

    for qid in unique_qids:
        idx = (qids == qid)
        ap = calculate_average_precision(scores[idx], true_labels[idx])
        ndcg = calculate_ndcg(scores[idx], true_labels[idx])
        mean_ap_list.append(ap)
        mean_ndcg_list.append(ndcg)

    mean_ap = np.mean(mean_ap_list)
    mean_ndcg = np.mean(mean_ndcg_list)

    return mean_ap, mean_ndcg
