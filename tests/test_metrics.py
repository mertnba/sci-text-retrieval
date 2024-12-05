import numpy as np
from metrics import calculate_average_precision, calculate_ndcg, calculate_metrics

def test_calculate_average_precision():
    """
    Test the calculation of average precision.
    """
    scores = np.array([0.9, 0.8, 0.4, 0.2])
    true_labels = np.array([1, 0, 1, 1])
    expected_ap = 0.7222  # Manually calculated expected result
    calculated_ap = calculate_average_precision(scores, true_labels)
    assert np.isclose(calculated_ap, expected_ap, atol=1e-4), f"Expected {expected_ap}, got {calculated_ap}"

def test_calculate_ndcg():
    """
    Test the calculation of NDCG.
    """
    scores = np.array([0.9, 0.8, 0.4, 0.2])
    true_labels = np.array([3, 2, 3, 0])
    expected_ndcg = 0.912  # Manually calculated expected result
    calculated_ndcg = calculate_ndcg(scores, true_labels)
    assert np.isclose(calculated_ndcg, expected_ndcg, atol=1e-3), f"Expected {expected_ndcg}, got {calculated_ndcg}"

def test_calculate_metrics():
    """
    Test the calculation of MAP and NDCG across multiple queries.
    """
    scores = np.array([0.9, 0.8, 0.4, 0.2, 0.7, 0.6, 0.3])
    true_labels = np.array([1, 0, 1, 1, 0, 1, 0])
    qids = np.array([1, 1, 1, 1, 2, 2, 2])
    expected_map = 0.7222  # Average precision for query 1
    expected_ndcg = 0.912  # NDCG for query 1
    calculated_map, calculated_ndcg = calculate_metrics(scores, true_labels, qids)

    assert np.isclose(calculated_map, expected_map, atol=1e-4), f"Expected MAP {expected_map}, got {calculated_map}"
    assert np.isclose(calculated_ndcg, expected_ndcg, atol=1e-3), f"Expected NDCG {expected_ndcg}, got {calculated_ndcg}"

if __name__ == "__main__":
    test_calculate_average_precision()
    test_calculate_ndcg()
    test_calculate_metrics()
    print("All tests passed!")
