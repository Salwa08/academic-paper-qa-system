from evaluation.retrieval_metrics import hit_rate_at_k, recall_at_k, mrr_at_k, ndcg_at_k


def test_retrieval_metrics_match_hand_calculated_values():
    ranked = ["a", "b", "c", "d"]
    relevant = ["b", "d"]

    assert hit_rate_at_k(ranked, relevant, 1) == 0.0
    assert hit_rate_at_k(ranked, relevant, 2) == 1.0
    assert recall_at_k(ranked, relevant, 2) == 0.5
    assert mrr_at_k(ranked, relevant, 3) == 0.5
    assert round(ndcg_at_k(ranked, relevant, 4), 4) == 0.6509

