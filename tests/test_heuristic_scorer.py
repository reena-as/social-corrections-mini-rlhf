from social_corrections.evaluation.heuristic_scorer import aggregate, score


def test_scores_known_polite_example():
    s = score("I understand why this is confusing. Let's work through it together step by step.")
    assert s.acknowledgment == 1
    assert s.constructive == 1
    assert s.contains_harsh_word == 0


def test_flags_harsh_word():
    s = score("That's just wrong.")
    assert s.contains_harsh_word == 1


def test_hedging_detected():
    s = score("It might be the case that this works.")
    assert s.hedging == 1


def test_aggregate_handles_empty():
    agg = aggregate([])
    assert agg["n"] == 0
    for k in ("acknowledgment", "constructive", "hedging", "composite"):
        assert agg[k] == 0.0


def test_aggregate_handles_populated():
    texts = [
        "I see what you mean. Let's try a different approach.",
        "No.",
        "This is definitely wrong.",
    ]
    agg = aggregate(texts)
    assert agg["n"] == 3
    assert 0.0 <= agg["acknowledgment"] <= 1.0
    assert 0.0 <= agg["contains_harsh_word"] <= 1.0
