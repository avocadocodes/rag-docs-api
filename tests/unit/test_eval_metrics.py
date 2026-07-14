import pytest
from eval.metrics import faithfulness_score, abstention_accuracy, recall_at_k, mrr


def test_faithfulness_all_supported():
    assert faithfulness_score(["SUPPORTED", "SUPPORTED", "SUPPORTED"]) == 1.0


def test_faithfulness_none_supported():
    assert faithfulness_score(["UNSUPPORTED", "UNSUPPORTED"]) == 0.0


def test_faithfulness_partial():
    result = faithfulness_score(["SUPPORTED", "UNSUPPORTED", "SUPPORTED"])
    assert abs(result - 2/3) < 1e-9


def test_faithfulness_empty_list():
    assert faithfulness_score([]) == 0.0


def test_faithfulness_neutral_not_counted_as_supported():
    result = faithfulness_score(["SUPPORTED", "NEUTRAL"])
    assert result == 0.5


def test_abstention_accuracy_all_correct():
    results = [(True, True), (True, True), (True, True)]
    assert abstention_accuracy(results) == 1.0


def test_abstention_accuracy_none_correct():
    results = [(False, True), (False, True)]
    assert abstention_accuracy(results) == 0.0


def test_abstention_accuracy_mixed():
    results = [
        (True, True),
        (False, True),
        (False, False),
        (True, False),
    ]
    assert abstention_accuracy(results) == 0.5


def test_abstention_accuracy_empty():
    assert abstention_accuracy([]) == 0.0


def test_recall_at_k_still_works():
    results = [([1, 2, 3], [1]), ([4, 5], [1])]
    assert recall_at_k(results, k=1) == 0.5


def test_mrr_still_works():
    results = [([1, 2, 3], [3])]
    assert abs(mrr(results) - 1/3) < 1e-9
