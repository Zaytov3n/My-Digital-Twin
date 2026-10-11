from utils.rag import retrieve
from utils.tools import (
    build_twin_context,
    calculate_conditional_probabilities,
    calculate_probabilities,
    predict_next_activity,
)


def test_probabilities_sum_to_one():
    probs = calculate_probabilities(["Work", "Work", "Sleep", "Leisure"])
    assert probs["Work"] == 0.5
    assert abs(sum(probs.values()) - 1.0) < 0.01


def test_empty_sequence():
    assert calculate_probabilities([]) == {}
    assert calculate_conditional_probabilities([]) == {}


def test_conditional_probabilities():
    cond = calculate_conditional_probabilities(["Work", "Sleep", "Work", "Sleep"])
    assert cond["Work"] == {"Sleep": 1.0}
    assert cond["Sleep"] == {"Work": 1.0}


def test_predict_next_activity():
    cond = {"Work": {"Sleep": 0.7, "Leisure": 0.3}}
    assert predict_next_activity("Work", cond) == "Sleep"
    assert predict_next_activity("Unknown", cond) is None


def test_build_twin_context():
    ctx = build_twin_context(["Work", "Sleep", "Work"])
    assert ctx["number_of_activity_records"] == 3
    assert ctx["current_recorded_activity"] == "Work"
    assert ctx["most_likely_next_activity"] == "Sleep"


def test_retrieve_returns_relevant_entry():
    results = retrieve("when should I exercise?")
    assert any("Exercise" in r["title"] for r in results)