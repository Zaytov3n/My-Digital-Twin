"""Activity-pattern tools: probability calculations and context building."""
from collections import Counter, defaultdict
from datetime import datetime
from typing import Dict, List, Optional

DEFAULT_ACTIVITIES = [
    "Sleep", "Work", "Work", "Leisure",
    "Sleep", "Social", "Exercise", "Sleep",
    "Work", "Leisure", "Leisure", "Sleep",
    "Exercise", "Work", "Work", "Sleep",
]
ACTIVITY_OPTIONS = ["Sleep", "Work", "Exercise", "Social", "Leisure"]


def calculate_probabilities(sequence: List[str]) -> Dict[str, float]:
    if not sequence:
        return {}
    counts = Counter(sequence)
    total = len(sequence)
    return {a: round(c / total, 3) for a, c in counts.items()}


def calculate_conditional_probabilities(
    sequence: List[str],
) -> Dict[str, Dict[str, float]]:
    transitions = defaultdict(Counter)
    for current, nxt in zip(sequence[:-1], sequence[1:]):
        transitions[current][nxt] += 1
    result = {}
    for current, counts in transitions.items():
        total = sum(counts.values())
        result[current] = {a: round(c / total, 3) for a, c in counts.items()}
    return result


def predict_next_activity(
    current: str, cond: Dict[str, Dict[str, float]]
) -> Optional[str]:
    probs = cond.get(current, {})
    return max(probs, key=probs.get) if probs else None


def build_twin_context(activities: List[str]) -> Dict:
    cond = calculate_conditional_probabilities(activities)
    current = activities[-1] if activities else None
    return {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "current_recorded_activity": current,
        "activity_frequencies": calculate_probabilities(activities),
        "next_activity_probabilities": cond,
        "most_likely_next_activity": (
            predict_next_activity(current, cond) if current else None
        ),
        "number_of_activity_records": len(activities),
    }