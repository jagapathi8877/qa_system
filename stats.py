"""
stats.py — per-session dashboard metrics + lightweight query classification.

SessionStats is deliberately dumb-and-pure: it only records and aggregates.
app.py owns the session -> stats mapping.
"""

import re
from collections import OrderedDict


def classify_query(question: str) -> str:
    """Keyword heuristics, as per the spec.

    "what is/are X"      -> Definition
    "how to/how do X"    -> Procedural
    "compare / vs / difference between" -> Comparison
    everything else      -> Factual
    """
    q = " ".join((question or "").lower().split())
    if not q:
        return "Factual"
    if re.search(r"\b(compare|comparison|versus|vs\.?|difference(s)?\s+between)\b", q):
        return "Comparison"
    if re.search(r"\b(what\s+(is|are|was|were)|define|definition\s+of|meaning\s+of)\b", q):
        return "Definition"
    if re.search(r"\b(how\s+(to|do|does|did|can|should)|steps\s+to|way\s+to)\b", q):
        return "Procedural"
    return "Factual"


class SessionStats:
    """Aggregates counts for one browser session (its cookie id)."""

    def __init__(self):
        self.total_queries = 0
        self.answered_queries = 0
        self.no_answer_queries = 0
        self.total_response_ms = 0.0
        self.categories = OrderedDict(
            [("Definition", 0), ("Factual", 0), ("Procedural", 0), ("Comparison", 0)]
        )

    def record(self, category: str, answered: bool, response_ms: int):
        self.total_queries += 1
        self.total_response_ms += response_ms
        if answered:
            self.answered_queries += 1
        else:
            self.no_answer_queries += 1
        if category in self.categories:
            self.categories[category] += 1
        else:
            self.categories[category] = 1

    def average_response_ms(self) -> float:
        if self.total_queries == 0:
            return 0.0
        return self.total_response_ms / self.total_queries

    def snapshot(self) -> dict:
        return {
            "total_queries": self.total_queries,
            "answered": self.answered_queries,
            "no_answer": self.no_answer_queries,
            "avg_response_ms": round(self.average_response_ms(), 1),
            "categories": dict(self.categories),
        }
