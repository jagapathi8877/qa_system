"""
simplifier.py — rule-based text simplification.

Two cheap, deterministic transforms used by the explanation feature:
  1. LEXICON: common academic/formal words replaced with everyday equivalents
     ("utilize" → "use", "numerous" → "many", …).
  2. Sentence splitting: long sentences are cut at discourse connectors
     ("; ", ", and ", ", but ", ", which ", " while ") so no sentence runs
     longer than ~MAX_WORDS words.

Pure-Python and offline — no model needed, so simplification stays instant
even on a CPU-only machine.
"""

import re

LEXICON = {
    "utilize": "use", "utilizes": "uses", "utilized": "used", "utilizing": "using",
    "demonstrate": "show", "demonstrates": "shows", "demonstrated": "showed",
    "demonstrating": "showing",
    "approximately": "about", "numerous": "many", "obtain": "get",
    "obtains": "gets", "obtained": "got", "obtaining": "getting",
    "additional": "more", "additionally": "also", "furthermore": "also",
    "however": "but", "consequently": "so", "therefore": "so",
    "nevertheless": "even so", "comprise": "consist of",
    "comprises": "consists of", "facilitate": "help", "facilitates": "helps",
    "facilitated": "helped", "assist": "help", "assists": "helps",
    "assisted": "helped", "determine": "find out", "determines": "finds out",
    "determined": "found out", "require": "need", "requires": "needs",
    "required": "needed", "sufficient": "enough", "insufficient": "not enough",
    "perform": "do", "performs": "does", "performed": "did",
    "attempt": "try", "attempts": "tries", "attempted": "tried",
    "initiate": "start", "initiates": "starts", "initiated": "started",
    "implement": "set up", "implements": "sets up", "implemented": "set up",
    "regarding": "about",
    "locate": "find", "locates": "finds", "located": "found",
    "allow": "let", "allows": "lets", "permit": "let", "permits": "lets",
    "modify": "change", "modifies": "changes", "modified": "changed",
    "substantial": "large", "automatically": "on its own",
    "generate": "make", "generates": "makes", "generated": "made",
    "inefficient": "slow", "difficulties": "problems", "difficulty": "problem",
    "scattered": "spread out", "numerically": "in numbers",
    "retrieval": "finding", "retrieve": "fetch", "retrieves": "fetches",
    "retrieved": "fetched",
}

_LEXICON_RE = re.compile(
    r"\b(" + "|".join(re.escape(w) for w in sorted(LEXICON, key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)

# Places a long sentence can be safely cut into two.
_CUTS = [
    re.compile(r"(?<=[a-z];)\s+(?=[A-Za-z])"),          # "; next clause"
    re.compile(r",\s+(and|but|so|yet)\s+(?=\w)"),       # ", and …"
    re.compile(r",\s+(which|who|while|although|whereas)\s+"),
    re.compile(r"\bhowever,\s+"),
]

MAX_WORDS = 26
MIN_HALF_WORDS = 5


def _sub(match: re.Match) -> str:
    word = match.group(0)
    simple = LEXICON[word.lower()]
    if word[0].isupper():
        simple = simple[0].upper() + simple[1:]
    return simple


def _split_sentence(sentence: str) -> list:
    """Cut one sentence at connectors until every piece is short."""
    pieces = [sentence]
    for pattern in _CUTS:
        nxt = []
        for piece in pieces:
            if len(piece.split()) <= MAX_WORDS:
                nxt.append(piece)
                continue
            parts = pattern.split(piece, maxsplit=1)
            if len(parts) >= 2:
                head = parts[0].rstrip(",;")
                tail = parts[-1].strip()
                connector = parts[1] if len(parts) == 3 else ""
                tail = (connector + " " + tail) if connector else tail
                if len(head.split()) >= MIN_HALF_WORDS and len(tail.split()) >= MIN_HALF_WORDS:
                    head = head if head.endswith((".", "!", "?")) else head + "."
                    nxt.extend([head, tail[0].upper() + tail[1:] if tail else tail])
                    continue
            nxt.append(piece)
        pieces = nxt
    out = []
    for piece in pieces:
        piece = re.sub(r"\s+", " ", piece).strip()
        if piece:
            out.append(piece if piece.endswith((".", "!", "?")) else piece + ".")
    return out


def simplify(text: str) -> str:
    """Return a simplified version of the text (own-words feel, short sentences)."""
    if not text:
        return ""
    text = _LEXICON_RE.sub(_sub, text)
    result = []
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        if len(sentence.split()) > MAX_WORDS:
            result.extend(_split_sentence(sentence))
        else:
            result.append(sentence)
    return " ".join(result)


def simplify_points(sentences: list) -> list:
    """Simplify a list of key-point sentences, one entry per simple sentence."""
    points = []
    for sent in sentences:
        for piece in re.split(r"(?<=[.!?])\s+", simplify(sent)):
            piece = piece.strip()
            if len(piece.split()) >= 4:
                points.append(piece)
    return points[:6]
