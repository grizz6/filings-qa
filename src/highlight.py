"""Find the sentence of a cited passage that best supports the answer sentence citing it.

The model cites whole passages ([n]), not sentences. To point a reader at the right place,
each answer sentence that cites [n] is compared with every sentence of passage n by the
words they share; the passage sentence with the most overlap is the likely support. It is a
reading aid, not a check: the cited passage as a whole is what the answer rests on.
"""

from __future__ import annotations

import re

STOPWORDS = frozenset(
    """a about above after again against all also an and any are as at be because been before
    being below between both but by can could did do does doing during each few for from
    further had has have having he her here hers him his how i if in into is it its itself
    may might more most must no nor not of off on once only or other our out over own same
    she should so some such than that the their them then there these they this those
    through to too under until up very was we were what when where which while who whom why
    will with would you your company company's companys""".split()
)
SENTENCE_END = re.compile(r"(?<=[.!?;])\s+")
CITATION = re.compile(r"\[(\d+)\]")


def words(text: str) -> set[str]:
    return {
        w for w in re.findall(r"[a-z][a-z'-]+", text.lower()) if len(w) > 2 and w not in STOPWORDS
    }


def sentences(text: str) -> list[str]:
    return [s.strip() for s in SENTENCE_END.split(" ".join(text.split())) if s.strip()]


def citing_text(answer: str, n: int) -> str:
    """The answer sentences that cite [n], citations removed."""
    cited = [s for s in sentences(answer) if str(n) in CITATION.findall(s)]
    return CITATION.sub("", " ".join(cited))


def supporting_sentence(answer: str, n: int, passage: str) -> str | None:
    """The passage sentence sharing the most words with what the answer says citing [n]."""
    claim = words(citing_text(answer, n))
    if not claim:
        return None
    best, best_overlap = None, 0
    for sentence in sentences(passage):
        overlap = len(claim & words(sentence))
        if overlap > best_overlap:
            best, best_overlap = sentence, overlap
    return best if best_overlap >= 2 else None
