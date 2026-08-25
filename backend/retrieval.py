"""TF-IDF + cosine-similarity retrieval over the fixed knowledge base.

See spec Design Notes -> "Storage/retrieval trade-off" for why TF-IDF was
chosen over embeddings or LLM-only routing. The retriever is policy-free: it
always returns every candidate's raw score and never filters or decides
refusal -- that decision belongs solely to `qa.answer_question` (Design
Notes -> "Refusal mechanics").
"""

import re
from dataclasses import dataclass

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from backend.kb_data import KB_ENTRIES

_TOKEN_RE = re.compile(r"[A-Za-z]+|\d+")

# Standard English stopwords, plus club-specific filler words that show up
# in nearly every question about the club (its own name, "campus") or that
# occur only incidentally in one section's content without being about that
# section's topic (e.g. "club"/"clubs" appears solely in the Achievements
# entry's "partnerships with 3 college clubs" -- an unrelated use that would
# otherwise make Achievements a false-positive match for any generic
# "the club" question). Excluding these keeps discriminative power on words
# that actually distinguish one section from another.
_STOP_WORDS = ENGLISH_STOP_WORDS | {"gdg", "campus", "club", "clubs"}


def _stem(token: str) -> str:
    """Naive suffix-stripping stemmer (no extra NLP dependency needed for a
    7-entry KB). Normalizes simple plural/verb forms so "teams" matches
    "Team", "leads" matches "Lead", etc., without pulling in nltk/spacy."""
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 4 and token.endswith("es"):
        return token[:-2]
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def _analyzer(text: str) -> list[str]:
    """Tokenize + lowercase + stem + drop stopwords. Used identically for
    fitting the corpus and for transforming queries, so both sides
    normalize the same way."""
    return [
        _stem(tok)
        for tok in (m.lower() for m in _TOKEN_RE.findall(text))
        if tok not in _STOP_WORDS
    ]


_LABEL_REPEATS = 3


def _indexed_text(entry: dict) -> str:
    """Text representation used ONLY for building the retrieval index.

    Includes the section label (repeated a few times so it carries enough
    TF-IDF weight against the entry's own longer content -- otherwise a
    single label mention gets diluted by unrelated but longer sections)
    alongside the content, so topical queries (e.g. "list all the teams")
    match the right section even when the section's raw content text
    doesn't itself contain that word (e.g. the Teams entry lists team
    names/leads but never the literal word "team", while Rules incidentally
    mentions "team switching"). This does not change what's stored in
    `KB_ENTRIES` or what's returned as `RetrievalResult.content` -- both
    stay verbatim.
    """
    return f"{(entry['section'] + ' ') * _LABEL_REPEATS}{entry['content']}"


@dataclass(frozen=True)
class RetrievalResult:
    """One scored KB candidate for a query."""

    section: str
    content: str
    score: float


class TfidfRetriever:
    """Fits a TF-IDF vectorizer over `KB_ENTRIES` and retrieves by cosine similarity."""

    def __init__(self, entries: list[dict] | None = None) -> None:
        self._entries = entries if entries is not None else KB_ENTRIES
        self._vectorizer = TfidfVectorizer(analyzer=_analyzer)
        corpus = [_indexed_text(entry) for entry in self._entries]
        self._doc_matrix = self._vectorizer.fit_transform(corpus)

    def retrieve(self, query: str, top_k: int = 3) -> list[RetrievalResult]:
        """Return up to `top_k` KB entries for `query`, sorted by score desc.

        Every candidate's raw, unrounded similarity score is preserved --
        callers decide what to do with it (spec: "Retrieval returns section
        + raw similarity score for every candidate; score is never discarded
        before reaching the caller.").
        """
        if not query or not query.strip():
            # Guard empty/whitespace-only input before vectorizing -- an
            # empty query has zero learned vocabulary terms, and scoring it
            # against the corpus would otherwise raise inside scikit-learn.
            return [
                RetrievalResult(section=entry["section"], content=entry["content"], score=0.0)
                for entry in self._entries[:top_k]
            ]

        query_vector = self._vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, self._doc_matrix)[0]

        results = [
            RetrievalResult(
                section=entry["section"],
                content=entry["content"],
                score=float(similarities[i]),
            )
            for i, entry in enumerate(self._entries)
        ]
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]
