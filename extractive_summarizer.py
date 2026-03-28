"""
Algorithmic extractive summarizer for Tamil/English mixed text.
Uses sentence scoring with TF-IDF style weighting.
"""
from __future__ import annotations

from collections import Counter
import math
import re
from typing import List


class ExtractiveTamilSummarizer:
    """Language-agnostic extractive summarizer with Tamil-aware tokenization."""

    # Common Tamil and English stopwords to reduce noisy scoring.
    STOPWORDS = {
        "and", "or", "the", "is", "are", "was", "were", "to", "of", "in", "on", "for", "with", "a", "an",
        "இந்த", "அந்த", "மற்றும்", "என்று", "என", "ஒரு", "இது", "அது", "உள்ள", "பற்றி", "மூலம்", "ஆகிய", "ஆகும்",
        "என்பது", "என்பன", "முதல்", "பின்", "மேலும்", "கூட", "இல்லை", "உம்", "ல்", "ன்",
    }

    SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?।])\s+|\n+")
    TOKEN_PATTERN = re.compile(r"[\u0B80-\u0BFFA-Za-z0-9]+")

    def _split_sentences(self, text: str) -> List[str]:
        sentences = [s.strip() for s in self.SENTENCE_SPLIT_PATTERN.split(text) if s.strip()]
        return [s for s in sentences if len(s) > 20]

    def _tokenize(self, text: str) -> List[str]:
        tokens = [t.lower() for t in self.TOKEN_PATTERN.findall(text)]
        return [t for t in tokens if len(t) > 1 and t not in self.STOPWORDS]

    def _compute_idf(self, sentence_tokens: List[List[str]]) -> dict:
        df = Counter()
        for toks in sentence_tokens:
            for tok in set(toks):
                df[tok] += 1

        n_docs = max(len(sentence_tokens), 1)
        return {tok: math.log((1 + n_docs) / (1 + freq)) + 1.0 for tok, freq in df.items()}

    def _sentence_score(self, tokens: List[str], idf: dict, global_tf: Counter) -> float:
        if not tokens:
            return 0.0

        score = 0.0
        for tok in tokens:
            tf = 1.0 + math.log(1 + global_tf[tok])
            score += tf * idf.get(tok, 0.0)

        # Normalize to avoid very long sentences always winning.
        return score / (len(tokens) ** 0.6)

    def summarize(self, context_chunks: List[str], max_sentences: int = 6) -> str:
        """
        Build an extractive summary from retrieved chunks.

        Args:
            context_chunks: Retrieved document chunks
            max_sentences: Number of sentences in final summary

        Returns:
            Tamil/English mixed summary constructed from source sentences
        """
        raw_text = "\n".join(chunk for chunk in context_chunks if chunk and chunk.strip())
        if not raw_text.strip():
            return "சுருக்கம் உருவாக்க போதுமான தகவல் இல்லை."

        sentences = self._split_sentences(raw_text)
        if not sentences:
            compact = raw_text.strip()[:500]
            return compact if compact else "சுருக்கம் உருவாக்க போதுமான தகவல் இல்லை."

        tokenized = [self._tokenize(sentence) for sentence in sentences]
        idf = self._compute_idf(tokenized)

        global_tf = Counter(tok for toks in tokenized for tok in toks)
        scored = []
        for idx, toks in enumerate(tokenized):
            score = self._sentence_score(toks, idf, global_tf)
            scored.append((idx, score))

        # Pick top candidates by score, then preserve original order for readability.
        scored.sort(key=lambda x: x[1], reverse=True)
        selected_idx = sorted(idx for idx, _ in scored[:max_sentences])

        summary_sentences = [sentences[idx] for idx in selected_idx]
        summary_text = " ".join(summary_sentences).strip()

        # Keep output concise for UI readability.
        if len(summary_text) > 1200:
            summary_text = summary_text[:1200].rsplit(" ", 1)[0] + "..."

        return summary_text
