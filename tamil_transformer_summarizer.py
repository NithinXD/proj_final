"""
Transformer-based abstractive summarizer for Tamil and mixed Tamil-English text.
Default model: csebuetnlp/mT5_multilingual_XLSum (supports Tamil summarization).
"""
from __future__ import annotations

from dataclasses import dataclass
import re
from typing import List


@dataclass
class TransformerSummaryConfig:
    model_name: str = "csebuetnlp/mT5_multilingual_XLSum"
    max_input_chars: int = 6000
    chunk_chars: int = 1600
    chunk_overlap_chars: int = 200
    max_new_tokens: int = 180
    min_new_tokens: int = 60
    num_beams: int = 4


class TamilTransformerSummarizer:
    """High-quality abstractive summarizer powered by a multilingual mT5 model."""

    SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?।])\s+|\n+")

    def _clean_input_text(self, text: str) -> str:
        """Remove common header/footer and symbol noise before generation."""
        lines = [re.sub(r"\s+", " ", line).strip() for line in text.splitlines()]
        cleaned_lines: List[str] = []

        for line in lines:
            if not line:
                continue
            if re.fullmatch(r"[+*\-_=~.\s]+", line):
                continue
            if re.fullmatch(r"\d+(\s*[+*\-]\s*[+*\-])?", line):
                continue
            if re.fullmatch(r"\d+\s+.+", line) and len(line) < 80 and not re.search(r"[.!?]", line):
                continue
            if re.fullmatch(r".+\s+\d+", line) and len(line) < 80 and not re.search(r"[.!?]", line):
                continue
            cleaned_lines.append(line)

        return "\n".join(cleaned_lines)

    def __init__(self, config: TransformerSummaryConfig | None = None):
        self.config = config or TransformerSummaryConfig()
        self._pipeline = None

    def _load_pipeline(self):
        if self._pipeline is not None:
            return

        from transformers import pipeline  # Lazy import so app can start even if model is not yet downloaded.

        self._pipeline = pipeline(
            "summarization",
            model=self.config.model_name,
            tokenizer=self.config.model_name,
        )

    def _split_into_windows(self, text: str) -> List[str]:
        if len(text) <= self.config.chunk_chars:
            return [text]

        sentences = [s.strip() for s in self.SENTENCE_SPLIT_PATTERN.split(text) if s.strip()]
        if not sentences:
            return [text[i : i + self.config.chunk_chars] for i in range(0, len(text), self.config.chunk_chars)]

        windows = []
        current = []
        current_len = 0

        for sentence in sentences:
            s_len = len(sentence)
            if current and current_len + s_len > self.config.chunk_chars:
                windows.append(" ".join(current))

                # Keep overlap by carrying tail text into the next window.
                carry = windows[-1][-self.config.chunk_overlap_chars :]
                current = [carry, sentence]
                current_len = len(carry) + s_len
            else:
                current.append(sentence)
                current_len += s_len

        if current:
            windows.append(" ".join(current))

        return windows

    def _generate_summary(self, text: str) -> str:
        self._load_pipeline()
        prompt = f"summarize: {text.strip()}"
        outputs = self._pipeline(
            prompt,
            max_new_tokens=self.config.max_new_tokens,
            min_new_tokens=self.config.min_new_tokens,
            num_beams=self.config.num_beams,
            do_sample=False,
            truncation=True,
        )
        return outputs[0]["generated_text"].strip()

    def summarize(self, context_chunks: List[str]) -> str:
        raw_text = "\n".join(chunk for chunk in context_chunks if chunk and chunk.strip())
        raw_text = self._clean_input_text(raw_text)
        if not raw_text.strip():
            return "Insufficient context to generate summary."

        raw_text = raw_text[: self.config.max_input_chars]
        windows = self._split_into_windows(raw_text)

        # Map-reduce summarization for long contexts.
        partials = [self._generate_summary(window) for window in windows]
        if len(partials) == 1:
            return partials[0]

        reduced_text = " ".join(partials)
        final_summary = self._generate_summary(reduced_text)
        return final_summary
