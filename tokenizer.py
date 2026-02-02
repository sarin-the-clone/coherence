"""
Unified tokenizer for coherence and long-range dependence analysis.

Includes:
  • Integrated text normalization and pre-tokenization
  • Multiple tokenization backends (GPT-4 BPE, WordPiece, SentencePiece, spaCy, UTF-8 bytes)

----------------------------------------------------------------------
📘  PREPROCESSING OVERVIEW
----------------------------------------------------------------------

Preprocessing defines how raw text is cleaned and segmented *before* it is
converted into subword tokens.  It is essential for reproducibility, because
even small changes (case, spacing, Unicode form) can alter the token sequence
and therefore coherence metrics such as β or H.

Pipeline stages:

    raw text
        ↓
    normalization   → Unicode cleanup, optional case/digit handling
        ↓
    pre-tokenization → optional splitting (none/simple/sentence)
        ↓
    tokenizer.encode()

----------------------------------------------------------------------
🔍  NORMALIZATION OPTIONS
----------------------------------------------------------------------

Unicode normalization  (always applied)
    • Uses NFKC form to collapse visually equivalent characters
      (e.g. “ﬁ” → “fi”, full-width → half-width)

Whitespace normalization  (always applied)
    • Collapses multiple spaces/newlines into one.

add_space_punct  (default: True)
    • Adds spaces around punctuation marks [.,!?;:]
    • Ensures punctuation is separated from words for tokenizers that
      depend on whitespace (e.g. spaCy, WordPiece).
    • For byte-level BPE, you can disable it — punctuation is handled internally.

lowercase  (default: False)
    • Converts text to lowercase.
    • Useful when comparing stylistically different corpora (e.g. novels vs chat logs).
    • Disable if you want to preserve case information (e.g. for proper nouns).

normalize_digits  (default: False)
    • Replaces all digits with "0".
    • Helps test whether numerical content contributes to apparent coherence.

----------------------------------------------------------------------
🔍  PRE-TOKENIZATION MODES
----------------------------------------------------------------------

Pre-tokenization controls how text is chunked before subword encoding:

    'none'     → No splitting (recommended for byte-level BPE, e.g. GPT-4)
    'simple'   → Splits into words and punctuation using regex.
                 Good for WordPiece or spaCy when you want a quick whitespace split.
    'sentence' → Splits on sentence boundaries (.,!,?) followed by space.
                 Useful when you want to process sentences independently.

Pre-tokenization is *not* needed for byte-level BPE (GPT-4) because it handles
punctuation and spacing internally.  For other tokenizers, it ensures consistent
segmentation.

----------------------------------------------------------------------
When to use which
----------------------------------------------------------------------

Use normalization and pre-tokenization when:
  • Comparing human and model-generated corpora from heterogeneous sources.
  • Working with tokenizers sensitive to whitespace or punctuation (WordPiece, spaCy).
  • Evaluating metrics where orthographic differences can masquerade as coherence.

Skip or minimize preprocessing when:
  • Using GPT-4 / byte-level BPE and you want the tokenizer to see the raw text structure.
  • Investigating stylistic or orthographic effects (e.g. capitalization patterns).

----------------------------------------------------------------------
"""

from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import List, Dict, Any
import importlib
import hashlib
import json
import re
import unicodedata


@dataclass
class TokenizerMetadata:
    name: str
    version: str
    vocab_size: int | None
    extra: Dict[str, Any]


class Tokenizer:
    """Unified tokenizer + text preprocessing interface."""

    def __init__(
        self,
        kind: str = "gpt4",
        model: str | None = None,
        lowercase: bool = False,
        normalize_digits: bool = False,
        add_space_punct: bool = True,
        pretokenize: str = "none",
    ):
        """
        Parameters
        ----------
        kind : str
            Tokenizer family ("gpt4", "wordpiece", "sentencepiece", "spacy", "bytes").
        model : str | None
            Optional model name/path for HuggingFace or SentencePiece.
        lowercase : bool
            Convert text to lowercase before tokenization.
        normalize_digits : bool
            Replace digits with '0'.
        add_space_punct : bool
            Add spaces around punctuation for consistent token boundaries.
        pretokenize : {'none','simple','sentence'}
            Pre-tokenization mode (see module docstring).
        """
        self.kind = kind.lower()
        self.model = model
        self.lowercase = lowercase
        self.normalize_digits = normalize_digits
        self.add_space_punct = add_space_punct
        self.pretokenize = pretokenize
        self._load()

    # ------------------------------------------------------------------
    # 1. Text preprocessing
    # ------------------------------------------------------------------
    def _normalize(self, text: str) -> str:
        """Apply Unicode, whitespace, case, digit, and punctuation normalization."""
        t = unicodedata.normalize("NFKC", text)
        t = re.sub(r"\s+", " ", t)
        if self.add_space_punct:
            t = re.sub(r"([.,!?;:])", r" \1 ", t)
        if self.lowercase:
            t = t.lower()
        if self.normalize_digits:
            t = re.sub(r"\d", "0", t)
        return t.strip()

    def _pretokenize(self, text: str) -> list[str]:
        """Split text according to selected pre-tokenization mode."""
        if self.pretokenize == "none":
            return [text]
        elif self.pretokenize == "simple":
            return re.findall(r"\w+|[^\w\s]", text, re.UNICODE)
        elif self.pretokenize == "sentence":
            return re.split(r"(?<=[.!?]) +", text)
        else:
            raise ValueError(f"Unknown pretokenize mode: {self.pretokenize}")

    # ------------------------------------------------------------------
    # 2. Tokenizer initialization
    # ------------------------------------------------------------------
    def _load(self):
        if self.kind in {"gpt4", "gpt-4"}:
            import tiktoken
            enc = tiktoken.encoding_for_model("gpt-4")
            self.tokenizer = enc
            self.meta = TokenizerMetadata(
                name="gpt4_byte_bpe",
                version=tiktoken.__version__,
                vocab_size=enc.n_vocab,
                extra={"encoding": enc.name},
            )

        elif self.kind == "wordpiece":
            from transformers import AutoTokenizer
            model_name = self.model or "bert-base-uncased"
            tok = AutoTokenizer.from_pretrained(model_name)
            self.tokenizer = tok
            self.meta = TokenizerMetadata(
                name="wordpiece",
                version=importlib.metadata.version("transformers"),
                vocab_size=tok.vocab_size,
                extra={"model_name": model_name},
            )

        elif self.kind == "sentencepiece":
            import sentencepiece as spm
            model_path = self.model or "spiece.model"
            sp = spm.SentencePieceProcessor()
            sp.load(model_path)
            self.tokenizer = sp
            self.meta = TokenizerMetadata(
                name="sentencepiece",
                version=spm.__version__,
                vocab_size=sp.get_piece_size(),
                extra={"model_path": model_path},
            )

        elif self.kind == "spacy":
            import spacy
            lang = self.model or "en_core_web_sm"
            nlp = spacy.load(lang, disable=["parser", "ner", "tagger"])
            self.tokenizer = nlp.tokenizer
            self.meta = TokenizerMetadata(
                name="spacy_word",
                version=importlib.metadata.version("spacy"),
                vocab_size=None,
                extra={"lang": lang},
            )

        elif self.kind == "bytes":
            self.tokenizer = None
            self.meta = TokenizerMetadata(
                name="utf8_bytes",
                version="builtin",
                vocab_size=256,
                extra={},
            )
        else:
            raise ValueError(f"Unknown tokenizer kind: {self.kind}")

    # ------------------------------------------------------------------
    # 3. Encode / Decode interface
    # ------------------------------------------------------------------
    def encode(self, text: str) -> list[int]:
        """Normalize → pre-tokenize → encode into token IDs (or byte values)."""
        text = self._normalize(text)
        chunks = self._pretokenize(text)

        if self.kind.startswith("gpt4"):
            return [tid for c in chunks for tid in self.tokenizer.encode(c)]
        elif self.kind == "wordpiece":
            return [tid for c in chunks for tid in self.tokenizer.encode(c, add_special_tokens=False)]
        elif self.kind == "sentencepiece":
            return [tid for c in chunks for tid in self.tokenizer.encode_as_ids(c)]
        elif self.kind == "spacy":
            return [t.text for c in chunks for t in self.tokenizer(c)]
        elif self.kind == "bytes":
            return [b for c in chunks for b in c.encode("utf-8")]

    def decode(self, tokens: list[int]) -> str:
        """Decode token IDs back to string (if supported)."""
        if self.kind.startswith("gpt4"):
            return self.tokenizer.decode(tokens)
        elif self.kind == "wordpiece":
            return self.tokenizer.decode(tokens)
        elif self.kind == "sentencepiece":
            return self.tokenizer.decode_ids(tokens)
        elif self.kind == "spacy":
            return " ".join(tokens)
        elif self.kind == "bytes":
            return bytes(tokens).decode("utf-8", errors="ignore")

    # ------------------------------------------------------------------
    # 4. Metadata
    # ------------------------------------------------------------------
    def metadata(self) -> Dict[str, Any]:
        """Return tokenizer + preprocessing metadata for reproducibility."""
        d = asdict(self.meta)
        d["preprocessing"] = dict(
            lowercase=self.lowercase,
            normalize_digits=self.normalize_digits,
            add_space_punct=self.add_space_punct,
            pretokenize=self.pretokenize,
        )
        d["fingerprint"] = hashlib.md5(json.dumps(d, sort_keys=True).encode()).hexdigest()
        return d



        