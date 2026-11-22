"""Section 6.1's vector tier: the embedding model, behind the ``[embed]`` extra.

Server side only -- invariant 1 keeps every ML package out of the client's import graph,
and ``fastembed`` is imported inside :class:`Embedder` so that importing this module costs
nothing on a daemon that never turns the tier on.

**Off unless configured.** ``RELORE_EMBED_MODEL`` names the model; unset, nothing embeds and
nothing searches by vector. The model is recorded per document in ``embedding_model``, and
a search reads only rows whose value equals the configured model -- so an edited document
(the reconcile resets the column) or a model swap never serves a stale vector; the next
``relored embed`` pass replaces it.

Measured before shipping (huggingface/relore#79): ``BAAI/bge-small-en-v1.5`` fused with the
lexical legs, gated to prose queries, took the prose precedent slice from R@10 0.575 to
0.725 with no frozen-set row moving. Larger models (bge-base, Qwen3-Embedding 0.6B and 4B)
were not better in aggregate.
"""

from __future__ import annotations

import os
from collections.abc import Sequence

MODEL_ENV = "RELORE_EMBED_MODEL"
#: The model the measurement chose, and the column's width: ``vector(384)``.
DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
DIMENSIONS = 384
#: Texts per model call. Small, because a batch pads to its longest text.
EMBED_BATCH = 32

#: BGE's retrieval instruction. Documents are embedded bare; only the query carries it.
QUERY_PREFIXES = {
    "BAAI/bge-small-en-v1.5": "Represent this sentence for searching relevant passages: ",
}


class Embedder:
    """One loaded model. Load once per process: it is the expensive part."""

    def __init__(self, model: str = DEFAULT_MODEL) -> None:
        try:
            from fastembed import TextEmbedding
        except ImportError:
            raise SystemExit(
                f"{MODEL_ENV} is set but the embed extra is not installed.\n"
                "  pip install 'relore[embed]'"
            ) from None
        self.model = model
        self._model = TextEmbedding(model_name=model)
        self._prefix = QUERY_PREFIXES.get(model, "")
        width = len(self.query("probe"))
        if width != DIMENSIONS:
            # The column is vector(384); a wider model fails on the first write, mid-pass.
            raise SystemExit(f"{model} embeds to {width} dimensions; the index holds {DIMENSIONS}")

    def documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Shortest first, in small batches, returned in the caller's order.

        A batch pads to its longest member, and id order mixes 40-character comments with
        3,000-character tracebacks: measured on the transformers sample, 13 docs/s in id
        order against 37 sorted, and fastembed's default batch of 256 held 8 GB.
        """
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        vectors = self._model.embed([texts[i] for i in order], batch_size=EMBED_BATCH)
        out: list[list[float]] = [[] for _ in texts]
        for i, vector in zip(order, vectors, strict=True):
            out[i] = vector.tolist()
        return out

    def query(self, text: str) -> list[float]:
        return next(iter(self._model.embed([self._prefix + text]))).tolist()


def configured(environ: dict[str, str] | None = None) -> Embedder | None:
    """The embedder ``RELORE_EMBED_MODEL`` asks for, or ``None`` when the tier is off."""
    model = (environ if environ is not None else os.environ).get(MODEL_ENV, "").strip()
    return Embedder(model) if model else None
