"""Chunking, per section 5.4: GitHub-native units over token windows.

A short comment is never split. A single review comment usually *is* one complete
thought, and splitting it destroys the thing that made it worth retrieving.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any

from relore.ingest.versions import CHUNKING_VERSION

# ~1500 tokens, via the usual four-characters-per-token proxy. Deliberately not a real
# tokenizer: that would add a dependency to a boundary that is a heuristic anyway, and
# the only thing this number has to do is keep one document from swallowing a thread.
MAX_CHUNK_CHARS = 6000

_HEADING = re.compile(r"^(#{1,6})\s+\S", re.MULTILINE)


@dataclass(frozen=True)
class Chunk:
    index: int
    text: str


def content_hash(
    text: str,
    metadata: dict[str, Any] | None = None,
    *,
    chunking_version: int = CHUNKING_VERSION,
) -> str:
    """Section 5.1, with the metadata included in the identity.

    The chunking version is inside the hash so bumping it invalidates every document
    exactly once. The embedding model is deliberately *not*: it is recorded in
    ``embedding_model``, which every rewrite resets, so a model change re-embeds through
    ``relored embed`` without rewriting a document or filling ``documents_history``.

    Metadata is in there because for an inline review comment the file and line *are*
    part of what the document is. Hashing the text alone means a comment whose line moved
    -- or went null after a force-push -- keeps its stale location for ever, since the
    reconcile in section 5.1 compares only hashes and would see no change. The code lens
    would then resolve the wrong enclosing symbol from it, silently.
    """
    canonical = json.dumps(metadata or {}, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(f"{chunking_version}\x00{text}\x00{canonical}".encode()).hexdigest()


def chunk(text: str, *, split: bool = True) -> list[Chunk]:
    """Split ``text`` into documents.

    ``split=False`` is the promise that a unit is never divided regardless of length --
    a title, or an inline review comment tied to one file and line.
    """
    text = text.strip()
    if not text:
        return []
    if not split or len(text) <= MAX_CHUNK_CHARS:
        return [Chunk(0, text)]
    return [Chunk(i, part) for i, part in enumerate(_split_headings(text))]


def _split_headings(text: str) -> list[str]:
    """Heading-aware first, paragraphs second, hard cut only as a last resort."""
    starts = [m.start() for m in _HEADING.finditer(text)]
    if starts and starts[0] != 0:
        starts.insert(0, 0)
    sections = (
        [text[a:b].strip() for a, b in zip(starts, [*starts[1:], len(text)], strict=True)]
        if len(starts) > 1
        else [text]
    )

    out: list[str] = []
    for section in sections:
        if not section:
            continue
        if len(section) <= MAX_CHUNK_CHARS:
            out.append(section)
        else:
            out.extend(_split_paragraphs(section))
    return out or [text[:MAX_CHUNK_CHARS]]


def _split_paragraphs(section: str) -> list[str]:
    out: list[str] = []
    buffer = ""
    for paragraph in section.split("\n\n"):
        candidate = f"{buffer}\n\n{paragraph}" if buffer else paragraph
        if len(candidate) <= MAX_CHUNK_CHARS:
            buffer = candidate
            continue
        if buffer:
            out.append(buffer)
        # A single paragraph over the limit is usually a pasted log or tensor dump. There
        # is no natural boundary inside it, so cut on length and accept it.
        while len(paragraph) > MAX_CHUNK_CHARS:
            out.append(paragraph[:MAX_CHUNK_CHARS])
            paragraph = paragraph[MAX_CHUNK_CHARS:]
        buffer = paragraph
    if buffer:
        out.append(buffer)
    return out
