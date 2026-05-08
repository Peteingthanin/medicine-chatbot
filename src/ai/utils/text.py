"""
utils/text.py — Pure text-processing helpers for chunk formatting.
"""
from ai.models.schemas import SourceChunk


def chunks_to_context(chunks) -> str:
    """Flatten retrieved chunks into an LLM-readable context block."""
    parts = []
    for c in chunks:
        detail_text = "\n".join(f"- {d}" for d in c.detail)
        parts.append(
            f"[{c.pill_name}] {c.section}"
            + (f" > {c.subsection}" if c.subsection else "")
            + f"\n{detail_text}"
        )
    return "\n\n".join(parts)


def to_source_chunks(chunks) -> list[SourceChunk]:
    """Convert internal RetrievedChunk dataclasses to API SourceChunk models."""
    return [
        SourceChunk(
            pill_name=c.pill_name,
            section=c.section,
            subsection=c.subsection,
            detail=c.detail,
            score=round(c.score, 4),
        )
        for c in chunks
    ]
