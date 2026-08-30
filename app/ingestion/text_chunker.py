import re


def _split_long_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = max(end - chunk_overlap, start + 1)
    return chunks


def chunk_text(
    text: str,
    chunk_size: int = 800,
    chunk_overlap: int = 120,
) -> list[str]:
    """Create heading-aware chunks, then split oversized sections safely.

    The function keeps the original public API (list[str]) so existing
    ingestion code continues to work. Logical headings are kept with the
    following content, improving retrieval compared with character-only
    splitting.
    """
    if not text or not text.strip():
        return []
    if chunk_size <= 0 or chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_size must be positive and overlap must be smaller than chunk_size")

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return _split_long_text(text.strip(), chunk_size, chunk_overlap)

    heading_pattern = re.compile(
        r"^(?:#{1,6}\s+|(?:section|chapter|part)\s+[\w.-]+(?:\s*[-:.)]\s*|\s+))",
        re.IGNORECASE,
    )

    sections: list[str] = []
    current: list[str] = []
    for line in lines:
        looks_like_heading = bool(heading_pattern.match(line)) or (
            len(line) <= 100 and line.isupper() and len(line.split()) <= 12
        )
        if looks_like_heading and current:
            sections.append("\n".join(current))
            current = [line]
        else:
            current.append(line)
    if current:
        sections.append("\n".join(current))

    chunks: list[str] = []
    for section in sections:
        if len(section) <= chunk_size:
            chunks.append(section)
            continue
        parts = _split_long_text(section, chunk_size, chunk_overlap)
        chunks.extend(parts)

    return chunks
