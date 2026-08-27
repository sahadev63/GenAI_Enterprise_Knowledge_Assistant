def chunk_text(
    text: str,
    chunk_size: int = 100,
    chunk_overlap: int = 20,
) -> list[str]:

    chunks = []

    start = 0

    while start < len(text):

        end = start + chunk_size

        chunk = text[start:end]

        chunks.append(chunk)

        start = end - chunk_overlap

    return chunks