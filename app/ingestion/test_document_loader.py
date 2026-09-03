from pathlib import Path

import pytest

from app.ingestion.document_loader import load_document


def test_load_document_rejects_empty_file(tmp_path: Path):
    empty_file = tmp_path / "empty.txt"
    empty_file.write_bytes(b"")

    with pytest.raises(ValueError, match="empty"):
        load_document(str(empty_file))


def test_load_document_rejects_empty_extracted_text(tmp_path: Path):
    blank_file = tmp_path / "blank.txt"
    blank_file.write_text("   \n\t", encoding="utf-8")

    with pytest.raises(ValueError, match="readable text"):
        load_document(str(blank_file))
