from pathlib import Path

import pandas as pd
from pypdf import PdfReader


SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".txt",
    ".csv",
    ".xlsx",
}


def extract_text_from_pdf(file_path: str) -> str:
    """Extract text from a PDF file."""

    reader = PdfReader(file_path)

    pages = []

    for page in reader.pages:
        text = page.extract_text()

        if text:
            pages.append(text)

    return "\n".join(pages)


def extract_text_from_txt(file_path: str) -> str:
    """Extract text from a text file."""

    with open(
        file_path,
        "r",
        encoding="utf-8",
        errors="ignore"
    ) as file:
        return file.read()


def extract_text_from_csv(file_path: str) -> str:
    """Convert CSV content into text."""

    dataframe = pd.read_csv(file_path)

    return dataframe.to_string(index=False)


def extract_text_from_excel(file_path: str) -> str:
    """Convert Excel content into text."""

    dataframe = pd.read_excel(file_path)

    return dataframe.to_string(index=False)


def load_document(file_path: str) -> dict:
    """
    Load a supported document and return its extracted text.
    """

    path = Path(file_path)

    extension = path.suffix.lower()

    if extension not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type: {extension}"
        )

    if extension == ".pdf":
        text = extract_text_from_pdf(file_path)

    elif extension == ".txt":
        text = extract_text_from_txt(file_path)

    elif extension == ".csv":
        text = extract_text_from_csv(file_path)

    elif extension == ".xlsx":
        text = extract_text_from_excel(file_path)

    else:
        raise ValueError(
            f"Unsupported file type: {extension}"
        )

    return {
    "document_id": path.stem,
    "file_name": path.name,
    "file_type": extension,
    "source": str(path),
    "text": text,
    "character_count": len(text),
}