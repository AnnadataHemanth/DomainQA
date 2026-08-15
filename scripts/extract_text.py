from __future__ import annotations

import json
import re
from pathlib import Path

import yaml
from pypdf import PdfReader


PROJECT_ROOT = Path(__file__).resolve().parents[1]

PDF_DIR = PROJECT_ROOT / "data" / "raw" / "pdfs"
MANIFEST_PATH = PROJECT_ROOT / "data" / "raw" / "papers.yaml"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"


def load_manifest() -> dict:
    """Load paper metadata from papers.yaml."""
    with MANIFEST_PATH.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)

    if not data or "papers" not in data:
        raise ValueError("Invalid papers.yaml.")

    return data


def normalize_text(text: str) -> str:
    """Clean extracted PDF text while preserving paragraph structure."""

    # Normalize line endings.
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Remove excessive spaces.
    text = re.sub(r"[ \t]+", " ", text)

    # Fix common PDF line-break hyphenation.
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)

    # Collapse 3+ consecutive newlines.
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def extract_paper(paper: dict) -> dict:
    """Extract text from every page while preserving page metadata."""

    pdf_path = PDF_DIR / f"{paper['id']}.pdf"

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    reader = PdfReader(str(pdf_path))

    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        raw_text = page.extract_text() or ""
        text = normalize_text(raw_text)

        if not text:
            continue

        pages.append(
            {
                "page": page_number,
                "text": text,
            }
        )

    full_text = "\n\n".join(page["text"] for page in pages)

    return {
        "id": paper["id"],
        "title": paper["title"],
        "arxiv_id": paper.get("arxiv_id"),
        "num_pages": len(reader.pages),
        "extracted_pages": len(pages),
        "text": full_text,
        "pages": pages,
    }


def main() -> None:
    """Extract all downloaded papers into JSON files."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest()

    for paper in manifest["papers"]:
        pdf_path = PDF_DIR / f"{paper['id']}.pdf"

        if not pdf_path.exists():
            print(f"[SKIP] {paper['id']} - PDF not found")
            continue

        output_path = OUTPUT_DIR / f"{paper['id']}.json"

        if output_path.exists():
            print(f"[EXISTS] {output_path.name}")
            continue

        try:
            result = extract_paper(paper)

            with output_path.open("w", encoding="utf-8") as file:
                json.dump(
                    result,
                    file,
                    indent=2,
                    ensure_ascii=False,
                )

            print(
                f"[OK] {paper['id']} | "
                f"{result['num_pages']} pages | "
                f"{result['extracted_pages']} extracted"
            )

        except Exception as exc:
            print(f"[ERROR] {paper['id']}: {exc}")


if __name__ == "__main__":
    main()