from __future__ import annotations

import json
import re
from pathlib import Path

import pymupdf
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MANIFEST_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "final_papers.yaml"
)

PDF_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "final_pdfs"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "final"
)


def clean_text(text: str) -> str:
    text = text.replace("\r", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def load_manifest() -> list[dict]:
    with MANIFEST_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        return yaml.safe_load(file)["papers"]


def extract_pdf(
    paper: dict,
) -> None:

    paper_id = paper["id"]

    pdf_path = (
        PDF_DIR
        / f"{paper_id}.pdf"
    )

    output_path = (
        OUTPUT_DIR
        / f"{paper_id}.json"
    )

    if not pdf_path.exists():
        print(
            f"[SKIP] {paper_id} - PDF not found"
        )
        return

    document = pymupdf.open(pdf_path)

    pages = []

    for page_number, page in enumerate(
        document,
        start=1,
    ):
        text = clean_text(
            page.get_text()
        )

        pages.append(
            {
                "page": page_number,
                "text": text,
                "char_count": len(text),
                "word_count": len(
                    text.split()
                ),
            }
        )

    record = {
        "id": paper_id,
        "title": paper["title"],
        "arxiv_id": paper.get("arxiv_id"),
        "pages": pages,
        "page_count": len(pages),
        "total_characters": sum(
            p["char_count"]
            for p in pages
        ),
        "total_words": sum(
            p["word_count"]
            for p in pages
        ),
    }

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            record,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print(
        f"[OK] {paper_id} | "
        f"{len(pages)} pages | "
        f"{record['total_words']} words"
    )


def main() -> None:

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    papers = load_manifest()

    print(
        f"Final corpus papers: "
        f"{len(papers)}\n"
    )

    for paper in papers:
        extract_pdf(paper)

    print(
        f"\nOutput: {OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()