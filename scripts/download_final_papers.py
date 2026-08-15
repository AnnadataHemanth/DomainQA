from __future__ import annotations

import time
from pathlib import Path

import requests
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MANIFEST_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "final_papers.yaml"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "final_pdfs"
)


def load_manifest() -> dict:
    """Load the final paper manifest."""

    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"Manifest not found: {MANIFEST_PATH}"
        )

    with MANIFEST_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        data = yaml.safe_load(file)

    if not data or "papers" not in data:
        raise ValueError(
            "Invalid final_papers.yaml: "
            "expected a 'papers' key."
        )

    return data


def download_pdf(
    arxiv_id: str,
    output_path: Path,
) -> None:
    """Download a PDF from arXiv."""

    url = f"https://arxiv.org/pdf/{arxiv_id}"

    response = requests.get(
        url,
        timeout=60,
        headers={
            "User-Agent": "DomainQA-Research/1.0"
        },
    )

    response.raise_for_status()

    output_path.write_bytes(
        response.content
    )


def main() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest = load_manifest()

    papers = manifest["papers"]

    print(
        f"Final corpus papers: {len(papers)}"
    )
    print(
        f"Output directory: {OUTPUT_DIR}\n"
    )

    downloaded = 0
    existing = 0
    failed = 0

    for paper in papers:

        paper_id = paper["id"]
        arxiv_id = paper.get("arxiv_id")

        if not arxiv_id:
            print(
                f"[SKIP] {paper_id}: "
                "no arXiv ID configured."
            )
            continue

        output_path = (
            OUTPUT_DIR
            / f"{paper_id}.pdf"
        )

        if output_path.exists():
            print(
                f"[EXISTS] {output_path.name}"
            )
            existing += 1
            continue

        try:
            print(
                f"[DOWNLOAD] {paper['title']}"
            )

            download_pdf(
                arxiv_id,
                output_path,
            )

            print(
                f"[SAVED] {output_path}"
            )

            downloaded += 1

        except requests.RequestException as exc:
            print(
                f"[ERROR] Failed to download "
                f"{paper_id}: {exc}"
            )
            failed += 1

        time.sleep(3)

    print("\n=== Final Download Summary ===")
    print(
        f"Total papers: {len(papers)}"
    )
    print(
        f"Downloaded:   {downloaded}"
    )
    print(
        f"Existing:     {existing}"
    )
    print(
        f"Failed:       {failed}"
    )


if __name__ == "__main__":
    main()