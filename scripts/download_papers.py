from __future__ import annotations

import os
import time
from pathlib import Path

import requests
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = PROJECT_ROOT / "data" / "raw" / "papers.yaml"
OUTPUT_DIR = PROJECT_ROOT / "data" / "raw" / "pdfs"


def load_manifest() -> dict:
    """Load the paper manifest from YAML."""
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(f"Manifest not found: {MANIFEST_PATH}")

    with MANIFEST_PATH.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)

    if not data or "papers" not in data:
        raise ValueError("Invalid papers.yaml: expected a 'papers' key.")

    return data


def download_pdf(arxiv_id: str, output_path: Path) -> None:
    """Download a PDF from arXiv."""
    url = f"https://arxiv.org/pdf/{arxiv_id}"

    response = requests.get(
        url,
        timeout=60,
        headers={"User-Agent": "DomainQA-Research/1.0"},
    )

    response.raise_for_status()

    output_path.write_bytes(response.content)


def main() -> None:
    """Download all papers listed in the manifest."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest()

    for paper in manifest["papers"]:
        paper_id = paper["id"]
        arxiv_id = paper.get("arxiv_id")

        if not arxiv_id:
            print(f"[SKIP] {paper_id}: no arXiv ID configured.")
            continue

        output_path = OUTPUT_DIR / f"{paper_id}.pdf"

        if output_path.exists():
            print(f"[EXISTS] {output_path.name}")
            continue

        try:
            print(f"[DOWNLOAD] {paper['title']}")
            download_pdf(arxiv_id, output_path)
            print(f"[SAVED] {output_path}")
        except requests.RequestException as exc:
            print(f"[ERROR] Failed to download {paper_id}: {exc}")

        # Avoid hammering arXiv.
        time.sleep(3)


if __name__ == "__main__":
    main()