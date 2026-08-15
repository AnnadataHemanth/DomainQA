from __future__ import annotations

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def main() -> None:
    files = sorted(PROCESSED_DIR.glob("*.json"))

    if not files:
        print("No processed documents found.")
        return

    total_chars = 0
    total_pages = 0

    print("\n=== DomainQA Corpus Inspection ===\n")

    for file_path in files:
        with file_path.open("r", encoding="utf-8") as file:
            document = json.load(file)

        text = document.get("text", "")
        pages = document.get("pages", [])

        total_chars += len(text)
        total_pages += len(pages)

        print(f"ID:              {document['id']}")
        print(f"Title:           {document['title']}")
        print(f"Pages:           {document['num_pages']}")
        print(f"Extracted pages: {document['extracted_pages']}")
        print(f"Characters:      {len(text):,}")
        print(f"Words:           {len(text.split()):,}")
        print("-" * 70)

    print("\n=== Corpus Summary ===")
    print(f"Documents:   {len(files)}")
    print(f"Pages:       {total_pages:,}")
    print(f"Characters:  {total_chars:,}")
    print(f"Approx words:{total_chars / 5:,.0f}")


if __name__ == "__main__":
    main()