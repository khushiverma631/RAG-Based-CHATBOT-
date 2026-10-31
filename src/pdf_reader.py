
from pathlib import Path
import pymupdf

# Find the project folder and PDF file.
PROJECT_DIR = Path(__file__).resolve().parent.parent
PDF_PATH = PROJECT_DIR / "data" / "research_paper.pdf"


def extract_pdf_pages(pdf_path):
    """Extract text from each PDF page."""

    pages = []

    with pymupdf.open(pdf_path) as document:
        for page_number, page in enumerate(document, start=1):
            text = page.get_text("text", sort=False).strip()

            pages.append({
                "page": page_number,
                "text": text
            })

    return pages


def main():
    if not PDF_PATH.exists():
        print(f"PDF not found: {PDF_PATH}")
        print("Place research_paper.pdf inside the data folder.")
        return

    pages = extract_pdf_pages(PDF_PATH)

    total_characters = sum(len(page["text"]) for page in pages)

    print(f"Total pages: {len(pages)}")
    print(f"Total extracted characters: {total_characters}")

    for page in pages[:2]:
        print(f"\n--- Page {page['page']} ---")
        print(page["text"][:1000])

    if total_characters == 0:
        print("\nNo text extracted. This PDF may require OCR.")


if __name__ == "__main__":
    main()
