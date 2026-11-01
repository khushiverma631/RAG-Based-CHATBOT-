
from pathlib import Path
import pymupdf


PROJECT_DIR = Path(__file__).resolve().parent.parent
PDF_PATH = PROJECT_DIR / "data" / "research_paper.pdf"


def extract_pdf_pages(pdf_path):
    """Return the text and page number for each PDF page."""
    pages = []

    with pymupdf.open(pdf_path) as document:
        for page_number, page in enumerate(document, start=1):
            text = page.get_text("text", sort=False).strip()

            if text:
                pages.append({
                    "page": page_number,
                    "text": text
                })

    return pages


def split_into_chunks(text, chunk_size=1000, overlap=150):
    """Split text into overlapping character-based chunks."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])

        if end == len(text):
            break

        start = end - overlap

    return chunks


#new change 


def ingest_pdf(pdf_path):
    """Extract a PDF and create chunks with source and page metadata."""
    pdf_path = Path(pdf_path)
    pages = extract_pdf_pages(pdf_path)
    all_chunks = []

    for page in pages:
        page_chunks = split_into_chunks(page["text"])

        for chunk_index, chunk_text in enumerate(page_chunks):
            all_chunks.append({
                "text": chunk_text,
                "page": page["page"],
                "chunk_index": chunk_index,
                "source": pdf_path.name
            })

    return all_chunks





def main():
    if not PDF_PATH.exists():
        print(f"PDF not found: {PDF_PATH}")
        return

    chunks = ingest_pdf(PDF_PATH)

    print(f"Total chunks created: {len(chunks)}")

    for chunk in chunks[:3]:
        print(f"\nPage: {chunk['page']}")
        print(f"Chunk: {chunk['chunk_index']}")
        print(chunk["text"][:300])


if __name__ == "__main__":
    main()
