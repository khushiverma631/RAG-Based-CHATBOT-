from pathlib import Path

from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

from ingest import ingest_pdf, PDF_PATH


class DocumentRetriever:
    def __init__(self, pdf_paths=None):
        # Use the existing PDF by default, for backward compatibility.
        if pdf_paths is None:
            pdf_paths = [PDF_PATH]

        self.chunks = []

        print("Loading and processing PDFs...")

        for pdf_path in pdf_paths:
            pdf_path = Path(pdf_path)

            if not pdf_path.is_file():
                raise FileNotFoundError(f"PDF not found: {pdf_path}")

            if pdf_path.suffix.lower() != ".pdf":
                raise ValueError(f"Not a PDF file: {pdf_path.name}")

            pdf_chunks = ingest_pdf(pdf_path)
            self.chunks.extend(pdf_chunks)

        if not self.chunks:
            raise ValueError("No text chunks found in the supplied PDFs.")

        print(f"Total chunks from all PDFs: {len(self.chunks)}")

        print("Loading embedding model...")
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

        texts = [chunk["text"] for chunk in self.chunks]

        print("Creating document embeddings...")
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=True
        )

        embeddings = np.asarray(embeddings, dtype="float32")
        faiss.normalize_L2(embeddings)

        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings)

        print(f"Indexed {self.index.ntotal} chunks.")

    def retrieve(self, question, top_k=5):
        query_embedding = self.model.encode(
            [question],
            convert_to_numpy=True
        )

        query_embedding = np.asarray(
            query_embedding, dtype="float32"
        )
        faiss.normalize_L2(query_embedding)

        k = min(top_k, len(self.chunks))
        scores, indices = self.index.search(query_embedding, k)

        results = []

        for idx, score in zip(indices[0], scores[0]):
            if idx < 0:
                continue

            chunk = self.chunks[int(idx)]

            results.append({
                "text": chunk["text"],
                "page": chunk["page"],
                "chunk_index": chunk["chunk_index"],
                "source": chunk["source"],
                "score": float(score)
            })

        return results


if __name__ == "__main__":
    retriever = DocumentRetriever()

    question = "How does VIN help an agent plan a path?"
    results = retriever.retrieve(question)

    print(f"\nQuestion: {question}")

    for rank, result in enumerate(results, start=1):
        print(f"\nResult {rank}")
        print(f"Source: {result['source']}")
        print(f"Page: {result['page']}")
        print(f"Similarity: {result['score']:.4f}")
        print(result["text"])
