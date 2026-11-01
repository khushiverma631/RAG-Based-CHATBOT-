
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

from ingest import ingest_pdf, PDF_PATH


def main():
    # 1. Extract and split the PDF into chunks
    chunks = ingest_pdf(PDF_PATH)

    if not chunks:
        print("No text chunks found.")
        return

    # 2. Load the embedding model
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # 3. Convert each chunk into an embedding
    texts = [chunk["text"] for chunk in chunks]

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        show_progress_bar=True
    )

    # FAISS expects 32-bit floating-point vectors
    embeddings = np.asarray(embeddings, dtype="float32")

    # 4. Normalize vectors so inner product equals cosine similarity
    faiss.normalize_L2(embeddings)

    # 5. Create a FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)

    # 6. Store all chunk embeddings in the index
    index.add(embeddings)

    print(f"Chunks indexed: {index.ntotal}")

    # 7. Convert the user's question into an embedding
    question = "How does the U-Net architecture work?"

    query_embedding = model.encode(
        [question],
        convert_to_numpy=True
    )
    query_embedding = np.asarray(query_embedding, dtype="float32")
    faiss.normalize_L2(query_embedding)

    # 8. Retrieve the three most relevant chunks
    k = min(5, len(chunks))
    scores, indices = index.search(query_embedding, k)

    print(f"\nQuestion: {question}")

    for rank, (idx, score) in enumerate(
        zip(indices[0], scores[0]), start=1
    ):
        chunk = chunks[int(idx)]

        print(f"\nResult {rank}")
        print(f"Page: {chunk['page']}")
        print(f"Similarity: {score:.4f}")
        print(f"Text: {chunk['text'][:500]}")


if __name__ == "__main__":
    main()
