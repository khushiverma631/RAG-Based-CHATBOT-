
from sentence_transformers import SentenceTransformer, util
from ingest import ingest_pdf, PDF_PATH


def main():
    # Step 1: Read and chunk the research paper.
    chunks = ingest_pdf(PDF_PATH)

    if not chunks:
        print("No text chunks found. Check your PDF.")
        return

    # Step 2: Load a pretrained embedding model.
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Step 3: Convert every chunk into a numerical vector.
    chunk_texts = [chunk["text"] for chunk in chunks]

    embeddings = model.encode(
        chunk_texts,
        convert_to_tensor=True,
        show_progress_bar=True
    )

    print(f"\nNumber of chunks: {len(chunks)}")
    print(f"Embedding matrix shape: {embeddings.shape}")

    # Step 4: Embed a question.
    question = "How does the U-Net architecture work?"

    question_embedding = model.encode(
        question,
        convert_to_tensor=True
    )

    # Step 5: Compare the question with all document chunks.
    similarities = util.cos_sim(question_embedding, embeddings)[0]

    # Step 6: Retrieve the three most similar chunks.
    top_results = similarities.topk(min(3, len(chunks)))

    print(f"\nQuestion: {question}")

    for rank, (index, score) in enumerate(
        zip(top_results.indices.tolist(), top_results.values.tolist()),
        start=1
    ):
        chunk = chunks[index]

        print(f"\n--- Result {rank} ---")
        print(f"Page: {chunk['page']}")
        print(f"Similarity: {score:.4f}")
        print(chunk["text"][:500])


if __name__ == "__main__":
    main()
