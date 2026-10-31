
from dotenv import load_dotenv
from langchain_groq import ChatGroq

from retriever import DocumentRetriever


# Load the API key from the project's .env file
load_dotenv()


def main():
    # Step 1: Initialize our existing retriever
    retriever = DocumentRetriever()

    # Step 2: Initialize the Groq LLM
    llm = ChatGroq(
        model="openai/gpt-oss-20b",
        temperature=0.2,
        max_tokens=500,
    )

    # Step 3: Ask a question about the research paper
    question = "How does VIN help an agent plan a path?"

    # Step 4: Retrieve relevant passages
    results = retriever.retrieve(question, top_k=3)

    # Step 5: Build context from the retrieved passages
    context_parts = []

    for result in results:
        context_parts.append(
            f"[Page {result['page']}]\n{result['text']}"
        )

    context = "\n\n".join(context_parts)

    # Step 6: Tell the LLM how to answer
    prompt = f"""
You are a research-paper assistant.

Answer the question using only the context provided below.
If the context does not contain enough information, say so.
Do not invent facts or page references.

Context:
{context}

Question:
{question}

Answer:
"""

    # Step 7: Generate the answer
    response = llm.invoke(prompt)

    # Step 8: Display the answer and source pages
    print(f"\nQuestion: {question}")
    print("\nAnswer:\n")
    print(response.content)

    pages = sorted({result["page"] for result in results})
    print(f"\nRetrieved source pages: {pages}")


if __name__ == "__main__":
    main()
