# AskMyDoc — PDF-Based RAG Chatbot

AskMyDoc is a Retrieval-Augmented Generation (RAG) chatbot that allows users to upload one or multiple PDF documents and ask questions about their content. It retrieves relevant passages from the uploaded documents and uses a Large Language Model (LLM) to generate contextual answers.

## How It Works

AskMyDoc follows a chunk-based RAG architecture:

1. **PDF Text Extraction:** Extract text from uploaded PDF documents using PyMuPDF.
2. **Text Chunking:** Split extracted text into smaller, overlapping chunks to preserve context across chunk boundaries.
3. **Embedding Generation:** Convert text chunks into numerical vector representations using Sentence Transformers.
4. **Vector Indexing:** Store the embeddings in a FAISS index for efficient similarity-based retrieval.
5. **Relevant Context Retrieval:** Retrieve the top-matching chunks for a user's question using cosine similarity.
6. **Answer Generation:** Pass the retrieved context and user question to an LLM through the Groq API to generate a relevant answer.

## Tech Stack

- **Language:** Python
- **Backend:** Flask
- **PDF Processing:** PyMuPDF
- **Embeddings:** Sentence Transformers (`all-MiniLM-L6-v2`)
- **Vector Search:** FAISS
- **LLM Integration:** LangChain and Groq API
- **Frontend:** HTML, CSS, JavaScript


## Setup and Installation

### 1. Clone the Repository 

```bash
git clone https://github.com/khushiverma631/AskMyDoc.git
cd AskMyDoc
```

### 2. Create and Activate the Environment

Make sure Python and Conda are installed.

```bash
conda create -n askmydoc python=3.11 -y
conda activate askmydoc
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file in the project root directory and add your Groq API key:

```env
GROQ_API_KEY=your_groq_api_key_here
```

Replace the placeholder with your own API key. **Never commit your `.env` file to GitHub.**

### 5. Run the Application

From the project root, run:

```bash
python src/app.py
```

Open the local URL displayed in your terminal, typically:

```text
http://127.0.0.1:5000
```

### 6. Use AskMyDoc

1. Upload one or more PDF documents.
2. Wait for document processing and embedding generation to finish.
3. Ask questions about the uploaded documents.
4. Review the generated answers and retrieved source passages.

## Features

- Upload and query multiple PDF documents.
- Retrieve relevant document passages using semantic similarity.
- Generate answers using retrieved context.
- Display source information, including document names and page numbers.
- Provide similarity scores for retrieved passages.
- Support general-knowledge responses when document context is insufficient, with appropriate labeling.

## Project Goal

The goal of AskMyDoc is to make information in PDF documents easier to access by combining semantic search with LLM-based answer generation.
