# PDF Chat Application

A web-based application that allows users to upload PDF documents and interact with them through natural language queries. The system extracts text from PDFs, converts it into vector embeddings, retrieves relevant content using semantic search, and generates answers using a language model.

---

## Overview

This project is built to solve a common problem: reading and extracting useful information from large PDF documents. Instead of manually searching through files, users can simply ask questions and receive answers based on the document content.

The application follows a retrieval-augmented generation (RAG) approach:
- It does not rely purely on a language model
- It first finds relevant context from documents
- Then generates answers grounded in that context

---

## Features

- Upload and process multiple PDF files at once  
- Automatic text extraction and cleaning  
- Smart chunking of large text into smaller segments  
- Semantic search using embeddings  
- Context-aware answer generation  
- Displays answers along with their source documents  

---

## Tech Stack

**Frontend**
- Streamlit (for building the UI)

**Backend**
- Python

**Core Libraries**
- `sentence-transformers` – for generating embeddings  
- `faiss` – for fast similarity search  
- `pypdf` – for extracting text from PDFs  
- `requests` – for API communication  

**Model API**
- Hugging Face Inference API (Mistral-7B-Instruct)

---

## System Architecture

The workflow of the application can be broken down into the following steps:

### 1. PDF Upload
Users upload one or more PDF files through the Streamlit interface.

### 2. Text Extraction
- Each PDF is read page by page
- Text is extracted and cleaned
- Empty or invalid content is ignored

### 3. Text Chunking
- Large text is split into smaller chunks
- Overlapping is used to preserve context between chunks

### 4. Embedding Generation
- Each chunk is converted into a vector using Sentence Transformers
- These embeddings represent semantic meaning

### 5. Vector Storage
- All embeddings are stored in a FAISS index
- This enables fast similarity-based retrieval

### 6. Query Processing
- User input is converted into an embedding
- Top-K most relevant chunks are retrieved

### 7. Answer Generation
- Retrieved chunks are passed as context to the language model
- The model generates an answer based only on that context

---

## Example Workflow

1. Upload a PDF document  
2. Click on "Process PDFs"  
3. Ask a question such as:  
   > "What are the key findings of the report?"  
4. The system retrieves relevant sections  
5. The model generates an answer based on those sections  

---

## Limitations

- Performance depends on the quality of PDF text extraction  
- Scanned PDFs or image-based content may not work properly  
- Responses are limited to retrieved context (no external knowledge)  
- Hugging Face API may introduce latency or rate limits  

---

## Future Improvements

- Support for scanned PDFs using OCR  
- Improved chunking using NLP techniques  
- Integration of local LLMs for faster inference  
- Better UI with chat history and highlighting sources  
- Support for additional file formats (DOCX, TXT)  

---

## License

This project is open source.
