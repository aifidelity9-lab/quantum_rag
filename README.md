# Quantum Book RAG

A small local RAG project for asking questions about a book or document collection.

It can:

- ingest `PDF`, `TXT`, and `MD` files,
- split them into chunks,
- build a local TF-IDF style retrieval index with `numpy`,
- answer questions from the top retrieved passages,
- optionally send grounded context to Gemini or Ollama for answer generation.

The project is intentionally lightweight. It does not require `faiss`, `chromadb`, or any cloud vector database.

## Features

- local document ingestion
- retrieval-first answers with source snippets
- optional Gemini integration through the official REST API
- optional Ollama integration for local generation
- VS Code launch and task configuration included

## Project Layout

```text
quantum_rag/
|-- books/              # source documents
|-- data/               # generated indexes
|-- src/quantum_rag/    # package code
|-- .vscode/            # VS Code tasks and launch configs
|-- pyproject.toml
|-- requirements.txt
`-- README.md
```

## Requirements

- Python 3.10+
- Windows PowerShell or VS Code terminal

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -e .
```

## Quick Start

Build the sample index:

```powershell
.\.venv\Scripts\python -m quantum_rag ingest --input src/sample_quantum.txt --name demo_quantum
```

Ask a question:

```powershell
.\.venv\Scripts\python -m quantum_rag ask --name demo_quantum --question "What is a wave function?"
```

Start interactive chat:

```powershell
.\.venv\Scripts\python -m quantum_rag chat --name demo_quantum
```

## Add Your Own Book

Put your file in `books/`, for example:

```text
books/griffiths_quantum.pdf
```

Then ingest it:

```powershell
.\.venv\Scripts\python -m quantum_rag ingest --input books/griffiths_quantum.pdf --name quantum_book
```

Then ask questions:

```powershell
.\.venv\Scripts\python -m quantum_rag ask --name quantum_book --question "What is the physical meaning of the wave function?"
```

## Use Gemini

Set your API key:

```powershell
$env:GEMINI_API_KEY="your_api_key"
```

Ask with Gemini:

```powershell
.\.venv\Scripts\python -m quantum_rag ask --name quantum_book --question "Explain quantum tunneling." --llm-provider gemini --llm-model gemini-2.5-flash
```

You can also pass the key directly:

```powershell
.\.venv\Scripts\python -m quantum_rag ask --name quantum_book --question "Explain quantum tunneling." --llm-provider gemini --gemini-api-key "your_api_key"
```

## Use Ollama

If Ollama is already running locally:

```powershell
.\.venv\Scripts\python -m quantum_rag ask --name quantum_book --question "Explain quantum tunneling." --llm-provider ollama --llm-model qwen2.5:1.5b
```

Default Ollama endpoint:

```text
http://127.0.0.1:11434/api/generate
```

Override it with `--ollama-url` or `OLLAMA_URL`.

## VS Code

The project includes `.vscode` configuration for:

- `RAG: Ingest Sample`
- `RAG: Ask Sample`
- `RAG: Chat Sample`

And tasks for:

- creating the virtual environment
- installing dependencies
- ingesting the sample file

## Notes

- Retrieval works offline after Python dependencies are installed.
- Gemini requires internet access and a valid API key.
- Ollama support is optional.
- Generated answers fall back to retrieval-only output if the LLM call fails.

## Future Improvements

- embedding-based retrieval
- citation formatting
- FastAPI or Streamlit UI
- OCR for scanned PDFs
