<div align="center">

# Quantum Book RAG

### A bright, lightweight RAG starter for studying quantum textbooks with Gemini or Ollama

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Gemini](https://img.shields.io/badge/LLM-Gemini-1A73E8?style=for-the-badge&logo=google&logoColor=white)
![Ollama](https://img.shields.io/badge/Local-Ollama-111111?style=for-the-badge)
![VS Code](https://img.shields.io/badge/IDE-VS%20Code-007ACC?style=for-the-badge&logo=visualstudiocode&logoColor=white)
![RAG](https://img.shields.io/badge/RAG-Minimal%20%26%20Readable-FF6B35?style=for-the-badge)

</div>

---

## Why This Project

`Quantum Book RAG` is a small, readable RAG project built for one practical goal:

**load a quantum physics book, retrieve the most relevant passages, and ask an LLM to answer using those passages as grounded context.**

It is intentionally lightweight:

- no vector database
- no LangChain dependency
- no heavy framework abstraction
- easy to read, easy to modify, easy to run in VS Code

That makes it a strong study project if you want to learn how a RAG pipeline works before moving to a larger stack.

## What It Does

```text
Document -> Chunking -> Local Retrieval Index -> Top Passages -> Gemini/Ollama Answer
```

It can:

- ingest `PDF`, `TXT`, and `MD` files
- split text into retrieval chunks
- build a local TF-IDF style index with `numpy`
- return the most relevant grounded passages
- send those passages to Gemini or Ollama for a final answer

## Highlights

| Feature | Description |
|---|---|
| Local retrieval | Fast, simple retrieval stored as `numpy` arrays and JSON metadata |
| Gemini support | Uses Gemini via the official REST `generateContent` API |
| Ollama support | Supports local generation if Ollama is running |
| VS Code ready | Includes `.vscode` launch and task config |
| Good for learning | Small enough to understand end-to-end |

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

Start chat mode:

```powershell
.\.venv\Scripts\python -m quantum_rag chat --name demo_quantum
```

## Bring Your Own Book

Drop a textbook into `books/`, for example:

```text
books/griffiths_quantum.pdf
```

Index it:

```powershell
.\.venv\Scripts\python -m quantum_rag ingest --input books/griffiths_quantum.pdf --name quantum_book
```

Then query it:

```powershell
.\.venv\Scripts\python -m quantum_rag ask --name quantum_book --question "What is the physical meaning of the wave function?"
```

## Gemini Workflow

Set your API key:

```powershell
$env:GEMINI_API_KEY="your_api_key"
```

Ask with Gemini:

```powershell
.\.venv\Scripts\python -m quantum_rag ask --name quantum_book --question "Explain quantum tunneling." --llm-provider gemini --llm-model gemini-2.5-flash
```

Or pass the key directly:

```powershell
.\.venv\Scripts\python -m quantum_rag ask --name quantum_book --question "Explain quantum tunneling." --llm-provider gemini --gemini-api-key "your_api_key"
```

## Ollama Workflow

If Ollama is running locally:

```powershell
.\.venv\Scripts\python -m quantum_rag ask --name quantum_book --question "Explain quantum tunneling." --llm-provider ollama --llm-model qwen2.5:1.5b
```

Default endpoint:

```text
http://127.0.0.1:11434/api/generate
```

Override with `--ollama-url` or `OLLAMA_URL`.

## VS Code Experience

Included launch configs:

- `RAG: Ingest Sample`
- `RAG: Ask Sample`
- `RAG: Chat Sample`

Included tasks:

- create virtual environment
- install dependencies
- ingest sample file

## Design Tradeoff

This is a **minimal, retrieval-first RAG**, not a production-scale semantic search stack.

Current retrieval is based on hashed TF-IDF, which means:

- it is simple and fast
- it is easy to study
- it is less semantically powerful than embedding-based retrieval

For textbook study, this is often good enough to start. If needed later, the retrieval layer can be upgraded to embeddings.

## Notes

- Retrieval works offline after dependencies are installed.
- Gemini requires internet access and a valid API key.
- Ollama support is optional.
- If the LLM call fails, the app falls back to retrieval-only output.

## Future Upgrades

- embedding-based retrieval
- citation formatting
- FastAPI or Streamlit UI
- OCR for scanned PDFs
