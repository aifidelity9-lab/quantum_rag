from __future__ import annotations

import argparse
from pathlib import Path
import sys

from .indexer import build_index
from .llm import gemini_generate, ollama_generate
from .retriever import RagIndex, answer_question, format_context


ROOT = Path(__file__).resolve().parents[2]
BOOKS_DIR = ROOT / "books"
DATA_DIR = ROOT / "data"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Local book RAG for quantum physics study.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest = subparsers.add_parser("ingest", help="Ingest a book and build the index.")
    ingest.add_argument("--input", required=True, help="Path to PDF/TXT/MD book file.")
    ingest.add_argument("--name", required=True, help="Dataset name to save under data/.")
    ingest.add_argument("--chunk-size", type=int, default=900)
    ingest.add_argument("--overlap", type=int, default=180)
    ingest.add_argument("--dims", type=int, default=2048)

    ask = subparsers.add_parser("ask", help="Ask a single question.")
    ask.add_argument("--name", required=True, help="Dataset name under data/.")
    ask.add_argument("--question", required=True, help="Question to ask.")
    ask.add_argument("--top-k", type=int, default=5)
    ask.add_argument("--llm-provider", choices=["none", "ollama", "gemini"], default="none")
    ask.add_argument("--llm-model", default=None)
    ask.add_argument("--ollama-url", default=None)
    ask.add_argument("--gemini-api-key", default=None)

    chat = subparsers.add_parser("chat", help="Open an interactive chat loop.")
    chat.add_argument("--name", required=True, help="Dataset name under data/.")
    chat.add_argument("--top-k", type=int, default=5)
    chat.add_argument("--llm-provider", choices=["none", "ollama", "gemini"], default="none")
    chat.add_argument("--llm-model", default=None)
    chat.add_argument("--ollama-url", default=None)
    chat.add_argument("--gemini-api-key", default=None)

    return parser


def cmd_ingest(args: argparse.Namespace) -> int:
    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = ROOT / input_path

    if not input_path.exists():
        print(f"Input file not found: {input_path}")
        return 1

    dataset_dir = build_index(
        input_path=input_path,
        dataset_name=args.name,
        data_dir=DATA_DIR,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        dims=args.dims,
    )

    print(f"Indexed book into: {dataset_dir}")
    return 0


def render_answer(
    index: RagIndex,
    question: str,
    top_k: int,
    llm_provider: str = "none",
    llm_model: str | None = None,
    ollama_url: str | None = None,
    gemini_api_key: str | None = None,
) -> None:
    results = index.search(question, top_k=top_k)
    grounded_answer = answer_question(question, results)
    final_answer = grounded_answer
    llm_note: str | None = None

    if llm_provider == "ollama" and results:
        context = format_context(results)
        llm_result = ollama_generate(
            question=question,
            context=context,
            model=llm_model or "qwen2.5:3b",
            base_url=ollama_url,
        )
        if llm_result.ok:
            final_answer = llm_result.text
            llm_note = f"Generated with Ollama model `{llm_model or 'qwen2.5:3b'}`."
        else:
            llm_note = f"Ollama unavailable, used retrieval-only fallback: {llm_result.error}"
    elif llm_provider == "gemini" and results:
        context = format_context(results)
        llm_result = gemini_generate(
            question=question,
            context=context,
            model=llm_model,
            api_key=gemini_api_key,
        )
        if llm_result.ok:
            final_answer = llm_result.text
            llm_note = f"Generated with Gemini model `{llm_model or 'gemini-2.5-flash'}`."
        else:
            llm_note = f"Gemini unavailable, used retrieval-only fallback: {llm_result.error}"

    print()
    print("Answer")
    print("------")
    print(final_answer)
    if llm_note:
        print()
        print(f"Note: {llm_note}")
    print()
    print("Top Passages")
    print("------------")
    if not results:
        print("No relevant passages found.")
        return

    for rank, item in enumerate(results, start=1):
        location = f"page {item.page}" if item.page is not None else "no page"
        snippet = item.text.replace("\n", " ")
        if len(snippet) > 300:
            snippet = snippet[:300].rsplit(" ", 1)[0] + "..."
        print(f"{rank}. score={item.score:.4f} | {item.source} | {location}")
        print(f"   {snippet}")


def cmd_ask(args: argparse.Namespace) -> int:
    dataset_dir = DATA_DIR / args.name
    if not dataset_dir.exists():
        print(f"Dataset not found: {dataset_dir}")
        return 1

    index = RagIndex(dataset_dir)
    render_answer(
        index,
        args.question,
        args.top_k,
        args.llm_provider,
        args.llm_model,
        args.ollama_url,
        args.gemini_api_key,
    )
    return 0


def cmd_chat(args: argparse.Namespace) -> int:
    dataset_dir = DATA_DIR / args.name
    if not dataset_dir.exists():
        print(f"Dataset not found: {dataset_dir}")
        return 1

    index = RagIndex(dataset_dir)
    print("Interactive RAG chat. Type 'exit' to quit.")
    while True:
        try:
            question = input("\nQuestion> ").strip()
        except EOFError:
            print()
            break

        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            break

        render_answer(
            index,
            question,
            args.top_k,
            args.llm_provider,
            args.llm_model,
            args.ollama_url,
            args.gemini_api_key,
        )

    return 0


def main() -> int:
    BOOKS_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    parser = build_parser()
    args = parser.parse_args()

    if args.command == "ingest":
        return cmd_ingest(args)
    if args.command == "ask":
        return cmd_ask(args)
    if args.command == "chat":
        return cmd_chat(args)

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
