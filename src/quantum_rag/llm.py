from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any
from urllib.error import URLError, HTTPError
from urllib.request import Request, urlopen


DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
DEFAULT_GEMINI_URL_TEMPLATE = (
    "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
)


@dataclass
class LlmResult:
    ok: bool
    text: str
    error: str | None = None


def build_rag_prompt(question: str, context: str) -> str:
    return (
        "You are answering questions about a quantum physics book.\n"
        "Use only the provided context.\n"
        "If the context is insufficient, say so clearly.\n"
        "Cite supporting passages using their provided citation labels.\n\n"
        f"Question:\n{question}\n\n"
        f"Context:\n{context}\n\n"
        "Answer:"
    )


def ollama_generate(
    question: str,
    context: str,
    model: str,
    base_url: str | None = None,
    timeout_s: int = 45,
) -> LlmResult:
    url = base_url or os.getenv("OLLAMA_URL", DEFAULT_OLLAMA_URL)
    payload: dict[str, Any] = {
        "model": model,
        "prompt": build_rag_prompt(question, context),
        "stream": False,
    }

    req = Request(
        url=url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urlopen(req, timeout=timeout_s) as response:
            body = response.read().decode("utf-8", errors="ignore")
    except HTTPError as exc:
        return LlmResult(ok=False, text="", error=f"HTTPError {exc.code}: {exc.reason}")
    except URLError as exc:
        return LlmResult(ok=False, text="", error=f"URLError: {exc.reason}")
    except Exception as exc:
        return LlmResult(ok=False, text="", error=f"{type(exc).__name__}: {exc}")

    try:
        parsed = json.loads(body)
    except json.JSONDecodeError:
        return LlmResult(ok=False, text="", error="Invalid JSON returned from Ollama endpoint")

    text = str(parsed.get("response", "")).strip()
    if not text:
        return LlmResult(ok=False, text="", error="Empty response from Ollama endpoint")
    return LlmResult(ok=True, text=text)


def gemini_generate(
    question: str,
    context: str,
    model: str | None = None,
    api_key: str | None = None,
    timeout_s: int = 45,
) -> LlmResult:
    key = api_key or os.getenv("GEMINI_API_KEY")
    if not key:
        return LlmResult(ok=False, text="", error="Missing GEMINI_API_KEY")

    resolved_model = model or os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL)
    url = DEFAULT_GEMINI_URL_TEMPLATE.format(model=resolved_model)
    payload: dict[str, Any] = {
        "contents": [
            {
                "parts": [
                    {
                        "text": build_rag_prompt(question, context),
                    }
                ]
            }
        ]
    }

    req = Request(
        url=url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "x-goog-api-key": key,
        },
        method="POST",
    )

    try:
        with urlopen(req, timeout=timeout_s) as response:
            body = response.read().decode("utf-8", errors="ignore")
    except HTTPError as exc:
        details = ""
        try:
            details = exc.read().decode("utf-8", errors="ignore").strip()
        except Exception:
            details = ""
        suffix = f" | {details}" if details else ""
        return LlmResult(ok=False, text="", error=f"HTTPError {exc.code}: {exc.reason}{suffix}")
    except URLError as exc:
        return LlmResult(ok=False, text="", error=f"URLError: {exc.reason}")
    except Exception as exc:
        return LlmResult(ok=False, text="", error=f"{type(exc).__name__}: {exc}")

    try:
        parsed = json.loads(body)
    except json.JSONDecodeError:
        return LlmResult(ok=False, text="", error="Invalid JSON returned from Gemini endpoint")

    candidates = parsed.get("candidates") or []
    if not candidates:
        return LlmResult(ok=False, text="", error="Gemini returned no candidates")

    content = candidates[0].get("content") or {}
    parts = content.get("parts") or []
    texts = [str(part.get("text", "")).strip() for part in parts if part.get("text")]
    text = "\n".join(item for item in texts if item).strip()
    if not text:
        return LlmResult(ok=False, text="", error="Empty response from Gemini endpoint")
    return LlmResult(ok=True, text=text)
