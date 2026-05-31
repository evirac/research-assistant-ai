"""
llm.py — Ollama client that correctly captures thinking tokens.

Root cause of the original bug:
  OllamaLLM uses /api/generate which does NOT support thinking.
  Thinking only works through /api/chat where Ollama returns each
  token tagged with message.thinking (the actual text, not a bool).

This module replaces OllamaLLM with a thin direct-HTTP wrapper so
thinking tokens are always captured and never silently dropped.

Model management:
  Use set_model() / get_model() to switch models at runtime.
  list_available_models() queries Ollama for all pulled models.
"""

import json
import requests
from typing import Generator, List, Tuple

OLLAMA_BASE = "http://localhost:11434"

# ── Dynamic model state ───────────────────────────────────────────────────────
_current_model = "gemma4:e2b"   # default; overridden by set_model()

# Shared options used for every call
_OPTIONS = {
    "temperature": 0.7,
    "num_predict": 2048,
}


# ─────────────────────────────────────────────────────────────────
# Model management helpers
# ─────────────────────────────────────────────────────────────────

def get_model() -> str:
    """Return the currently active Ollama model name."""
    return _current_model


def set_model(model_name: str) -> None:
    """
    Switch the active model for all subsequent calls.

    Args:
        model_name: Exact Ollama model tag, e.g. "gemma4:12b" or "llama3:8b"
    """
    global _current_model
    _current_model = model_name
    print(f"[llm] Model switched to: {model_name}")


def list_available_models() -> List[str]:
    """
    Query Ollama for all locally pulled models.

    Returns a sorted list of model tags (e.g. ["gemma4:12b", "llama3:8b"]).
    Falls back to [current_model] if Ollama is unreachable.
    """
    try:
        resp = requests.get(f"{OLLAMA_BASE}/api/tags", timeout=5)
        resp.raise_for_status()
        models = resp.json().get("models", [])
        tags = sorted(m["name"] for m in models)
        return tags if tags else [_current_model]
    except Exception as e:
        print(f"[llm] Could not list models from Ollama: {e}")
        return [_current_model]


def get_model_info() -> dict:
    """
    Return info about the currently active model.
    Tries to get parameter size from Ollama model details.
    """
    try:
        resp = requests.post(
            f"{OLLAMA_BASE}/api/show",
            json={"name": _current_model},
            timeout=5,
        )
        resp.raise_for_status()
        data = resp.json()
        details = data.get("details", {})
        return {
            "name": _current_model,
            "parameter_size": details.get("parameter_size", "unknown"),
            "quantization_level": details.get("quantization_level", "unknown"),
            "family": details.get("family", "unknown"),
        }
    except Exception:
        return {"name": _current_model, "parameter_size": "unknown"}


# ─────────────────────────────────────────────────────────────────
# One-shot call  (used by ask_with_memory / ask_without_memory)
# ─────────────────────────────────────────────────────────────────

def ollama_chat(system_prompt: str, user_message: str) -> Tuple[str, str]:
    """
    Call Ollama /api/chat (non-streaming) with think=True.

    Returns:
        (thinking, answer)  — thinking may be empty string if the
        model decided not to reason (very short / trivial queries).
    """
    payload = {
        "model":    _current_model,
        "messages": [
            {"role": "system",  "content": system_prompt},
            {"role": "user",    "content": user_message},
        ],
        "stream":  False,
        "think":   True,
        "options": _OPTIONS,
    }

    resp = requests.post(
        f"{OLLAMA_BASE}/api/chat",
        json=payload,
        timeout=300,
    )
    resp.raise_for_status()

    data     = resp.json()
    msg      = data.get("message", {})
    answer   = msg.get("content",  "").strip()
    thinking = msg.get("thinking", "").strip()

    return thinking, answer


# ─────────────────────────────────────────────────────────────────
# Streaming call  (used by /ask/stream in routes.py)
# ─────────────────────────────────────────────────────────────────

def ollama_chat_stream(
    system_prompt: str,
    user_message:  str,
) -> Generator[Tuple[str, str], None, None]:
    """
    Stream tokens from Ollama /api/chat with think=True.

    Yields (token_type, text) pairs where token_type is:
        "thinking"  — a reasoning token (goes into the thinking block)
        "answer"    — a normal response token (shown to the user)

    The caller can present thinking tokens in a collapsible block
    while simultaneously streaming answer tokens into the chat.
    """
    payload = {
        "model":    _current_model,
        "messages": [
            {"role": "system",  "content": system_prompt},
            {"role": "user",    "content": user_message},
        ],
        "stream":  True,
        "think":   True,
        "options": _OPTIONS,
    }

    with requests.post(
        f"{OLLAMA_BASE}/api/chat",
        json=payload,
        stream=True,
        timeout=300,
    ) as resp:
        resp.raise_for_status()
        for raw_line in resp.iter_lines():
            if not raw_line:
                continue
            try:
                data    = json.loads(raw_line)
                msg     = data.get("message", {})
                content = msg.get("content",  "")
                # Ollama sets message.thinking to the token text (not a bool)
                # when the model is in reasoning mode; empty string otherwise.
                thinking_tok = msg.get("thinking", "")

                if thinking_tok:
                    yield "thinking", thinking_tok
                elif content:
                    yield "answer", content

            except (json.JSONDecodeError, KeyError):
                continue


# ─────────────────────────────────────────────────────────────────
# Backward-compat shim — parse_thinking still used in a few places
# ─────────────────────────────────────────────────────────────────

def parse_thinking(raw_output: str):
    """
    Legacy helper kept for any code that still builds a single string
    containing <think>…</think>.  With the new client this is rarely
    called, but it's here so old imports don't break.
    """
    import re
    pattern = re.compile(r"<think>(.*?)</think>", re.DOTALL)
    match = pattern.search(raw_output)
    if match:
        thinking = match.group(1).strip()
        answer   = pattern.sub("", raw_output).strip()
    else:
        thinking = ""
        answer   = raw_output.strip()
    return thinking, answer


# ─────────────────────────────────────────────────────────────────
# Quick self-test
# ─────────────────────────────────────────────────────────────────

def _test():
    print(f"Available models: {list_available_models()}")
    print(f"Current model: {get_model()}")
    print(f"Model info: {get_model_info()}")

    print("\nTesting ollama_chat (non-streaming)…")
    thinking, answer = ollama_chat(
        system_prompt="You are a helpful AI assistant. Answer concisely.",
        user_message="Explain in 2 sentences why transformers outperform RNNs.",
    )
    print(f"\nTHINKING ({len(thinking)} chars):\n{thinking[:400]}")
    print(f"\nANSWER:\n{answer}")

    print("\n\nTesting ollama_chat_stream…")
    think_buf  = []
    answer_buf = []
    for tok_type, text in ollama_chat_stream(
        system_prompt="You are a helpful AI assistant. Answer concisely.",
        user_message="Why is attention O(n²)?",
    ):
        if tok_type == "thinking":
            think_buf.append(text)
        else:
            answer_buf.append(text)
            print(text, end="", flush=True)

    print(f"\n\nThinking captured: {len(''.join(think_buf))} chars")
    print(f"Answer captured  : {len(''.join(answer_buf))} chars")


if __name__ == "__main__":
    _test()