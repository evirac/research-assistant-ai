"""
llm.py — Ollama client that correctly captures thinking tokens.

Root cause of the original bug:
  OllamaLLM uses /api/generate which does NOT support thinking.
  Thinking only works through /api/chat where Ollama returns each
  token tagged with message.thinking (the actual text, not a bool).

This module replaces OllamaLLM with a thin direct-HTTP wrapper so
thinking tokens are always captured and never silently dropped.
"""

import json
import requests
from typing import Generator, Tuple

OLLAMA_BASE = "http://localhost:11434"
MODEL       = "gemma4:e2b"

# Shared options used for every call
_OPTIONS = {
    "temperature": 0.7,
    "num_predict": 2048,
}


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
        "model":    MODEL,
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
        "model":    MODEL,
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
    print("Testing ollama_chat (non-streaming)…")
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