"""
debug_thinking_v2.py
--------------------
Run from project root:
    python debug_thinking_v2.py

Extends v1 with:
  - Ollama version check
  - Harder prompt (model skips thinking on trivial maths)
  - Raw streaming chunks from /api/chat with think=True
  - ChatOllama (LangChain) test — the right class for thinking
"""

import json
import subprocess
import sys
import requests

OLLAMA_BASE  = "http://localhost:11434"
MODEL        = "gemma4:e2b"

# Use a harder prompt — models skip thinking on "2+2"
HARD_PROMPT  = "Explain in 2 sentences why transformers outperform RNNs for long sequences."

SEP = "=" * 65

def banner(title):
    print(f"\n{SEP}")
    print(f"  {title}")
    print(SEP)

def ok(msg):   print(f"  ✅  {msg}")
def fail(msg): print(f"  ❌  {msg}")
def info(msg): print(f"  ℹ️   {msg}")


# ─────────────────────────────────────────────────────────────────
# LAYER 0 — Ollama version check
# ─────────────────────────────────────────────────────────────────
def check_ollama_version():
    banner("LAYER 0 — Ollama version  (think parameter needs >= 0.7.0)")
    try:
        result = subprocess.run(["ollama", "--version"], capture_output=True, text=True)
        version_str = (result.stdout + result.stderr).strip()
        print(f"  Ollama version: {version_str}")

        # Try to extract numeric version
        import re
        m = re.search(r"(\d+)\.(\d+)\.?(\d*)", version_str)
        if m:
            major, minor = int(m.group(1)), int(m.group(2))
            if major == 0 and minor < 7:
                fail(f"Version {major}.{minor} is BELOW 0.7.0 — 'think' parameter not supported")
                info("Fix: Download latest Ollama from https://ollama.com/download")
            else:
                ok(f"Version {major}.{minor} supports 'think' parameter")
        else:
            info("Could not parse version number from output")

        # Also check which model variant is actually installed
        print()
        result2 = subprocess.run(["ollama", "show", MODEL, "--modelfile"],
                                  capture_output=True, text=True)
        modelfile = result2.stdout
        if "think" in modelfile.lower() or "channel" in modelfile.lower():
            ok(f"{MODEL} modelfile references thinking tokens")
        else:
            info(f"{MODEL} modelfile (first 300 chars):\n      {modelfile[:300]}")

    except FileNotFoundError:
        fail("'ollama' command not found in PATH")
    except Exception as e:
        fail(f"Version check error: {e}")


# ─────────────────────────────────────────────────────────────────
# LAYER 2b — /api/chat streaming with think=True, hard prompt
# ─────────────────────────────────────────────────────────────────
def test_chat_streaming_hard():
    banner("LAYER 2b — /api/chat STREAMING with think=True  (harder prompt)")
    print(f"  Prompt: {HARD_PROMPT}\n")

    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": HARD_PROMPT}],
        "stream": True,
        "think": True,
        "options": {"temperature": 0.6, "num_predict": 600},
    }

    try:
        r = requests.post(
            f"{OLLAMA_BASE}/api/chat",
            json=payload,
            stream=True,
            timeout=120,
        )
        r.raise_for_status()

        thinking_chunks = []
        answer_chunks   = []
        raw_sample      = []   # first 5 raw JSON lines

        for i, line in enumerate(r.iter_lines()):
            if not line:
                continue
            try:
                data    = json.loads(line)
                msg     = data.get("message", {})
                content = msg.get("content", "")
                # Ollama uses message.thinking (bool) to flag thinking tokens
                is_think_token = msg.get("thinking", False)

                if i < 5:
                    raw_sample.append(f"    chunk[{i}]: {json.dumps(data)[:140]}")

                if is_think_token:
                    thinking_chunks.append(content)
                else:
                    answer_chunks.append(content)

            except json.JSONDecodeError:
                pass

        print("  First 5 raw streaming chunks:")
        for s in raw_sample:
            print(s)

        thinking_text = "".join(thinking_chunks)
        answer_text   = "".join(answer_chunks)

        print()
        if thinking_text:
            ok(f"Thinking content received! ({len(thinking_text)} chars)")
            print(f"  Preview: {repr(thinking_text[:200])}")
        else:
            fail("No thinking tokens in streaming response")
            info("message.thinking field was never True in any chunk")
            # Show whether <think> tags appeared in raw answer
            if "<think>" in answer_text:
                info("<think> tags ARE embedded in answer_text → "
                     "Ollama embeds them in content instead of using message.thinking field")
                print(f"  Raw answer snippet: {repr(answer_text[:300])}")
            else:
                info("No <think> tags in answer either — thinking fully absent from stream")

        print(f"  Answer preview: {repr(answer_text[:200])}")

    except Exception as e:
        fail(f"Streaming test error: {e}")
        import traceback; traceback.print_exc()


# ─────────────────────────────────────────────────────────────────
# LAYER 3b — ChatOllama (LangChain) with think=True
# ─────────────────────────────────────────────────────────────────
def test_chatollama():
    banner("LAYER 3b — LangChain ChatOllama  (chat endpoint, hard prompt)")
    try:
        from langchain_ollama import ChatOllama

        # ChatOllama supports think= as a direct param in newer versions
        llm = ChatOllama(
            model=MODEL,
            temperature=0.6,
            num_predict=600,
            think=True,
        )

        print(f"  Prompt: {HARD_PROMPT}\n")
        response = llm.invoke(HARD_PROMPT)

        content  = response.content
        thinking = response.additional_kwargs.get("thinking", "")

        print(f"  response.content:  {repr(content[:200])}")
        print(f"  additional_kwargs: {response.additional_kwargs}")

        if thinking:
            ok(f"ChatOllama returns thinking in additional_kwargs! ({len(thinking)} chars)")
            print(f"  Thinking preview: {repr(thinking[:200])}")
        elif "<think>" in content:
            ok("<think> tags embedded inside response.content")
            print(f"  Content snippet: {repr(content[:300])}")
        else:
            fail("ChatOllama: no thinking in additional_kwargs, no <think> in content")

    except TypeError as e:
        # Some versions don't accept think= at init time
        fail(f"ChatOllama does not accept think= param: {e}")
        info("Try: llm = ChatOllama(model=...) and pass options={'think': True}")
        try:
            from langchain_ollama import ChatOllama
            llm2 = ChatOllama(model=MODEL, temperature=0.6, num_predict=600,
                               extra_body={"think": True})
            resp2 = llm2.invoke(HARD_PROMPT)
            think2 = resp2.additional_kwargs.get("thinking", "")
            if think2:
                ok(f"extra_body={{'think':True}} works! ({len(think2)} chars)")
            else:
                fail("extra_body approach also no thinking")
        except Exception as e2:
            fail(f"Fallback also failed: {e2}")

    except ImportError:
        fail("langchain_ollama not installed")
    except Exception as e:
        fail(f"ChatOllama error: {e}")
        import traceback; traceback.print_exc()


# ─────────────────────────────────────────────────────────────────
# LAYER 5 — Direct streaming with think=True and print every chunk
# ─────────────────────────────────────────────────────────────────
def test_raw_stream_verbose():
    banner("LAYER 5 — Verbose raw stream  (ALL chunks shown, 10-chunk limit)")
    print(f"  Prompt: {HARD_PROMPT}\n")

    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": HARD_PROMPT}],
        "stream": True,
        "think": True,
        "options": {"temperature": 0.6, "num_predict": 200},
    }

    try:
        r = requests.post(f"{OLLAMA_BASE}/api/chat", json=payload,
                          stream=True, timeout=60)
        r.raise_for_status()

        print("  Every chunk (first 15):")
        for i, line in enumerate(r.iter_lines()):
            if i >= 15:
                print("  ... (truncated after 15 chunks)")
                break
            if line:
                try:
                    data = json.loads(line)
                    msg  = data.get("message", {})
                    print(f"    [{i:02d}] thinking={str(msg.get('thinking','—')):5s}  "
                          f"content={repr(msg.get('content','')[:50])}")
                except Exception:
                    print(f"    [{i:02d}] RAW: {repr(line[:80])}")

    except Exception as e:
        fail(f"Verbose stream error: {e}")


# ─────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n🔍 THINKING DEBUG v2 — VERSION + STREAMING")
    print(f"   Model:        {MODEL}")
    print(f"   Hard prompt:  {HARD_PROMPT}")

    check_ollama_version()
    test_chat_streaming_hard()
    test_chatollama()
    test_raw_stream_verbose()

    print(f"\n{SEP}")
    print("  Done. Paste full output to get the fix.")
    print(SEP + "\n")