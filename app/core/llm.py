import re
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate


def get_llm(thinking: bool = True):
    """
    Get the LLM instance.
    thinking=True (default): Gemma outputs <think>...</think> before its answer.
    Use parse_thinking() to split the raw output into (thinking, answer).
    """
    return OllamaLLM(
        model="gemma4:e2b",
        temperature=0.7,
        num_predict=2048,
        extra_body={
            "thinking": thinking
        }
    )


def parse_thinking(raw_output: str) -> tuple[str, str]:
    """
    Split Gemma's raw output into (thinking_text, answer_text).

    Gemma with thinking enabled wraps its reasoning in <think>...</think> tags.
    Everything outside those tags is the actual answer.

    Returns:
        (thinking, answer) — either can be empty string if not present.
    """
    think_pattern = re.compile(r"<think>(.*?)</think>", re.DOTALL)
    match = think_pattern.search(raw_output)

    if match:
        thinking = match.group(1).strip()
        # Remove the <think>...</think> block from the answer
        answer = think_pattern.sub("", raw_output).strip()
    else:
        thinking = ""
        answer = raw_output.strip()

    return thinking, answer


def test_llm():
    llm = get_llm()
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful AI assistant. Answer concisely."),
        ("human", "{question}")
    ])
    chain = prompt | llm
    response = chain.invoke({"question": "What is RAG in AI systems? Answer in 3 sentences."})
    thinking, answer = parse_thinking(response)
    print(f"THINKING:\n{thinking}\n\nANSWER:\n{answer}")


if __name__ == "__main__":
    test_llm()