from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

def get_llm(thinking: bool = False):
    return OllamaLLM(
        model="gemma4:e2b",
        temperature=0.7,
        num_predict=1024,
        # Disable thinking by default for faster RAG responses
        extra_body={
            "thinking": thinking
        }
    )

def test_llm():
    llm = get_llm()
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful AI assistant. Answer concisely."),
        ("human", "{question}")
    ])
    chain = prompt | llm
    response = chain.invoke({"question": "What is RAG in AI systems? Answer in 3 sentences."})
    print(response)

if __name__ == "__main__":
    test_llm()