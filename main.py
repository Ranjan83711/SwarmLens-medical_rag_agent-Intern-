import os, json
from typing import TypedDict, Any, List
from dotenv import load_dotenv
from groq import Groq
from langgraph.graph import StateGraph, END
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()

# --- State ---
class AgentState(TypedDict):
    question: str
    plan: dict
    retrieved_docs: list
    answer: str
    reflection: dict

# --- LLM Call ---
def groq_call(system: str, user: str) -> str:
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    model = "llama-3.3-70b-versatile"
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}],
        temperature=0.2,
        max_tokens=512
    )
    return resp.choices[0].message.content.strip()

# --- Retrieval Setup ---
def get_retriever():
    embeddings = HuggingFaceEmbeddings(model_name=os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2"))
    vectordb = Chroma(persist_directory=os.getenv("CHROMA_DIR", "./chroma_store"), embedding_function=embeddings)
    return vectordb.as_retriever(search_kwargs={"k": 3})

# --- Nodes ---
def plan_node(state: AgentState) -> AgentState:
    q = state["question"]
    plan = {"need_retrieval": len(q.split()) > 2, "query": q}
    print(f"[PLAN] {plan}")
    state["plan"] = plan
    return state

def retrieve_node(state: AgentState) -> AgentState:
    if not state["plan"]["need_retrieval"]:
        print("[RETRIEVE] Skipped retrieval")
        state["retrieved_docs"] = []
        return state

    retriever = get_retriever()
    docs = retriever.invoke(state["plan"]["query"])
    state["retrieved_docs"] = [{"content": d.page_content[:300]} for d in docs]
    print(f"[RETRIEVE] Retrieved {len(docs)} docs")
    return state

def answer_node(state: AgentState) -> AgentState:
    context = "\n".join([d["content"] for d in state.get("retrieved_docs", [])])
    question = state["question"]

    system = "You are a helpful medical assistant. Answer clearly using only provided context."
    user = f"Question: {question}\n\nContext:\n{context}\n\nAnswer:"
    answer = groq_call(system, user)
    print(f"[ANSWER] {answer[:120]}...")
    state["answer"] = answer
    return state

def reflect_node(state: AgentState) -> AgentState:
    question, answer = state["question"], state["answer"]
    system = "You are a strict evaluator. Score answer relevance (0–1)."
    user = f"Question: {question}\nAnswer: {answer}\nReturn JSON: {{'relevance':score,'comment':text}}"
    review = groq_call(system, user)
    try:
        reflection = json.loads(review)
    except:
        reflection = {"relevance": 0.7, "comment": review}
    print(f"[REFLECT] {reflection}")
    state["reflection"] = reflection
    return state

# --- LangGraph Build ---
def build_agent():
    graph = StateGraph(AgentState)
    graph.add_node("plan", plan_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("answer", answer_node)
    graph.add_node("reflect", reflect_node)

    graph.set_entry_point("plan")
    graph.add_edge("plan", "retrieve")
    graph.add_edge("retrieve", "answer")
    graph.add_edge("answer", "reflect")
    graph.add_edge("reflect", END)

    return graph.compile()

def run_agent(question: str) -> AgentState:
    app = build_agent()
    init: AgentState = {"question": question, "plan": {}, "retrieved_docs": [], "answer": "", "reflection": {}}
    final = app.invoke(init)
    return final

if __name__ == "__main__":
    result = run_agent("What are the symptoms of diabetes?")
    print("\n=== FINAL ANSWER ===")
    print(result["answer"])
    print("\n=== REFLECTION ===")
    print(result["reflection"])
