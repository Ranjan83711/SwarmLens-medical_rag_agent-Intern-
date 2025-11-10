import streamlit as st
from main import run_agent

st.set_page_config(page_title="Medical RAG Agent", page_icon="🧠")
st.title("🧠 Medical RAG Agent (LangGraph + Groq + Chroma)")
st.caption("Plan → Retrieve → Answer → Reflect | With LangSmith Tracing")

question = st.text_input("Ask a medical question:", "What are the causes of hypertension?")
if st.button("Run Agent"):
    with st.spinner("Thinking..."):
        result = run_agent(question)

    st.subheader("📝 Plan")
    st.json(result["plan"])

    st.subheader("📚 Retrieved Documents")
    for d in result["retrieved_docs"]:
        st.write(d["content"])

    st.subheader("💡 Answer")
    st.write(result["answer"])

    st.subheader("🔍 Reflection / Validation")
    st.json(result["reflection"])
