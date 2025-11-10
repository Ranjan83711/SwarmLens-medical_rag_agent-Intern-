---

# 🧠 Medical RAG Agent

This project is a simple **AI agent** built using **LangGraph** that answers medical questions from a given PDF document.
It uses the **RAG (Retrieval-Augmented Generation)** approach to search the PDF for relevant content and generate accurate answers.
A short reflection step is also added so the agent can check whether its answer is relevant to the question.

---

## Streamlit App: https://sducjqmuhqvrgcuhtmuobj.streamlit.app/

## ⚙️ Tech Stack

* **Python 3.13**
* **LangGraph** for agent workflow
* **Groq API** as LLM
* **ChromaDB** for local vector database
* **Hugging Face** for embeddings
* **LangSmith** for tracing
* **Streamlit** for the user interface

---

## 🧩 Project Flow

The workflow is divided into four simple steps (LangGraph nodes):

1. **plan** → Understand the user’s question and decide if retrieval is needed.
2. **retrieve** → Get the most relevant chunks from the vector database.
3. **answer** → Use the LLM to generate an answer from the retrieved content.
4. **reflect** → Check if the generated answer is relevant and complete.

Each step prints logs in the terminal, showing the progress:

---

## 🖥️ Environment Setup

### 1. Clone the project

```bash
git clone https://github.com/yourusername/medical-rag-agent.git
cd medical-rag-agent
```

### 2. Create and activate a virtual environment (Python 3.13)

```
conda create -n langgraph python=3.13 -y
conda activate langgraph
```

### 3. Install the requirements

```bash
pip install -r requirements.txt
```

### 4. Set up the environment file

Created a file named `.env` in the root folder and add the required keys:

```
GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=llama-3.3-70b-versatile
LANGSMITH_API_KEY=your_langsmith_api_key
LANGSMITH_TRACING=true
LANGSMITH_PROJECT=Medical-RAG-Agent
EMBED_MODEL=sentence-transformers/all-MiniLM-L6-v2
CHROMA_DIR=./chroma_store
```

---

## 📘 How to Use

### Step 1 — Add the PDF or .txt file

Place the file in the main project folder

### Step 2 — Ingest the data

Run the ingestion script to process and store embeddings:

```bash
python ingest.py
```

This creates a local folder `chroma_store/` that stores all vector data.

### Step 3 — Run the agent (terminal)

```bash
python main.py
```

Example output:

```
[PLAN] {'need_retrieval': True, 'query': 'What are the symptoms of diabetes?'}
[RETRIEVE] Retrieved 3 docs
[ANSWER] Diabetes symptoms include frequent urination, increased thirst...
[REFLECT] {'relevance': 0.9, 'comment': 'Answer is relevant and complete.'}
```

### Step 4 — Run with Streamlit UI

```bash
streamlit run app.py
```

Then open the browser at:

```
http://localhost:8501
```

We can ask the questions and view the step-by-step process and reflection.

---

## 🧠 Approach

1. **PDF Ingestion** – The document is loaded, split into text chunks, and stored as vector embeddings in ChromaDB using Hugging Face.
2. **Question Understanding (Plan)** – The agent checks if the question requires retrieval.
3. **Retrieval (RAG)** – The relevant chunks are fetched from Chroma based on the question.
4. **Answer Generation** – The Groq model generates a response using the retrieved context.
5. **Reflection** – The model evaluates its own answer for relevance and completeness.
6. **UI** – Streamlit provides an easy interface to interact with the system.

This approach makes the agent explainable and traceable, with clear step outputs and validation.

---

## 📂 Project Files

| File                      | Description                                                          |
| ------------------------- | -------------------------------------------------------------------- |
| `ingest.py`               | Loads and processes the PDF into ChromaDB                            |
| `main.py`                 | Contains the LangGraph workflow (plan → retrieve → answer → reflect) |
| `app.py`                  | Streamlit app for user interface                                     |
| `medical_rag_agent.ipynb` | Notebook version for submission                                      |
| `requirements.txt`        | Required Python packages                                             |
| `.env`                    | Environment configuration file                                       |

---

## 🧾 Example Output

```
[PLAN] {'need_retrieval': True, 'query': 'What is hypertension?'}
[RETRIEVE] Retrieved 2 docs
[ANSWER] Hypertension, or high blood pressure, is a chronic condition...
[REFLECT] {'relevance': 0.95, 'comment': 'The answer directly addresses the question.'}
```

---

## 👤 Author

**Ranjan Kumar Yadav**
📧 Email: [ranjan83711yadav@gmail.com](mailto:ranjan83711yadav@gmail.com)
🔗 [LinkedIn](https://www.linkedin.com/in/ranjan-kumar-yadav-05b62a231/)
💻 [GitHub](https://github.com/ranjan83711yadav)

---

