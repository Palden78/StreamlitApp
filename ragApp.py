import streamlit as st
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.api.types import Documents, Embeddings, EmbeddingFunction
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

# ------------------------
# 1. Setup
# ------------------------
load_dotenv()

CHROMA_DB_DIR = "./chroma_db_fresh"

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

class SentenceTransformerEmbeddingFunction(EmbeddingFunction):
    def __init__(self, model):
        self.model = model

    def __call__(self, texts: Documents) -> Embeddings:
        return self.model.encode(texts).tolist()

ef_function = SentenceTransformerEmbeddingFunction(embedding_model)

# Load Chroma client
client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
collection = client.get_or_create_collection(
    name="documents",
    embedding_function=ef_function
)


# ------------------------
# 2. Query function
# ------------------------
PROMPT_TEMPLATE = """
Answer the question based only on the following context: If you don't know the answer, please say I dont have enough information to answer that.
{context}
 - -
Answer the question based on the above context: {question}
"""

def query_rag(query_text):
  # Retrieving the context from the DB using similarity search
  collection=client.get_collection("documents",ef_function)
  results=collection.query(query_texts=[query_text],n_results=3,include=['distances','documents','metadatas'])
  print(results)
  print("_______")
  docs = results['documents'][0]  # list of top chunks
  print(docs)
  metas = results['metadatas'][0]
  combined = []
  for d, m in zip(docs, metas):
    combined.append(f"Source: {m.get('source','unknown')} (page {m.get('page')})\n{d}")
  if not docs:
    return "I couldn’t find anything relevant in the database."

  context_text = "\n\n".join(docs)
  print("************")
  print(context_text)
  print("************")
  prompt_template=ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
  prompt=prompt_template.format(context=context_text,question=query_text)
  model=ChatOpenAI(
      model="openai/gpt-4o-mini",
      openai_api_base="https://openrouter.ai/api/v1",
      openai_api_key="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",
      temperature=0.1
  )
  response=model.invoke(prompt)
  response_text=response.content
  print(f"Response: {response_text}")
  return response_text


# ------------------------
# 3. Streamlit UI
# ------------------------
st.set_page_config(page_title="RAG Chatbot", page_icon="🤖")
st.title("🤖 RAG Chatbot (Streamlit)")

# Initialize session state for messages
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! Ask me anything from your documents."}]

# Display past messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
if prompt := st.chat_input("Ask me something..."):
    # Add user message
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Get RAG response
    with st.chat_message("assistant"):
        response = query_rag(prompt)
        st.markdown(response)

    # Save assistant response
    st.session_state.messages.append({"role": "assistant", "content": response})
