import streamlit as st
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from pathlib import Path

st.set_page_config(page_title="Information Retrieval App", page_icon="🔎", layout="centered")

BASE_DIR = Path(__file__).resolve().parent

@st.cache_data
def load_data():
    embeddings = np.load(BASE_DIR / "embeddings.npy")
    with open(BASE_DIR / "documents.txt", "r", encoding="utf-8") as f:
        documents = [line.strip() for line in f if line.strip()]
    return embeddings, documents

@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

embeddings, documents = load_data()
model = load_model()

# Normalize document embeddings for better cosine similarity behavior
embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

def get_query_embedding(query: str):
    v = model.encode(query)
    return v / np.linalg.norm(v)

def retrieve_top_k(query_embedding, embeddings, documents, k=10):
    similarities = cosine_similarity(query_embedding.reshape(1, -1), embeddings)[0]
    top_k_indices = similarities.argsort()[-k:][::-1]
    return [(documents[i], float(similarities[i])) for i in top_k_indices]

# Sidebar
st.sidebar.header("About")
st.sidebar.write("Semantic Information Retrieval using embeddings + cosine similarity.")
st.sidebar.write("Model: SentenceTransformers (all-MiniLM-L6-v2)")
st.sidebar.write("Tip: Adjust Top-K and minimum similarity for better results.")

# Main UI
st.title("🔎 Information Retrieval using Document Embeddings")

query = st.text_input("Enter your query:")

k = st.slider("Top K results", 1, 20, 10)
min_score = st.slider("Minimum similarity", 0.0, 1.0, 0.25)

if st.button("Search"):
    if not query.strip():
        st.warning("Please enter a query.")
        st.stop()

    query_embedding = get_query_embedding(query)
    results = retrieve_top_k(query_embedding, embeddings, documents, k=k)

    st.write(f"### Results (Top {k})")

    shown = 0
    for rank, (doc, score) in enumerate(results, start=1):
        if score < min_score:
            continue
        shown += 1
        st.write(f"**{rank}. Score: {score:.4f}**")
        st.progress(min(score, 1.0))
        with st.expander("Show document"):
            st.write(doc)

    if shown == 0:
        st.warning("No documents matched your minimum similarity. Try lowering the threshold or rephrasing your query.")


