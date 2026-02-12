import streamlit as st
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# Load data
embeddings = np.load("embeddings.npy")

with open("documents.txt", "r", encoding="utf-8") as f:
    documents = f.readlines()

# Cache model (important)
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

def get_query_embedding(query):
    return model.encode(query)

def retrieve_top_k(query_embedding, embeddings, documents, k=10):
    similarities = cosine_similarity(
        query_embedding.reshape(1, -1),
        embeddings
    )[0]

    top_k_indices = similarities.argsort()[-k:][::-1]

    return [(documents[i], similarities[i]) for i in top_k_indices]

# Streamlit UI
st.title("Information Retrieval using Document Embeddings")

query = st.text_input("Enter your query:")

if st.button("Search") and query.strip():
    query_embedding = get_query_embedding(query)
    results = retrieve_top_k(query_embedding, embeddings, documents)

    st.write("### Top 10 Relevant Documents:")

    for doc, score in results:
        st.write(f"- **{doc.strip()}** (Score: {score:.4f})")

