import numpy as np
from sentence_transformers import SentenceTransformer

# Load documents
with open("documents.txt", "r", encoding="utf-8") as f:
    documents = [line.strip() for line in f if line.strip()]

model = SentenceTransformer("all-MiniLM-L6-v2")

embeddings = model.encode(documents)

np.save("embeddings.npy", embeddings)

print("Embeddings created:", embeddings.shape)

