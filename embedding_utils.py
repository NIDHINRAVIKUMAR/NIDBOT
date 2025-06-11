import faiss
import numpy as np
import pickle
import os
from sentence_transformers import SentenceTransformer

def create_and_save_index(chunks):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(chunks)

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))

    # Save index
    os.makedirs("vector_store", exist_ok=True)
    faiss.write_index(index, "vector_store/index.faiss")

    # Save chunks
    with open("vector_store/chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)





