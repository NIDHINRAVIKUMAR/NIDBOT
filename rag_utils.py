import os
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle
from transformers import T5ForConditionalGeneration, T5Tokenizer

# Load models
embedder = SentenceTransformer("all-MiniLM-L6-v2")
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base")
tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")

INDEX_PATH = "vector_store/index.faiss"
CHUNKS_PATH = "vector_store/chunks.pkl"

def load_index():
    index = faiss.read_index(INDEX_PATH)
    with open(CHUNKS_PATH, "rb") as f:
        chunks = pickle.load(f)
    return index, chunks

def get_answer_from_query(query: str) -> str:
    # Check if index and chunks exist
    if not os.path.exists(INDEX_PATH):
        return "No index found. Please upload a PDF first."
    if not os.path.exists(CHUNKS_PATH):
        return "No chunks found. Please upload a PDF first."

    # Load the index and chunks
    index, chunks = load_index()

    # Generate the query embedding
    query_embedding = embedder.encode([query])

    # Perform the search in the index
    D, I = index.search(np.array(query_embedding), k=3)

    # Extract context from the chunks
    try:
        context = " ".join([chunks[i]["text"] for i in I[0]])  # Assumes chunks are dictionaries
    except TypeError:
        context = " ".join([chunks[i] for i in I[0]])  # If chunks are just text

    # Prepare the prompt for the language model
    prompt = f"Answer the question based on the context: {context}\nQuestion: {query}"

    # Tokenize the prompt and generate the answer using T5 model
    inputs = tokenizer.encode(prompt, return_tensors="pt", max_length=512, truncation=True)
    outputs = model.generate(inputs, max_length=200, num_beams=4, early_stopping=True)

    # Decode the answer
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return answer


