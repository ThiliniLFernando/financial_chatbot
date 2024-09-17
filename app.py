import streamlit as st
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# Load the FAISS index and chunks
index = faiss.read_index('faiss_index.bin')
chunks = np.load('chunks.npy', allow_pickle=True)
embedder = SentenceTransformer('all-MiniLM-L6-v2')  # Consider alternative model if needed

def search_faiss(query, top_k=5):
    query_embedding = embedder.encode([query])
    distances, indices = index.search(np.array(query_embedding), top_k)
    results = [chunks[i] for i in indices[0]]
    return results, distances

# Streamlit UI
st.title("Financial Advice Chatbot")

query = st.text_input("Enter your query about financial advice:")

# Add a button to trigger search
search_button = st.button("Search")

if search_button or query:  # Run search on button click or enter key press
    with st.spinner('Searching...'):
        results, distances = search_faiss(query)
        st.write(f"**Results for query:** {query}")
        for i, (result, distance) in enumerate(zip(results, distances[0])):
            st.write(f"**Result {i+1} (Distance: {distance:.2f}):**")
            st.write(result)