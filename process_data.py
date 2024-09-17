import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter

def process_and_store_data(text_file='data/financial_advice_data_sample.txt'):
    # Load your text
    with open(text_file, 'r', encoding='utf-8') as file:
        financial_text = file.read()

    # Chunk the text
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = text_splitter.split_text(financial_text)

    # Try loading the SentenceTransformer model (handle potential errors)
    try:
        embedder = SentenceTransformer('all-MiniLM-L6-v2')
    except Exception as e:
        print(f"Error loading model: {e}")
        # Consider using a default model or exiting gracefully
        return

    # Generate embeddings for each chunk
    chunk_embeddings = embedder.encode(chunks)

    # Create a FAISS index
    dimension = chunk_embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(chunk_embeddings))

    # Save FAISS index and chunks
    faiss.write_index(index, 'faiss_index.bin')
    np.save('chunks.npy', chunks)

if __name__ == "__main__":
    process_and_store_data()