# index.py
import os
import chromadb
from pypdf import PdfReader
# We no longer need to import SentenceTransformer directly here
from chromadb.utils import embedding_functions

# --- Configuration ---
PDF_PATH = "knowledge_base/clean_code_book.pdf"
COLLECTION_NAME = "clean_code_book"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

def main():
    print("Starting the indexing process...")

    # 1. Extract Text from PDF
    print(f"Loading and extracting text from {PDF_PATH}...")
    if not os.path.exists(PDF_PATH):
        print(f"Error: The file {PDF_PATH} was not found.")
        return
        
    reader = PdfReader(PDF_PATH)
    full_text = "".join(page.extract_text() for page in reader.pages)
    print(f"Successfully extracted {len(full_text)} characters.")

    # 2. Chunk the Text
    print("Splitting text into manageable chunks...")
    chunks = []
    for i in range(0, len(full_text), CHUNK_SIZE - CHUNK_OVERLAP):
        chunks.append(full_text[i:i + CHUNK_SIZE])
    print(f"Created {len(chunks)} chunks.")

    # 3. Setup ChromaDB and Embedding Function
    print("Initializing ChromaDB and setting up the embedding function...")
    client = chromadb.PersistentClient(path="db")
    
    # --- THIS IS THE KEY CHANGE ---
    # Use the official SentenceTransformerEmbeddingFunction helper from ChromaDB
    sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL_NAME
    )

    # Now, pass this object to the collection
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=sentence_transformer_ef # Pass the class instance here
    )

    # 4. Index the Chunks
    print("Adding document chunks to the collection. This may take a while...")
    chunk_ids = [str(i) for i in range(len(chunks))]
    
    batch_size = 100
    for i in range(0, len(chunks), batch_size):
        batch_chunks = chunks[i:i + batch_size]
        batch_ids = chunk_ids[i:i + batch_size]
        collection.add(
            documents=batch_chunks,
            ids=batch_ids
        )
        print(f"  - Added batch {i//batch_size + 1}/{(len(chunks)-1)//batch_size + 1}")

    print("\n✅ Indexing complete!")
    print(f"Total documents in collection '{COLLECTION_NAME}': {collection.count()}")

if __name__ == "__main__":
    main()
