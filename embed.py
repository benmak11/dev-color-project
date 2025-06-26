import faiss
import numpy as np
import pickle
import os
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.text_splitter import MarkdownHeaderTextSplitter
from config import OPENAI_API_KEY, EMBEDDING_MODEL, CHUNK_SIZE, CHUNK_OVERLAP

def load_knowledge(file_path):
    """Load text from knowledge file"""
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

def create_semantic_chunks(text):
    """Create semantically meaningful chunks using markdown headers"""
    # First split by markdown headers
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]
    
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    md_splits = markdown_splitter.split_text(text)
    
    # Then split large chunks further using recursive character splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    
    chunks = []
    for split in md_splits:
        if len(split.page_content) > CHUNK_SIZE:
            sub_chunks = text_splitter.split_text(split.page_content)
            chunks.extend(sub_chunks)
        else:
            chunks.append(split.page_content)
    
    return chunks

def create_faiss_index(text):
    """Convert text into embeddings and store in FAISS with optimizations"""
    embeddings = OpenAIEmbeddings(
        openai_api_key=OPENAI_API_KEY, 
        model=EMBEDDING_MODEL,
        chunk_size=1000  # Batch size for embedding requests
    )
    
    # Create semantic chunks
    chunks = create_semantic_chunks(text)
    print(f"Created {len(chunks)} semantic chunks")
    
    # Generate embeddings in batches
    print("Generating embeddings...")
    vectors = embeddings.embed_documents(chunks)
    
    # Create optimized FAISS index
    dim = len(vectors[0])
    index = faiss.IndexFlatIP(dim)  # Inner product for cosine similarity
    
    # Normalize vectors for cosine similarity
    vectors_normalized = np.array(vectors).astype('float32')
    faiss.normalize_L2(vectors_normalized)
    
    index.add(vectors_normalized)
    
    return index, chunks

if __name__ == "__main__":
    text = load_knowledge("data/devcolorfaq.txt")
    index, chunks = create_faiss_index(text)
    
    # Save index and chunks
    faiss.write_index(index, "faiss_index.bin")
    
    # Save chunks with metadata
    chunks_data = {
        'chunks': chunks,
        'metadata': {
            'embedding_model': EMBEDDING_MODEL,
            'chunk_size': CHUNK_SIZE,
            'chunk_overlap': CHUNK_OVERLAP,
            'total_chunks': len(chunks)
        }
    }
    
    with open("data/chunks.pkl", "wb") as f:
        pickle.dump(chunks_data, f)
    
    # Also save as text for backward compatibility
    with open("data/chunks.txt", "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(chunk + "\n=====\n")

    print(f"✅ FAISS index and {len(chunks)} text chunks saved.")
    print(f"📊 Index dimension: {index.d}")
    print(f"🔍 Index size: {index.ntotal} vectors")
