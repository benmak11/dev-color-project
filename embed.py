import faiss
import numpy as np
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from config import OPENAI_API_KEY, EMBEDDING_MODEL

def load_knowledge(file_path):
    """Load text from knowledge file"""
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

def create_faiss_index(text):
    """Convert text into embeddings and store in FAISS"""
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY, model=EMBEDDING_MODEL)
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    
    chunks = splitter.split_text(text)
    vectors = embeddings.embed_documents(chunks)
    
    dim = len(vectors[0])
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(vectors).astype('float32'))
    
    return index, chunks

if __name__ == "__main__":
    text = load_knowledge("data/devcolorfaq.txt")
    index, chunks = create_faiss_index(text)
    
    faiss.write_index(index, "faiss_index.bin")
    with open("data/chunks.txt", "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(chunk + "\n=====\n")

    print("✅ FAISS index and text chunks saved.")
