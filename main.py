import faiss
import numpy as np
import openai
from langchain_openai import OpenAIEmbeddings
from config import OPENAI_API_KEY, GPT_MODEL, EMBEDDING_MODEL

# Initialize OpenAI client
client = openai.OpenAI(api_key=OPENAI_API_KEY)

def load_faiss_index():
    """Load FAISS index and text chunks."""
    index = faiss.read_index("faiss_index.bin")
    with open("data/chunks.txt", "r", encoding="utf-8") as f:
        chunks = f.read().split("=====\n")
    return index, chunks

def get_top_match(query, index, chunks):
    """Retrieve the closest match for the query."""
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY, model=EMBEDDING_MODEL)
    query_vector = np.array([embeddings.embed_query(query)]).astype('float32')
    
    # Get the top match
    _, idx = index.search(query_vector, 1)
    return chunks[idx[0][0]]

def generate_response(query, context):
    """Generate a response using OpenAI's GPT model."""
    prompt = f"Context: {context}\n\nUser Query: {query}\n\n Response:"
    
    response = client.chat.completions.create(
        model = GPT_MODEL,
        messages = [
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message.content

if __name__ == "__main__":
    index, chunks = load_faiss_index()
    
    while True:
        query = input("\nEnter your question (or 'exit' to quit): ")
        if query.lower() == "exit":
            break
        
        context = get_top_match(query, index, chunks)
        response = generate_response(query, context)
        print("\n💡 AI Response:", response)
