import faiss
import numpy as np
import openai
import pickle
import time
from functools import lru_cache
from langchain_openai import OpenAIEmbeddings
from config import (
    OPENAI_API_KEY, GPT_MODEL, EMBEDDING_MODEL, 
    SIMILARITY_THRESHOLD, TOP_K_RESULTS, CACHE_EMBEDDINGS
)

# Initialize OpenAI client
client = openai.OpenAI(api_key=OPENAI_API_KEY)

class OptimizedRAGSystem:
    def __init__(self):
        self.index = None
        self.chunks = None
        self.embeddings = None
        self.load_data()
    
    def load_data(self):
        """Load FAISS index and text chunks with error handling."""
        try:
            # Try to load optimized pickle format first
            with open("data/chunks.pkl", "rb") as f:
                chunks_data = pickle.load(f)
                self.chunks = chunks_data['chunks']
                print(f"📚 Loaded {len(self.chunks)} chunks from optimized format")
        except FileNotFoundError:
            # Fallback to text format
            with open("data/chunks.txt", "r", encoding="utf-8") as f:
                self.chunks = f.read().split("=====\n")
            print(f"📚 Loaded {len(self.chunks)} chunks from text format")
        
        self.index = faiss.read_index("faiss_index.bin")
        self.embeddings = OpenAIEmbeddings(
            openai_api_key=OPENAI_API_KEY, 
            model=EMBEDDING_MODEL
        )
        print(f"🔍 Index loaded with {self.index.ntotal} vectors")
    
    @lru_cache(maxsize=1000) if CACHE_EMBEDDINGS else lambda x: x
    def get_query_embedding(self, query):
        """Get embedding for query with optional caching."""
        return self.embeddings.embed_query(query)
    
    def get_top_matches(self, query, top_k=None):
        """Retrieve multiple top matches for better context."""
        if top_k is None:
            top_k = TOP_K_RESULTS
            
        query_vector = np.array([self.get_query_embedding(query)]).astype('float32')
        faiss.normalize_L2(query_vector)  # Normalize for cosine similarity
        
        # Get top k matches
        similarities, indices = self.index.search(query_vector, top_k)
        
        results = []
        for i, (similarity, idx) in enumerate(zip(similarities[0], indices[0])):
            if similarity > SIMILARITY_THRESHOLD:
                results.append({
                    'chunk': self.chunks[idx],
                    'similarity': similarity,
                    'rank': i + 1
                })
        
        return results
    
    def generate_response(self, query, context_results):
        """Generate response using multiple context chunks."""
        if not context_results:
            return "I don't have enough information to answer that based on my knowledge base."
        
        # Combine multiple chunks for better context
        context_text = "\n\n".join([result['chunk'] for result in context_results])
        
        # Create a more detailed prompt
        prompt = f"""Based on the following context, provide a comprehensive and accurate answer to the user's question. If the context doesn't contain enough information, say so.

Context:
{context_text}

Question: {query}

Answer:"""
        
        try:
            response = client.chat.completions.create(
                model=GPT_MODEL,
                messages=[
                    {
                        "role": "system", 
                        "content": "You are a helpful AI assistant. Answer questions based only on the provided context. Be concise but thorough."
                    },
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.3  # Lower temperature for more consistent responses
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    def process_query(self, query):
        """Process a single query with performance monitoring."""
        start_time = time.time()
        
        # Get relevant context
        context_results = self.get_top_matches(query)
        
        # Generate response
        response = self.generate_response(query, context_results)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        return {
            'response': response,
            'context_count': len(context_results),
            'processing_time': processing_time,
            'context_results': context_results
        }

def main():
    """Main interactive loop with performance monitoring."""
    print("🚀 Initializing optimized RAG system...")
    rag_system = OptimizedRAGSystem()
    print("✅ System ready!\n")
    
    # Performance statistics
    total_queries = 0
    total_time = 0
    
    while True:
        query = input("\n🤔 Enter your question (or 'exit' to quit, 'stats' for performance): ")
        
        if query.lower() == "exit":
            break
        elif query.lower() == "stats":
            if total_queries > 0:
                avg_time = total_time / total_queries
                print(f"\n📊 Performance Statistics:")
                print(f"   Total queries: {total_queries}")
                print(f"   Average response time: {avg_time:.2f}s")
                print(f"   Total processing time: {total_time:.2f}s")
            continue
        
        if not query.strip():
            continue
        
        # Process query
        result = rag_system.process_query(query)
        
        # Update statistics
        total_queries += 1
        total_time += result['processing_time']
        
        # Display results
        print(f"\n💡 AI Response: {result['response']}")
        print(f"⏱️  Response time: {result['processing_time']:.2f}s")
        print(f"📄 Context chunks used: {result['context_count']}")
        
        if result['context_results']:
            print(f"🎯 Best match similarity: {result['context_results'][0]['similarity']:.3f}")

if __name__ == "__main__":
    main()
