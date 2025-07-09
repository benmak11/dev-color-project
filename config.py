import os

OPENAI_API_KEY = "your_api_key_here" # OpenAI API key
EMBEDDING_MODEL = "text-embedding-3-small" # Updated to newer, faster model
GPT_MODEL = "gpt-4o-mini" # More cost-effective model for RAG
SIMILARITY_THRESHOLD = 0.7 # Adjust this threshold based on your use case
TOP_K_RESULTS = 3 # Number of top results to retrieve for better context
CHUNK_SIZE = 500 # Smaller chunks for more precise retrieval
CHUNK_OVERLAP = 50 # Overlap between chunks
CACHE_EMBEDDINGS = True # Enable embedding caching
