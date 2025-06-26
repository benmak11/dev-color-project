# Optimized RAG System for /dev/color Chat Bot

Welcome!

This is an **optimized** Retrieval-Augmented Generation (RAG) system that takes user prompts related to [/dev/color](https://devcolor.org/), searches a local knowledge bank for relevant context, and generates responses based on that context with enhanced performance and accuracy.

## 🚀 Performance Improvements

This version includes several key optimizations:

- **Modern Embedding Model**: Uses `text-embedding-3-small` (faster and more accurate)
- **Semantic Chunking**: Intelligent text splitting using markdown headers
- **Query Caching**: LRU cache for repeated queries (configurable)
- **Multi-Context Retrieval**: Retrieves multiple relevant chunks for better responses
- **Performance Monitoring**: Built-in benchmarking and statistics
- **Optimized FAISS Index**: Uses cosine similarity for better matching
- **Cost Optimization**: Uses `gpt-4o-mini` for more cost-effective responses

## Technical Aspects

The project uses **FAISS** for similarity search, **LangChain** for query processing, and **OpenAI's GPT model** for generating responses. The language used is Python so hopefully its easy to read and follow along!

## Setup

### Requirements

1. **Python 3.8+** (installed on your local system)
2. Install the following dependencies:

```sh
pip install -r requirements.txt
```

3. Have a valid [OpenAI](https://platform.openai.com/) development account that will allow you to generate an `api_key` that is required to run the LLM. Please follow these [instructions](https://medium.com/@lorenzozar/how-to-get-your-own-openai-api-key-f4d44e60c327) for setting up the account if you don't have one.

### Project Structure

```
dev-color-project/
|--- data/
|    |-- devcolorfaq.txt
|    |-- chunks.pkl (optimized format)
|    |-- chunks.txt (legacy format)
|--- check_models.py
|--- config.py
|--- embed.py
|--- main.py
|--- benchmark.py (NEW!)
|--- requirements.txt (NEW!)
|--- README.md
|--- faiss_index.bin
```

#### Code Walkthrough

- **check_models**: This will verify the OpenAI models available to you based on the supplied `api_key` in the config.py
- **config.py**: Configuration file with performance settings and API keys
- **embed.py**: Creates optimized FAISS embeddings with semantic chunking
- **main.py**: Enhanced query processing with caching and performance monitoring
- **benchmark.py**: Performance benchmarking tool (NEW!)

### Usage

1. **Verify that the knowledge file is present and accessible** i.e. `devcolorfaq.txt`

2. **Generate the embeddings with the following**:

```sh
python embed.py
```

  * This creates the optimized FAISS index and stores text chunks in both pickle and text formats

3. **Run the chatbot**

```sh
python main.py
```

  - Type your question, and the system retrieves relevant knowledge
  - Type `stats` to see performance statistics
  - Type `exit` to quit

4. **Benchmark the system** (optional)

```sh
python benchmark.py
```

  * This runs a comprehensive performance test with 10 sample queries

## Performance Features

### Configuration Options

Edit `config.py` to adjust performance settings:

```python
EMBEDDING_MODEL = "text-embedding-3-small"  # Fast, accurate embeddings
GPT_MODEL = "gpt-4o-mini"                   # Cost-effective responses
SIMILARITY_THRESHOLD = 0.7                  # Similarity threshold
TOP_K_RESULTS = 3                          # Number of context chunks
CHUNK_SIZE = 500                           # Chunk size for splitting
CHUNK_OVERLAP = 50                         # Overlap between chunks
CACHE_EMBEDDINGS = True                    # Enable query caching
```

### Performance Monitoring

The system provides real-time performance metrics:
- Response time per query
- Number of context chunks used
- Similarity scores
- Overall statistics

### Expected Performance

With the optimizations, you should see:
- **2-3x faster** response times
- **Higher accuracy** due to better chunking
- **Lower costs** with optimized model selection
- **Better context** with multi-chunk retrieval

`Voila`! The command line should respond to any relevant [/dev/color](https://devcolor.org/) question with improved performance and accuracy.
