# Simple RAG system for /dev/color Chat Bot

Welcome!

This is a very simple Command-Line Retrieval-Augmented Generation (RAG for short) system that will take in user prompts related to [/dev/color](https://devcolor.org/) and allows the user to input a query, searches a local knowledge bank for relevant context, and generates a response based on that context.

## Technical Aspects

The project uses FAISS for similarity search, **LangChain** for query processing, and **OpenAI's GPT model** for generating responses. The language used is Python so hopefully its easy to read and follow along!

## Setup

### Requirements

1. **Python 3.8+** (installed on your local system)
2. Install the following dependencies:

```sh
pip install faiss-cpu langchain openai tiktoken
```

3. Have a valid [OpenAI](https://platform.openai.com/) development account that will allow you to generate an `api_key` that is required to run the LLM. Please follow these [instructions](https://medium.com/@lorenzozar/how-to-get-your-own-openai-api-key-f4d44e60c327) for setting up the account if you don't have one.

### Project Structure

```
dev-color-project/
|--- data/
|    |-- devcolorfaq.txt
|--- check_models.py
|--- config.py
|--- embed.py
|--- main.py
|--- README.md
```

#### Code Walkthrough

- **check_models**: This will verify the OpenAI models available to you based on the supplied `api_key` in the config.py
- **config.py**: This is the configuration file for setting up your `Open AI` api keys. If you don't have one please use the one already committed there.
- **embed.py**: This creates the FAISS embeddings from the knowledge file supplied i.e. `data/devcolorfaq.txt`. In turn this creates a `faiss_index.bin` file within the same project level for all the embeddings needed to process the queries and generates a `chunks.txt` file in the `/data` directory.
- **main.py**: This file handles user queries and retrieves the best-matching document response.

### Usage

1. **Verify that the knowledge file is present and accessible** i.e. `devcolorfaq.txt`

2. **Generate the embeddings with the following**:

```sh
python embed.py
```

  * This creates the FAISS index and store text chunks onto the local file system i.e. the current project folder

3. **Run the chatbot**

```sh
python main.py
```

  - Type your question, and the system retrieves relevant knowledge
  - OpenAI's GPT model generates a response based on the retrieved text

`Voila`! The command line should respond to any relevant [/dev/color](https://devcolor.org/) question until the user decides to quit.
