
# News Article RAG (Retrieval Augmented Generation)

This Project utilizes OpenAI's API to generate embeddings for ChromaDB and utilizes those embeddings to retrieve relevant documents to feed to LLM of our choice, GPT-3.5-Turbo in this instance. The main goal of this project is to understand the RAG framework and how it can reduce hallucinations which may be prevalent in a regular LLM implementation due to the lack of contextual awareness.


## Environment Variables

To run this project, you will need to add the following environment variables to your .env file

`OPENAI_API_KEY` - Your OpenAI API key.

To use OpenAI you have to have sufficient credits in your Account


## Deployment

To deploy this project start up a python virtual environment with the following command.

Create a Virtual environment
```bash
  python3 -m venv <virtual_env_name> 
```

Activate your Virtual environment
```bash
  source <virtual_env_name>/bin/activate
```

Install the following Dependencies using the corresponding code.
Dotenv for using the environment variables
```bash
  pip install python-dotenv
```

OpenAI library
```bash
  pip install openai
```

Chroma DB library
```bash
  pip install chromadb
```


