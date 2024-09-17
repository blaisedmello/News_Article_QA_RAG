import os
from dotenv import load_dotenv
import chromadb
from openai import OpenAI
from chromadb.utils import embedding_functions

# load environment variables from .env file
load_dotenv()

openai_key = os.getenv("OPENAI_API_KEY")

openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key = openai_key,
    model_name ="text-embedding-3-small"
)

# initialize chroma client for persistent vector storage
chroma_client = chromadb.PersistentClient(path="chroma_persistent_storage")
collection_name = "document_qa_collection"
collection = chroma_client.get_or_create_collection(name = collection_name, embedding_function = openai_ef)

Client = OpenAI(api_key=openai_key)

#resp = Client.chat.completions.create(
#    model = "gpt-3.5-turbo",
#    messages= [
#        {"role": "system", "content": "You are a helpful assistant."},
#        {"role": "user", "content": "What is average life expectancy in the United States?"}
#    ]
#)

# load the text files. read, store and return them as a structured format
def load_documents_from_directory(directory_path):
    print("=== Loading documents from directory ===")
    documents = []
    for filename in os.listdir(directory_path):
        if filename.endswith(".txt"):
            with open(os.path.join(directory_path, filename), "r", encoding = "utf-8"
                      )as file:
                        documents.append({"id":filename, "text":file.read()})
    return documents

# Function to split data into chunks
def split_text(text, chunk_size = 1000, chunk_overlap = 20):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - chunk_overlap
    return chunks
    