import os
from dotenv import load_dotenv
import chromadb
from openai import OpenAI
from chromadb.utils import embedding_functions

load_dotenv()

openai_key = os.getenv("OPENAI_API_KEY")

openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key = openai_key,
    model_name ="text-embedding-3-small"
)

chroma_client = chromadb.PersistentClient(path="chroma_persistent_storage")
collection_name = "document_qa_collection"
collection = chroma_client.get_or_create_collection(name = collection_name, embedding_function = openai_ef)

Client = OpenAI(project="proj_sdf1rUhS6ox4sv7eVnMZFY0I",api_key=openai_key)

resp = Client.chat.completions.create(
    model = "gpt-3.5-turbo",
    messages= [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the meaning of life?"}
    ]
)

print(resp.choices[0].message.content)