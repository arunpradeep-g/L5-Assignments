import os
import sys
import chromadb
from chromadb.utils import embedding_functions
from gpt4all import Embed4All  # <-- Fixed: Using Embed4All for embeddings
from google import genai       # <-- Fixed: Using the modern Google GenAI library

# Link both the toolkit binary path AND the Windows driver repository
gpt4all_lib_path = r"C:\Users\Arun\gpt4all\lib"

if os.path.exists(gpt4all_lib_path):
    os.add_dll_directory(gpt4all_lib_path)

# 1. Configure the New Gemini Client
# The modern SDK uses genai.Client().
client = genai.Client(api_key="AQ.Ab8RN6IWGcgvhJZUdwWezo263NdQ4Xc9xq8GLnO-EZJLPQ3dkQ")

# 2. Setup ChromaDB with local embeddings using Embed4All
print("Loading local embedding model safely...")
# Embed4All automatically manages the correct miniLM or nomic embedding model configuration
embed_client = Embed4All()

class GPT4AllEmbeddingFunction(embedding_functions.EmbeddingFunction):
    # Fixed: Explicitly defining a standard __init__ clears the ChromaDB warning completely
    def __init__(self):
        pass
    
    def __call__(self, input: chromadb.Documents) -> chromadb.Embeddings:
        return [embed_client.embed(text) for text in input]

# Initialize Chroma client and collection
chroma_client = chromadb.Client()
embedding_fn = GPT4AllEmbeddingFunction()
collection = chroma_client.get_or_create_collection(
    name="my_rag_collection",
    embedding_function=embedding_fn
)

# 3. Add Documents to ChromaDB (Knowledge Base)
documents = [
    "The secret project code name is Project Nebula.",
    "Project Nebula aims to reduce energy consumption by 40% using smart grids.",
    "The lead researcher for Project Nebula is Dr. Aris Thorne.",
    "Project Nebula is scheduled for deployment in December 2026."
]

ids = ["doc1", "doc2", "doc3", "doc4"]

print("Adding documents to ChromaDB...")
collection.add(documents=documents, ids=ids)

# 4. Query the Database
query_text = "What is the goal of Project Nebula and who leads it?"
print(f"\nUser Query: {query_text}")

# Retrieve top 4 relevant documents from ChromaDB
results = collection.query(query_texts=[query_text], n_results=4)

retrieved_docs = results["documents"][0]

print("\nRetrieved Context:")
for doc in retrieved_docs:
    print(f"- {doc}")

# 5. Generate Answer using Gemini LLM
context_str = "\n".join(retrieved_docs)
prompt = f"""
You are a helpful assistant. Use the following context to answer the question.
If the answer cannot be found in the context, say "I don't know."

Context:
{context_str}

Question: {query_text}
"""

print("\nGenerating response with Gemini...")
# Fixed: Modern syntax uses client.models.generate_content
#response = client.models.generate_content(
#    model='gemini-2.5-flash',
#    contents=prompt,
#)
chat = client.chats.create(model="gemini-2.5-flash")
response = chat.send_message(prompt)

print("\n--- Final Answer ---")
print(response.text)
print("-" * 30 + "\n")