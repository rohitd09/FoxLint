# from fastembed import TextEmbedding

# print(TextEmbedding.list_supported_models())

from chromadb import Client

client = Client()
collections = client.list_collections()  # Returns list of dicts
collection_names = [col['name'] for col in collections]
print(collection_names)