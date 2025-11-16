import os
from chromadb import Client
from langchain_text_splitters import RecursiveCharacterTextSplitter
from VectorStores.custom_embedding import CustomEmbedding

class ChromaCollectionManager:
    def __init__(self):
        self.client = Client()
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=0)
        self.custom_embeddings = CustomEmbedding()

    def get_or_create_collection(self, collection_name: str):
        collections = self.client.list_collections()
        if any(col['name'] == collection_name for col in collections):
            collection = self.client.get_collection(collection_name)
        else:
            collection = self.client.create_collection(collection_name)
        return collection

    def split_document(self, name: str):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(base_dir, '..', 'scraped_policy', f'{name}.txt')
        with open(file_path, "r", encoding="utf-8") as f:
            document_text = f.read()
        texts = self.text_splitter.split_text(document_text)
        return texts

    def add_chunks_to_chroma(self, name: str):
        collection = self.get_or_create_collection(name)
        chunks = self.split_document(name)
        
        embeddings = self.custom_embeddings(chunks)  # Pass list of chunks
        
        doc_ids = [f"{name}_{i}" for i in range(len(chunks))]
        
        collection.add(
            documents=chunks,
            metadatas=[{"source": name} for _ in chunks],
            ids=doc_ids,
            embeddings=embeddings
        )
        print(f"Added {len(chunks)} chunks to collection '{name}'")

    def get_similar_chunks(self, collection_name: str, query: str, top_k: int = 5):
        collections = self.client.list_collections()
        if not any(col.name == collection_name for col in collections):
            return {"error": f"Collection '{collection_name}' does not exist."}

        collection = self.client.get_collection(collection_name)
        
        query_embedding = self.custom_embeddings([query])[0] # Use custom_embeddings for query embedding
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        return results