from fastembed import TextEmbedding
from chromadb.api.types import EmbeddingFunction, Documents, Embeddings

class CustomEmbedding(EmbeddingFunction):
    def __init__(self):
        # Load embedding model once for efficiency if you expect multiple calls
        self.embedding_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
        
    def __call__(self, texts: Documents) -> Embeddings:
        """
        Custom function to use MiniLM-L6 v2 via fastembed. 
        Accepts a list of texts, returns a list of embeddings.
        """
        # Ensure texts is a list of strings
        # FastEmbed will do this efficiently for batches
        embeddings = self.embedding_model.embed(texts)
        return list(embeddings)  # Materialize generator if needed