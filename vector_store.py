# vector_store.py
# Purpose: Create and manage FAISS vector store for article embeddings.
# Uses Sentence Transformers for semantic embeddings and FAISS for efficient similarity search.

from sentence_transformers import SentenceTransformer # type: ignore
import faiss
import numpy as np
import pickle
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VectorStore:
    """Manages creation and querying of FAISS vector store for news articles."""
    
    def __init__(self, model_name="all-MiniLM-L6-v2", index_path="faiss_index.bin", metadata_path="metadata.pkl"):
        """
        Initialize vector store.
        Args:
            model_name (str): Sentence Transformer model for embeddings.
            index_path (str): Path to save/load FAISS index.
            metadata_path (str): Path to save/load article metadata.
        """
        self.model = SentenceTransformer(model_name)
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.index = None
        self.metadata = None
    
    def create_index(self, articles):
        """Create FAISS index from article texts."""
        logger.info("Creating FAISS index")
        texts = [article['text'] for article in articles]
        # Generate embeddings
        embeddings = self.model.encode(texts, show_progress_bar=True)
        # Initialize FAISS index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings.astype(np.float32))
        self.metadata = articles
        logger.info(f"Indexed {len(articles)} articles")
    
    def save_index(self):
        """Save FAISS index and metadata to disk."""
        if self.index is None or self.metadata is None:
            raise ValueError("Index or metadata not initialized.")
        faiss.write_index(self.index, self.index_path)
        with open(self.metadata_path, 'wb') as f:
            pickle.dump(self.metadata, f)
        logger.info(f"Saved index to {self.index_path} and metadata to {self.metadata_path}")
    
    def load_index(self):
        """Load FAISS index and metadata from disk."""
        try:
            self.index = faiss.read_index(self.index_path)
            with open(self.metadata_path, 'rb') as f:
                self.metadata = pickle.load(f)
            logger.info(f"Loaded index from {self.index_path} and metadata from {self.metadata_path}")
        except Exception as e:
            logger.error(f"Error loading index: {e}")
            raise
    
    def search(self, query, k=5):
        """Search for top-k relevant articles based on query."""
        if self.index is None:
            raise ValueError("Index not loaded or created.")
        query_embedding = self.model.encode([query])[0]
        distances, indices = self.index.search(np.array([query_embedding]).astype(np.float32), k)
        results = [self.metadata[i] for i in indices[0]]
        return results

if __name__ == "__main__":
    # Example usage
    from data_loader import NewsDataLoader
    loader = NewsDataLoader(sample_size=10)
    loader.load_data()
    loader.preprocess_data()
    articles = loader.get_articles()
    
    store = VectorStore()
    store.create_index(articles)
    store.save_index()
    store.load_index()
    results = store.search("technology news", k=2)
    print(f"Search results: {[article['title'] for article in results]}")