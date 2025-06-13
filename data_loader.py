# data_loader.py
# Purpose: Fetch and preprocess news articles from Hugging Face's cc_news dataset.
# Efficiently loads and cleans text data, preparing it for embedding and RAG.


from datasets import load_dataset
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NewsDataLoader:
    """Loads and preprocesses news articles from Hugging Face dataset."""
    
    def __init__(self, dataset_name="cc_news", sample_size=1000):
        """
        Initialize data loader.
        Args:
            dataset_name (str): Hugging Face dataset name.
            sample_size (int): Number of articles to sample.
        """
        self.dataset_name = dataset_name
        self.sample_size = sample_size
        self.data = None
    
    def load_data(self):
        """Load dataset from Hugging Face and sample articles."""
        try:
            logger.info(f"Loading dataset: {self.dataset_name}")
            dataset = load_dataset(self.dataset_name, split="train")
            # Convert to pandas for easier handling
            self.data = dataset.to_pandas().sample(n=min(self.sample_size, len(dataset)), random_state=42)
            logger.info(f"Loaded {len(self.data)} articles")
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def preprocess_data(self):
        """Clean and preprocess text data."""
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        logger.info("Preprocessing data")
        # Remove rows with missing text
        self.data = self.data.dropna(subset=['text'])
        # Basic cleaning: remove extra whitespace
        self.data['text'] = self.data['text'].str.strip().str.replace(r'\s+', ' ', regex=True)
        # Filter out very short articles
        self.data = self.data[self.data['text'].str.len() > 100]
        logger.info(f"Preprocessed {len(self.data)} articles")
    
    def get_articles(self):
        """Return preprocessed articles."""
        if self.data is None:
            raise ValueError("Data not loaded or preprocessed.")
        return self.data[['title', 'text']].to_dict('records')

if __name__ == "__main__":
    # Example usage
    loader = NewsDataLoader(sample_size=100)
    loader.load_data()
    loader.preprocess_data()
    articles = loader.get_articles()
    print(f"Sample article: {articles[0]['title']}")