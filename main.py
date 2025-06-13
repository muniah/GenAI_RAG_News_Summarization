# main.py
# Purpose: Orchestrates the RAG pipeline and provides a command-line interface.
# Ties together data loading, vector storage, and RAG for an end-to-end application.

from data_loader import NewsDataLoader
from vector_store import VectorStore
from rag_pipeline import RAGPipeline
from evaluator import Evaluator
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_evaluation():
    """Run evaluation on the RAG system."""
    # Initialize components
    loader = NewsDataLoader(sample_size=100)
    loader.load_data()
    loader.preprocess_data()
    articles = loader.get_articles()
    
    store = VectorStore()
    store.create_index(articles)
    store.save_index()
    
    rag = RAGPipeline(store)
    
    # Sample evaluation data 
    queries = ["AI advancements", "climate change"]
    ground_truth = {
        "AI advancements": [0, 1],  # Replace with actual relevant article indices
        "climate change": [2, 3]
    }
    reference_summaries = {
        "AI advancements": "Recent advancements in AI include new models for natural language processing.",
        "climate change": "Climate change initiatives focus on reducing carbon emissions."
    }
    
    evaluator = Evaluator(store, rag)
    metrics = evaluator.evaluate_end_to_end(queries, ground_truth, reference_summaries, k=3)
    logger.info("Evaluation Metrics: %s", metrics)

def main():
    """Run the news summarization system with optional evaluation mode."""
    mode = input("Enter mode (interactive/evaluate): ").lower()
    
    if mode == "evaluate":
        run_evaluation()
    else:
        # Load and preprocess data
        logger.info("Starting data loading")
        loader = NewsDataLoader(sample_size=100)
        loader.load_data()
        loader.preprocess_data()
        articles = loader.get_articles()
        
        # Create or load vector store
        logger.info("Initializing vector store")
        store = VectorStore()
        store.create_index(articles)
        store.save_index()
        
        # Initialize RAG pipeline
        logger.info("Initializing RAG pipeline")
        rag = RAGPipeline(store)
        
        # Interactive query loop
        while True:
            query = input("\nEnter a query (or 'quit' to exit): ")
            if query.lower() == 'quit':
                break
            result = rag.generate_summary(query, k=3)
            print(f"\nSummary: {result['summary']}")
            print(f"Retrieved articles: {result['retrieved_articles']}")

if __name__ == "__main__":
    main()