# evaluator.py
# Purpose: Evaluates the performance of the RAG-based news summarization system.
# Concept: Assesses retrieval (Precision@k, Recall@k, MRR) and generation (ROUGE) components.


from rouge_score import rouge_scorer
import numpy as np
import logging
from typing import List, Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Evaluator:
    """Evaluates retrieval and generation performance of the RAG system."""
    
    def __init__(self, vector_store, rag_pipeline):
        """
        Initialize evaluator with vector store and RAG pipeline.
        Args:
            vector_store (VectorStore): Instance for retrieval evaluation.
            rag_pipeline (RAGPipeline): Instance for generation evaluation.
        """
        self.vector_store = vector_store
        self.rag_pipeline = rag_pipeline
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    def evaluate_retrieval(self, queries: List[str], ground_truth: Dict[str, List[int]], k: int = 3) -> Dict[str, float]:
        """
        Evaluate retrieval performance using Precision@k, Recall@k, and MRR.
        Args:
            queries: List of test queries.
            ground_truth: Dict mapping queries to lists of relevant article indices.
            k: Number of articles to retrieve.
        Returns:
            Dict with average Precision@k, Recall@k, and MRR.
        """
        precisions, recalls, mrrs = [], [], []
        
        for query in queries:
            retrieved_articles = self.vector_store.search(query, k=k)
            retrieved_indices = [self.vector_store.metadata.index(article) for article in retrieved_articles]
            relevant_indices = ground_truth.get(query, [])
            
            # Precision@k: Proportion of retrieved articles that are relevant
            relevant_retrieved = len(set(retrieved_indices) & set(relevant_indices))
            precision = relevant_retrieved / k if k > 0 else 0
            precisions.append(precision)
            
            # Recall@k: Proportion of relevant articles retrieved
            recall = relevant_retrieved / len(relevant_indices) if relevant_indices else 0
            recalls.append(recall)
            
            # MRR: Reciprocal rank of the first relevant article
            mrr = 0
            for rank, idx in enumerate(retrieved_indices, 1):
                if idx in relevant_indices:
                    mrr = 1 / rank
                    break
            mrrs.append(mrr)
        
        return {
            "precision@k": np.mean(precisions),
            "recall@k": np.mean(recalls),
            "mrr": np.mean(mrrs)
        }
    
    def evaluate_generation(self, queries: List[str], reference_summaries: Dict[str, str], k: int = 3) -> Dict[str, float]:
        """
        Evaluate generation performance using ROUGE scores.
        Args:
            queries: List of test queries.
            reference_summaries: Dict mapping queries to reference summaries.
            k: Number of articles to retrieve for context.
        Returns:
            Dict with average ROUGE-1, ROUGE-2, and ROUGE-L scores.
        """
        rouge_scores = {"rouge1": [], "rouge2": [], "rougeL": []}
        
        for query in queries:
            result = self.rag_pipeline.generate_summary(query, k=k)
            generated_summary = result["summary"]
            reference_summary = reference_summaries.get(query, "")
            
            if reference_summary:
                scores = self.rouge_scorer.score(reference_summary, generated_summary)
                for metric in rouge_scores:
                    rouge_scores[metric].append(scores[metric].fmeasure)
        
        return {
            "rouge1": np.mean(rouge_scores["rouge1"]) if rouge_scores["rouge1"] else 0,
            "rouge2": np.mean(rouge_scores["rouge2"]) if rouge_scores["rouge2"] else 0,
            "rougeL": np.mean(rouge_scores["rougeL"]) if rouge_scores["rougeL"] else 0
        }
    
    def evaluate_end_to_end(self, queries: List[str], ground_truth: Dict[str, List[int]], reference_summaries: Dict[str, str], k: int = 3) -> Dict[str, float]:
        """
        Evaluate end-to-end performance combining retrieval and generation metrics.
        Args:
            queries: List of test queries.
            ground_truth: Dict mapping queries to relevant article indices.
            reference_summaries: Dict mapping queries to reference summaries.
            k: Number of articles to retrieve.
        Returns:
            Dict with retrieval and generation metrics.
        """
        retrieval_metrics = self.evaluate_retrieval(queries, ground_truth, k)
        generation_metrics = self.evaluate_generation(queries, reference_summaries, k)
        return {**retrieval_metrics, **generation_metrics}

if __name__ == "__main__":
    # Example usage
    from vector_store import VectorStore
    from rag_pipeline import RAGPipeline
    
    # Initialize components
    store = VectorStore()
    store.load_index()
    rag = RAGPipeline(store)
    
    # Sample evaluation data
    queries = ["AI advancements", "climate change"]
    ground_truth = {
        "AI advancements": [0, 1],  # Indices of relevant articles in metadata
        "climate change": [2, 3]
    }
    reference_summaries = {
        "AI advancements": "Recent advancements in AI include new models for natural language processing.",
        "climate change": "Climate change initiatives focus on reducing carbon emissions."
    }
    
    # Evaluate
    evaluator = Evaluator(store, rag)
    metrics = evaluator.evaluate_end_to_end(queries, ground_truth, reference_summaries, k=3)
    print("Evaluation Metrics:", metrics)