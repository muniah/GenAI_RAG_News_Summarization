# rag_pipeline.py
# Purpose: Implements RAG pipeline for retrieving relevant articles and generating summaries.
# Combines retrieval (FAISS) with generation (LLM via LangChain) for context-aware summarization.


from langchain.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from transformers import pipeline
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGPipeline:
    """Implements Retrieval-Augmented Generation for news summarization."""
    
    def __init__(self, vector_store, model_name="facebook/bart-large-cnn"):
        """
        Initialize RAG pipeline.
        Args:
            vector_store (VectorStore): Instance of VectorStore for retrieval.
            model_name (str): Hugging Face model for summarization.
        """
        self.vector_store = vector_store
        self.model_name = model_name
        self.llm = self._initialize_llm()
    
    def _initialize_llm(self):
        """Initialize Hugging Face model for summarization."""
        logger.info(f"Initializing LLM: {self.model_name}")
        hf_pipeline = pipeline("summarization", model=self.model_name, device=-1)
        llm = HuggingFacePipeline(pipeline=hf_pipeline)
        return llm
    
    def create_prompt(self, query, retrieved_articles):
        """Create prompt for summarization based on query and retrieved articles."""
        context = "\n".join([f"Article: {article['text'][:1000]}" for article in retrieved_articles])
        template = """
        Summarize the following news articles in 2-3 sentences, focusing on the topic: {query}.
        Context:
        {context}
        """
        prompt = PromptTemplate(template=template, input_variables=["query", "context"])
        return prompt.format(query=query, context=context)
    
    def generate_summary(self, query, k=3):
        """Generate summary for query using RAG."""
        logger.info(f"Generating summary for query: {query}")
        # Retrieve relevant articles
        retrieved_articles = self.vector_store.search(query, k=k)
        # Create prompt
        prompt = self.create_prompt(query, retrieved_articles)
        # Initialize LLM chain
        chain = LLMChain(llm=self.llm, prompt=PromptTemplate(template="{text}", input_variables=["text"]))
        # Generate summary
        summary = chain.run(text=prompt)
        return {
            "query": query,
            "summary": summary,
            "retrieved_articles": [article['title'] for article in retrieved_articles]
        }

if __name__ == "__main__":
    # Example usage
    from vector_store import VectorStore
    store = VectorStore()
    store.load_index()
    rag = RAGPipeline(store)
    result = rag.generate_summary("AI advancements", k=2)
    print(f"Summary: {result['summary']}")
    print(f"Retrieved articles: {result['retrieved_articles']}")