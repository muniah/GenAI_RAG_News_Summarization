# RAG-Based News Summarization
### Overview
This project implements a Retrieval-Augmented Generation (RAG) system for summarizing news articles from the Hugging Face cc_news dataset. It combines semantic search using Sentence Transformers and FAISS with text summarization using a BART model via LangChain. The system retrieves relevant articles based on a user query and generates concise summaries, with an evaluation framework to assess retrieval and generation performance. Used Hugging Face dataset (cc_news): https://huggingface.co/datasets/vblagoje/cc_news

### Objective
This project uses the cc_news dataset from Hugging Face to build a news summarization system. It uses:
- LangChain for RAG
- Sentence Transformers for embeddings 
- Faiss for vector storage

### Overview
1. Fetches news articles from the 
2. Creates a vector store of article embeddings using Sentence Transformers and FAISS
3. Implements RAG to retrieve relevant articles based on a user query.
4. Generates concise summaries using a large language model (via LangChain).

### Code Structure

`data_loader.py`: Fetches and preprocesses news data from Hugging Face.
`vector_store.py`: Creates and manages FAISS vector store with embeddings.
`rag_pipeline.py`: Implements RAG for retrieval and generation.
`main.py`: Orchestrates the pipeline and provides a user interface.
`evaluator.py`: Evaluates both retrieval and generation components

### Remark on Evaluator: 

The cc_news dataset from Hugging Face doesn’t provide reference summaries or labeled relevance data, so one needs to create a small evaluation dataset or use synthetic methods. For simplicity, I focused on automated metrics (ROUGE, Precision@k) how to integrate these metrics in the codebase. One shouldn't use the 'evaluator' mode, as I have not finished exploring it. If someone wants to use this - one can do the following:

1. Prepare Ground Truth Data: Inspect the first few articles in your cc_news sample (accessible via loader.get_articles()). Assign indices of relevant articles to each query in ground_truth. For example, if articles 0 and 1 are about AI, set "AI advancements": [0, 1].
2. Reference Summaries: Write short summaries (2-3 sentences) for each query based on relevant articles, or use the first 2-3 sentences of an article as a proxy.
3. Run python main.py, load articles, and print titles/texts to identify indices


### Notes
The project uses `facebook/bart-large-cnn` for summarization due to its strong performance on news data. For production, one can consider a larger model like `mistral-7b` via API. The sample size is 100 for quick testing; it can be increased for real-world use.
Version Pinning: Specific versions are chosen to avoid compatibility issues based on the project’s requirements.
faiss-cpu: Used for demonstration; replace with faiss-gpu if GPU support is needed.
torch: Included as a dependency for transformers and sentence-transformers. If GPU is available, install the GPU version of PyTorch separately for better performance.

### Explanations of Key Methods
`NewsDataLoader.load_data`: Fetches news from Hugging Face, demonstrating dataset handling.
`NewsDataLoader.preprocess_data`: Cleans text, ensuring quality input for embeddings.
`VectorStore.create_index`: Generates semantic embeddings and builds a FAISS index, showcasing vector search.
`VectorStore.search`: Performs similarity search, highlighting efficient retrieval for RAG.
`RAGPipeline.generate`: Combines retrieval and generation, illustrating RAG principles.
`main`: Orchestrates the pipeline, demonstrating system integration.


### Clone the Repository:
git clone https://github.com/your-username/rag-news-summarization.git
cd rag-news-summarization


### Create and Activate a Virtual Environment:
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate



### Interactive Querying:

In the "Interactive Exploration" cell, enter queries like "AI advancements" or "climate change" to generate summaries.
Example output:Summary: Recent AI advancements include new models for natural language processing...
Retrieved articles: ['Article Title 1', 'Article Title 2']


### Generation Performance:
ROUGE Scores: Measures overlap between generated and reference summaries.


### Setup:
The notebook provides sample ground truth and reference summaries. Replace with actual data by inspecting articles (use the visualization cell to find relevant indices).
Example metrics output: Evaluation Metrics: {'precision@k': 0.666, 'recall@k': 0.5, 'mrr': 0.75, 'rouge1': 0.45, 'rouge2': 0.3, 'rougeL': 0.4}

### Evaluation Data: 
The provided ground_truth and reference_summaries are placeholders. Create a proper evaluation dataset by inspecting articles and assigning relevant indices/summaries.

### Future Extensions:
Add a Streamlit web UI for interactive querying.
Use a dataset like CNN/DailyMail for pre-annotated summaries.
Implement additional metrics (e.g., BLEU) or visualizations (e.g., matplotlib plots of metrics).


### Troubleshooting:
Ensure dependencies match the requirements.txt versions.
If errors occur, recreate the virtual environment: `rm -rf venv`
`python -m venv venv`
`source venv/bin/activate`
`pip install -r requirements.txt`


### Contributing
Contributions are welcome! Please:

Fork the repository.
Create a feature branch (`git checkout -b feature/YourFeature`).
Commit changes (`git commit -m "Add YourFeature"`).
Push to the branch (`git push origin feature/YourFeature`).
Open a pull request.



