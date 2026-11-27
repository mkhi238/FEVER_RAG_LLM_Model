"""Load the FAISS index and run a few queries. Needs ~26 GB free RAM for the current IVFFlat index."""
from claimscan.config import config
from claimscan.retrieval.faiss_retriever import FAISSRetriever

retriever = FAISSRetriever(config.retriever_index_dir, config.retriever_model)

# Test basic queries
queries = [
    "What is the capital of France",  # Basic geography
    "When did World War 2 end?",  # comma was missing, which merged this query with the next
    "Was the moon landing real?"
]

for query in queries:
    print(f"\nQuery: '{query}'")
    results = retriever.search(query, k=5)
    for i, result in enumerate(results):
        print(f"  {i+1}. {result['id']}#{result['line']} (score: {result['score']})")