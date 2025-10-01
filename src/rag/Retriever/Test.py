from faiss_sentence_retriever import FAISSRetriever

INDEX_DIR = r"D:\crisis-claim-analysis\artifacts\dense\faiss_bge_small_en_v1_5"
retriever = FAISSRetriever(INDEX_DIR)

# Test basic queries
queries = [
    "What is the capital of France",  # Basic geography
    "When did World War 2 end?"
    "Was the moon landing real?"
]

for query in queries:
    print(f"\nQuery: '{query}'")
    results = retriever.search(query, k=5)
    for i, result in enumerate(results):
        print(f"  {i+1}. {result['id']}#{result['line']} (score: {result['score']})")