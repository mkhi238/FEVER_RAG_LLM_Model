import torch
from pathlib import Path
import pandas as pd
import sys
verifier_dir = Path(r"C:\Users\mukun\crisis-claim-analysis\src\rag\Verifier")
retriever_dir = Path(r"C:\Users\mukun\crisis-claim-analysis\src\rag\Retriever")

#adds the directory to the current directory (cd) to tell it where to go to search for modules to import
#.insert(0, ...) adds directory to beginning of sys.path (0 is insert at position 0 (first priority), it wll look at these paths first before looking here
#How Python finds imports: Current directory, Directories in sys.path (a list of paths), Standard library locations
#Standard library locations: os - operating system interface, json - JSON encoding/decodin, pathlib - file path operations, math - mathematical functions
#usually where Python itself is installed: C:\Users\mukun\AppData\Local\Programs\Python\Python311\Lib\
sys.path.insert(0, str(retriever_dir))
sys.path.insert(0, str(verifier_dir))
from DeBERTa_Model import FEVERVerifier
from faiss_sentence_retriever import FAISSRetriever
from config import config
from score_evidence import score_evidence_quality

def get_evidence(claim, evidence_results, corpus_df):
  candidates = []
  for result in evidence_results:
    
    #match corpus_df sentence to result sentence and line

    #result only holds id, line, score. Corpus df holds id, line, text (id is like 'obama')
    match = corpus_df[(corpus_df['id'] == result['id']) & (corpus_df['line'] == result['line'])]
    if not match.empty:
      text = match.iloc[0]['text']
      new_score = score_evidence_quality(claim, text, result['score'])
      candidates.append({
                'text': text,
                'quality_score': new_score,
                'retrieval_score': result['score']
            })
  
  candidates.sort(key = lambda x: x['quality_score'], reverse = True)
  top_cands = []
  for c in candidates[:3]:
    top_cands.append(c['text'])
  return top_cands

def fact_check(claim):

  evidence_results = retriever.search(claim, k=30)

  print(f"Retrieved {len(evidence_results)} results from FAISS")

  evidence = get_evidence(claim, evidence_results, corpus_df) 
  if not evidence:
    return "NOT ENOUGH INFO", "No relevant evidence found in the corpus."

  #dim=1: Tells argmax to find the maximum along dimension 1 (the class dimension)
  #logits shape is [batch_size, num_classes] = [1, 3] for single prediction
  #dim=0 = across batches , dim=1 = across classes

  #.item(): Converts a single-value tensor to a Python number, torch.argmax(logits, dim=1) returns tensor [2], .item returns 2
  encoding = model.encode_text(claim, evidence[:3])

  with torch.no_grad():
    logits = model(encoding['input_ids'], encoding['attention_mask'])
    #collects the highest liklehood outcome
    prediction_idx = torch.argmax(logits, dim=1).item()
    probs = torch.softmax(logits, dim=1)[0]
    print(f"\nModel predictions:")
    print(f"  TRUE: {probs[0]:.4f}")
    print(f"  FALSE: {probs[1]:.4f}")
    print(f"  NOT ENOUGH INFO: {probs[2]:.4f}")

  label_map = {0: "TRUE", 1: "FALSE", 2: "NOT ENOUGH INFO"}
  prediction = label_map[prediction_idx]

  confidence = torch.max(probs).item()
  if confidence < 0.5:
    original_prediction = prediction
    if prediction != "NOT ENOUGH INFO":
      prediction = "NOT ENOUGH INFO"
      print (f"Low confidence ({confidence:.3f}) of original {original_prediction} prediction - considered now to be NOT ENOUGH INFO")

  if prediction == "TRUE":
    response = f"This statement is TRUE supported by the following evidence: {evidence[0]}"

  elif prediction == "FALSE":
    response = f"This statement is FALSE refuted by the following evidence: {evidence[0]}"
  
  else:
    response = f"There is insufficient evidence to prove this statements truth"
  
  return prediction, response

if __name__ == "__main__":

    model = FEVERVerifier()
    model_path = config.verifier_output_dir / "trained_model" / "model.pt"
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    retriever = FAISSRetriever(config.retriever_index_dir, config.retriever_model)
    corpus_df = pd.read_parquet(config.corpus)

    print("Loading complete \n")

    test_claims = [
    # Clear factual claims
    "Water boils at 100 degrees Celsius",
    "Tokyo is the capital of Japan",
    "Albert Einstein developed the theory of relativity",
    
    # Current/temporal claims (test temporal filtering)
    "Trump is the president of the United States",
    "Joe Biden is the current president of the United States",
    "Barack Obama is the former president of the United States",
    "Canada was a colony of the UK",
    "India was also a colony of the UK",
    "Socrates was a real person",
    
    # Sports/records (test comparative + biographical)
    "Lebron James is the all time points leader in the NBA",
    "Michael Jordan played for the Chicago Bulls",
    "Tom Brady won seven Super Bowls",
    "Lionel Messi has won a world cup or seven Super Bowls",
    
    # Clear false claims
    "The moon is made of cheese",
    "Cats can fly naturally",
    "The Pacific Ocean is the smallest ocean",
    "The Earth is flat",
    "The Eiffel Tower is in London",
    "Australia is the largest country in the world",
    "The sun orbits the Earth",
    "Steph Curry is the all time 3-pt leader in the WNBA",
    
    # Authorship claims (test authorship filtering)
    "Charles Dickens wrote Oliver Twist",
    "J.K. Rowling wrote Harry Potter",
    "William Shakespeare authored Hamlet",
    "Hamlet is a play by Shakespeare",
    "George Orwell wrote 1984",
    "Jane Austen wrote Pride and Prejudice",
    "Ernest Hemingway wrote The Old Man and the Sea",
    
    # Geographic claims (test geographic filtering)
    "Rome is the capital of Italy",
    "Moscow is located in Russia",
    "The Amazon rainforest is in South America",
    
    # Biographical claims (test biographical filtering)
    "Steve Jobs founded Apple",
    "Nelson Mandela was president of South Africa",
    "Martin Luther King Jr. was born in Atlanta",
    "Leonardo da Vinci painted the Mona Lisa",
    "Drake made Gods Plan",
    
    # Edge cases that might lack evidence
    "The weather is nice today",
    "This coffee tastes good"
    ]
    
    for claim in test_claims:
      pred, response = fact_check(claim)
      print(f"\nClaim: {claim}")
      print(f"Result: {response}")


#1. Retrieval: Get k=10 candidates from FAISS
#2. Quality scoring: Score all 10, sort by quality
#3. Select top 3: Pick the 3 highest quality pieces
#4. Model inference: Send those 3 to DeBERTa

#If k=10 and the right evidence is ranked #15 in FAISS, you never see it. Your quality scorer can't pick what it never receives.
#If k=50 and the right evidence is at #15, your quality scorer gets it, scores it highly, and includes it in the top 3 sent to the model.


def fact_check_cleaned(claim, k=30, conf_threshold=0.5):
    """Fact check without debug prints - for MLflow experiments"""
    evidence_results = retriever.search(claim, k=k)
    evidence = get_evidence(claim, evidence_results, corpus_df)
    
    if not evidence:
        return "NOT ENOUGH INFO"
    
    encoding = model.encode_text(claim, evidence[:3])
    
    with torch.no_grad():
        logits = model(encoding['input_ids'], encoding['attention_mask'])
        prediction_idx = torch.argmax(logits, dim=1).item()
        probs = torch.softmax(logits, dim=1)[0]
    
    label_map = {0: "TRUE", 1: "FALSE", 2: "NOT ENOUGH INFO"}
    prediction = label_map[prediction_idx]
    
    confidence = torch.max(probs).item()
    if confidence < conf_threshold and prediction != "NOT ENOUGH INFO":
        prediction = "NOT ENOUGH INFO"
    
    return prediction