def score_evidence_quality(claim, evidence_text, retrieval_score, e = 0.01):
  
  score = 1.0 / (retrieval_score + e)

  claim_lower = claim.lower()
  evidence_lower = evidence_text.lower()

  if any(phrase in evidence_lower for phrase in ['if', 'suppose', 'hypothetical', 'would be']):
      score *= 0.5
  if any(phrase in evidence_lower for phrase in ['probably', 'perhaps', 'maybe']):
      score *= 0.75
  
  claim_entities = set(word for word in claim_lower.split() if len(word) > 3)
  evidence_entities = set(word for word in evidence_lower.split() if len(word) > 3)
  overlap = len(claim_entities & evidence_entities) / max(len(claim_entities), 1)
  score *= (1 + overlap)

  if len(evidence_text.split()) < 5:
    score *= 0.3
    
  if 'disambiguation' in evidence_lower:
    score *= 0.2

  score *= filter_evidence_by_claim_type(claim_lower, evidence_lower)
  
  return score
  

def filter_evidence_by_claim_type(claim, evidence):

  for i in ['wrote', 'written', 'author', 'authored', 'created', 'made', 'directed', 'produced', 'discovered', 'featured', 'featured on']:
    if i in claim:
      if any(j in evidence for j in ['wrote', 'written by', 'authored by', 'author of', 'created by', 'recorded by', 'directed by', 'produced by']):
        return 1.5
      
      elif any(l in evidence for l in ['featuring', 'starring']):
         return 1.25
         
      
      elif any(k in evidence for k in ['published', 'edition', 'adaptation', 'based on']):
        return 0.6
      
  for i in ['made of', 'composed of']:
    if i in claim:
       if any(l in evidence for l in ['made of', 'composed of', 'comprised of', 'consists of', 'built of']):
          return 1.25
          
  
  for i in ['born', 'died', 'lived']:
      if i in claim:
          if any(j in evidence for j in ['born in', 'died in', 'was born', 'birth']):
              return 1.3
      
  for i in ['largest', 'smallest', 'tallest', 'biggest', 'most', 'least']:
    if i in claim:

      claim_comparative = [word for word in ['largest', 'smallest', 'tallest', 'biggest', 'most', 'least'] if word in claim]
      evidence_comparative = [word for word in ['largest', 'smallest', 'tallest', 'biggest', 'most', 'least'] if word in evidence]

      if claim_comparative and claim_comparative == evidence_comparative:
        return 1.2  
      else:
        return 1.0 
      
  for i in ['capital', 'located', 'country']:
      if i in claim:
          if any(phrase in evidence for phrase in ['capital of', 'located in', 'country', 'city']):
              return 1.3
          

  for i in ['current', 'now', 'former', 'past', 'previous']:
    if i in claim:
        # If claim explicitly says "current" or "now", penalize past evidence
        if any(word in claim for word in ['current', 'now']):
            if any(word in evidence for word in ['served', 'was', 'acted', 'worked', 'former', 'previously']):
                return 0.6
        # If claim explicitly says "former" or "past", boost past evidence  
        elif any(word in claim for word in ['former', 'past', 'previous']):
            if any(word in evidence for word in ['served', 'was', 'acted', 'worked', 'former', 'previously']):
                return 1.3
        return 1.0
    
  return 1.0 
     