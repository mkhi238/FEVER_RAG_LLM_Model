"""Turn FAISS hits (page id + line) into evidence sentences, ranked by the quality scorer.

Moved unchanged from rag/Inference/test_rag.py. api/main.py imported it from
score_evidence, where it never lived.
"""
from claimscan.evidence.scoring import score_evidence_quality


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
