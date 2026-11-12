"""Turn FAISS hits (page id + line) into evidence sentences, ranked by the quality scorer.

Moved from rag/Inference/test_rag.py (api/main.py imported it from score_evidence, where
it never lived). The ranking is unchanged; it can now also return each sentence's
"Page#line" and look sentences up in an (id, line)-indexed corpus.
"""
import pandas as pd

from claimscan.evidence.scoring import score_evidence_quality


def _lookup_text(corpus_df, page, line):
  """Sentence text for (page, line), or None. Fast path for a corpus indexed by (id, line)."""
  if isinstance(corpus_df.index, pd.MultiIndex) and corpus_df.index.names == ["id", "line"]:
    try:
      row = corpus_df.loc[(page, line)]
    except KeyError:
      return None
    return row["text"] if isinstance(row, pd.Series) else row.iloc[0]["text"]
  match = corpus_df[(corpus_df['id'] == page) & (corpus_df['line'] == line)]
  return None if match.empty else match.iloc[0]['text']


def index_corpus(corpus_df):
  """Index the corpus by (id, line) so get_evidence looks sentences up instead of scanning 25M rows."""
  return corpus_df.set_index(["id", "line"]).sort_index()


def get_evidence(claim, evidence_results, corpus_df, with_ids=False):
  """Top-3 evidence sentences by quality score.

  Returns their texts, or with ``with_ids=True`` a list of (text, "Page#line") pairs.
  ``corpus_df`` may be the raw corpus or ``index_corpus(corpus_df)`` (much faster).
  """
  candidates = []
  for result in evidence_results:
    text = _lookup_text(corpus_df, result['id'], result['line'])
    if text is not None:
      new_score = score_evidence_quality(claim, text, result['score'])
      candidates.append({
                'text': text,
                'docid': f"{result['id']}#{result['line']}",
                'quality_score': new_score,
                'retrieval_score': result['score']
            })

  candidates.sort(key = lambda x: x['quality_score'], reverse = True)
  if with_ids:
    return [(c['text'], c['docid']) for c in candidates[:3]]
  return [c['text'] for c in candidates[:3]]
