"""Turn retriever hits (page id + line) into evidence sentences, ranked by the quality scorer.

Moved from rag/Inference/test_rag.py. Can return each sentence's "Page#line" and look
sentences up in an (id, line)-indexed corpus. Ranking: see claimscan/neurosymbolic_layer/.
"""
import pandas as pd

from claimscan.neurosymbolic_layer import score_evidence_quality


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


TOP_N = 3
MAX_PER_PAGE = 2


def _diverse_top(candidates, n=TOP_N, max_per_page=MAX_PER_PAGE):
  """Best ``n`` candidates with at most ``max_per_page`` from one page; if there are not
  enough other pages, the skipped ones fill the remaining slots (never returns fewer)."""
  picked, skipped, per_page = [], [], {}
  for c in candidates:
    if per_page.get(c['page'], 0) < max_per_page:
      picked.append(c)
      per_page[c['page']] = per_page.get(c['page'], 0) + 1
    else:
      skipped.append(c)
    if len(picked) == n:
      return picked
  return picked + skipped[:n - len(picked)]


def get_evidence(claim, evidence_results, corpus_df, with_ids=False):
  """Top-3 evidence sentences by quality score, at most 2 from the same page.

  ``evidence_results`` must be best-first (every retriever returns them that way); the
  scorer uses each hit's position, not its raw score, so faiss / bm25 / hybrid all work.
  Returns texts, or with ``with_ids=True`` a list of (text, "Page#line") pairs.
  ``corpus_df`` may be the raw corpus or ``index_corpus(corpus_df)`` (much faster).
  """
  candidates = []
  for rank, result in enumerate(evidence_results):
    text = _lookup_text(corpus_df, result['id'], result['line'])
    if text is not None:
      candidates.append({
                'text': text,
                'page': result['id'],
                'docid': f"{result['id']}#{result['line']}",
                'quality_score': score_evidence_quality(claim, text, rank, page=result['id']),
                'retrieval_score': result['score']
            })

  candidates.sort(key = lambda x: x['quality_score'], reverse = True)
  top = _diverse_top(candidates)
  if with_ids:
    return [(c['text'], c['docid']) for c in top]
  return [c['text'] for c in top]
