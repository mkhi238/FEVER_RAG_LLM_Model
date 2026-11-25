"""Evaluation harness.

  gold.py            FEVER claims + gold evidence (pairs and evidence sets), with the line-0 fix
  eval_retriever.py  recall@k for faiss / bm25 / hybrid      (python -m claimscan.eval.eval_retriever)
  fever_eval.py      end-to-end FEVER metric on the pipeline  (python -m claimscan.eval.fever_eval)
  metrics/fever/     the FEVER metric from huggingface/evaluate (PR #710), unchanged

The later RLVR reward for the research agent should import from here, so the
evaluation and the reward cannot drift apart.
"""
