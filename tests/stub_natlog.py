"""Stub encode / nli for the natlog tests: no models, CI-safe."""
import numpy as np


def bow_encode(texts):
  """Bag-of-words vectors (lowercased words, punctuation stripped): cosine = word overlap."""
  words = [[w.strip(".,").lower() for w in t.split()] for t in texts]
  vocab = {w: i for i, w in enumerate(sorted({w for ws in words for w in ws}))}
  out = np.zeros((len(texts), max(len(vocab), 1)), dtype=np.float32)
  for r, ws in enumerate(words):
    for w in ws:
      out[r, vocab[w]] += 1
  return out


def stub_nli(entails=(), contradicts=()):
  """nli(premise, hypothesis): equal text entails; the given (premise, hypothesis) pairs entail; contradictions
  count both ways; everything else is neutral."""
  ent, con = set(entails), set(contradicts) | {(h, p) for p, h in contradicts}
  def nli(premise, hypothesis):
    if (premise, hypothesis) in con:
      return "contradiction"
    if premise.lower() == hypothesis.lower() or (premise, hypothesis) in ent:
      return "entailment"
    return "neutral"
  return nli
