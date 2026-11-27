"""Split a claim into short spans, the units the natural-logic proof aligns and tags (one operator per span).

Like the FLAIR chunker ProoFVer (Krishna et al., TACL 2022) and QA-NatVer (Aly et al., EMNLP 2023) use, but with spaCy:
  noun phrases   spaCy noun_chunks                     "Barack Obama", "an Oscar"
  verb groups    aux + negation + verb, contiguous     "was born", "did not work", "has never won"
  anything else  one token per span
Then, as both papers do, a span with no content word (only stopwords / punctuation) is merged into the NEXT span
(the previous one if it is last), so "in" + "Hawaii" -> "in Hawaii" and "is" + "an accountant" -> "is an accountant".
A preposition always goes to the next span, even when spaCy counts it as a content word ("throughout").
Negation ("not", "n't", "never") stays inside its verb group: the ¬ operator needs to see it.
Punctuation at the edges of a span is dropped ("in Hawaii." -> "in Hawaii"); inside a span it stays ("Coster-Waldau").

multi_granular() adds QA-NatVer's multi-granular chunks: every run of 1..max_join consecutive spans (~8 accuracy points
in their ablation).
"""
from functools import lru_cache


@lru_cache(maxsize=1)
def _nlp():
  """en_core_web_sm, loaded once, with the parser (noun_chunks needs it)."""
  import spacy
  return spacy.load("en_core_web_sm")


def _in_verb_group(tok) -> bool:
  return tok.pos_ in ("AUX", "VERB") or tok.dep_ in ("aux", "auxpass", "neg")


def _has_content(toks) -> bool:
  return any(not (t.is_stop or t.is_punct or t.is_space) for t in toks)


def split_spans(text: str) -> list[str]:
  """Contiguous spans in order; " ".join(spans) gives back `text` up to whitespace and the punctuation between spans."""
  doc = _nlp()(text)
  starts = {c.start: c.end for c in doc.noun_chunks}
  chunks, i = [], 0                       # chunks = [start, end) token ranges covering the doc
  while i < len(doc):
    if i in starts:
      j = starts[i]
    elif _in_verb_group(doc[i]):
      j = i + 1
      while j < len(doc) and j not in starts and _in_verb_group(doc[j]):
        j += 1
    else:
      j = i + 1
    chunks.append([i, j])
    i = j

  merged, carry = [], None                # carry = start of spans waiting to join the next one
  for start, end in chunks:
    start = start if carry is None else carry
    toks = doc[start:end]
    if _has_content(toks) and not (len(toks) == 1 and toks[0].pos_ == "ADP"):
      merged.append([start, end])
      carry = None
    else:
      carry = start
  if carry is not None:                   # leftover (e.g. the final "."): join the previous span
    if merged:
      merged[-1][1] = len(doc)
    else:
      merged.append([carry, len(doc)])
  out = []
  for s, e in merged:                     # punctuation at a span's edges ("in Hawaii." -> "in Hawaii") is dropped
    while s < e and (doc[s].is_punct or doc[s].is_space):
      s += 1
    while e > s and (doc[e - 1].is_punct or doc[e - 1].is_space):
      e -= 1
    if s < e:
      out.append(doc[s:e].text)
  return out


def multi_granular(spans: list[str], max_join: int = 4) -> list[str]:
  """Every run of 1..max_join consecutive spans joined by a space: the single spans first (as given), then the pairs,
  and so on, each size left to right, without duplicates."""
  out = []
  for n in range(1, max_join + 1):
    for i in range(len(spans) - n + 1):
      chunk = " ".join(spans[i:i + n])
      if chunk not in out:
        out.append(chunk)
  return out
