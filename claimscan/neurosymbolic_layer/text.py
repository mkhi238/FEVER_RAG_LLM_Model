"""Word helpers for the neurosymbolic rules: whole-word phrase tests, content words, page-title words.

All keyword tests are whole-word (regex \\b), not substring: 'if' no longer matches "life".
"""
import re

STOPWORDS = frozenset("""
a an the and or but nor of in on at to for from by with as into onto over under about after before
is are was were be been being am do does did has have had having it its this that these those
he she they them his her their there here which who whom whose what when where while than then
also only some such other more not no yes can could would should will shall may might must
lrb rrb lsb rsb colon
""".split())

_WORD = re.compile(r"[A-Za-z0-9]+")
_PAREN = re.compile(r"-LRB-.*?-RRB-")       # FEVER page ids: Savages_-LRB-2012_film-RRB-
_ESCAPE = re.compile(r"-[A-Z]+-")           # -COLON-, -LRB- left over, ...
_PATTERNS: dict = {}


def has_any(text, phrases):
  """True if any phrase occurs in ``text`` as whole words (``text`` already lowercased)."""
  key = tuple(phrases)
  pat = _PATTERNS.get(key)
  if pat is None:
    pat = _PATTERNS[key] = re.compile(r"\b(?:" + "|".join(re.escape(p) for p in phrases) + r")\b")
  return pat.search(text) is not None


def content_words(text):
  """Lowercased content words. Words of 3 letters or fewer are kept only when capitalised
  (UK, Fox, Ali) or numeric; stopwords are always dropped."""
  out = set()
  for w in _WORD.findall(text):
    lw = w.lower()
    if lw in STOPWORDS:
      continue
    if len(lw) > 3 or w[0].isupper() or w.isdigit():
      out.add(lw)
  return out


def title_words(page):
  """Content words of a FEVER page id, without its (disambiguator): 'Savages_-LRB-2012_film-RRB-' -> {'savages'}."""
  if not page:
    return set()
  t = _ESCAPE.sub(" ", _PAREN.sub(" ", str(page))).replace("_", " ")
  return {w.lower() for w in _WORD.findall(t) if w.lower() not in STOPWORDS}
