"""The symbolic rules applied to each retrieved sentence (moved unchanged from the old evidence/scoring.py).

  * rule_multiplier: hedge words, claim/evidence word overlap (page title counts as evidence),
    a page-title bonus, and the claim-type rules, multiplied together and clamped to
    [MIN_MULT, MAX_MULT] so no single keyword can sink or dominate a sentence.
  * structural_penalty: scales a score down for very short sentences and disambiguation pages
    (applied outside the clamp).

score.py combines these with the retrieval rank prior.
"""
from claimscan.neurosymbolic_layer.text import _WORD, content_words, has_any, title_words

MIN_MULT, MAX_MULT = 0.5, 3.0

SUPERLATIVES = ['largest', 'smallest', 'tallest', 'biggest', 'most', 'least']
PAST_ROLE = ['served', 'worked', 'former', 'formerly', 'previously', 'retired']


def rule_multiplier(claim, evidence_text, page=None):
  """Product of every rule for (claim, sentence), clamped to [MIN_MULT, MAX_MULT]. 1.0 = no opinion."""
  claim_lower = claim.lower()
  evidence_lower = evidence_text.lower()
  mult = 1.0

  if has_any(evidence_lower, ['if', 'suppose', 'supposedly', 'hypothetical', 'hypothetically', 'would be']):
    mult *= 0.5
  if has_any(evidence_lower, ['probably', 'perhaps', 'maybe', 'possibly']):
    mult *= 0.75

  # overlap of the claim's content words with the sentence + its page title
  # (many sentences say "He ..." / "It ..." and only name the subject in the title)
  claim_words = content_words(claim)
  page_words = title_words(page)
  evidence_words = {w.lower() for w in _WORD.findall(evidence_text)} | page_words
  overlap = len(claim_words & evidence_words) / max(len(claim_words), 1)
  mult *= (1 + overlap)

  # page-title bonus: the claim names the sentence's page (graded by how much of the title)
  if page_words:
    claim_all = {w.lower() for w in _WORD.findall(claim)}
    mult *= 1 + 0.5 * len(page_words & claim_all) / len(page_words)

  mult *= filter_evidence_by_claim_type(claim_lower, evidence_lower)

  return min(max(mult, MIN_MULT), MAX_MULT)


def structural_penalty(score, evidence_text):
  """``score`` after the sentence-shape penalties (outside the clamp): short sentences, disambiguation pages."""
  evidence_lower = evidence_text.lower()
  if len(evidence_text.split()) < 5:
    score *= 0.3
  if 'disambiguation' in evidence_lower or 'may refer to' in evidence_lower:
    score *= 0.2
  return score


def filter_evidence_by_claim_type(claim, evidence):
  """Product of every claim-type rule that applies (both arguments lowercased). 1.0 = no opinion."""
  mult = 1.0

  if has_any(claim, ['wrote', 'written', 'author', 'authored', 'created', 'made', 'directed', 'produced',
                     'discovered', 'featured']):
    if has_any(evidence, ['wrote', 'written by', 'authored by', 'author of', 'created by', 'recorded by',
                          'directed by', 'produced by']):
      mult *= 1.5
    elif has_any(evidence, ['featuring', 'starring']):
      mult *= 1.25
    elif has_any(evidence, ['published', 'edition', 'adaptation', 'based on']):
      mult *= 0.6

  if has_any(claim, ['made of', 'composed of']):
    if has_any(evidence, ['made of', 'composed of', 'comprised of', 'consists of', 'built of']):
      mult *= 1.25

  if has_any(claim, ['born', 'died', 'lived']):
    if has_any(evidence, ['born in', 'died in', 'was born', 'birth', 'born']):
      mult *= 1.3

  claim_sup = [w for w in SUPERLATIVES if has_any(claim, [w])]
  if claim_sup and claim_sup == [w for w in SUPERLATIVES if has_any(evidence, [w])]:
    mult *= 1.2

  if has_any(claim, ['capital', 'located', 'country']):
    if has_any(evidence, ['capital of', 'located in', 'country', 'city']):
      mult *= 1.3

  # "now"/"current" claims vs past-role evidence. 'was' is not a past marker: nearly every sentence has it.
  if has_any(claim, ['current', 'currently', 'now']):
    if has_any(evidence, PAST_ROLE):
      mult *= 0.6
  elif has_any(claim, ['former', 'formerly', 'past', 'previous', 'previously']):
    if has_any(evidence, PAST_ROLE):
      mult *= 1.3

  return mult
