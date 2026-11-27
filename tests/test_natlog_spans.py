"""Claim -> spans (claimscan/neurosymbolic_layer/natlog/spans.py). Needs spaCy + en_core_web_sm (requirements-dev.txt)."""
import re
import importlib.util

import pytest

pytest.importorskip("spacy")
if importlib.util.find_spec("en_core_web_sm") is None:
  pytest.skip("spaCy model en_core_web_sm is not installed", allow_module_level=True)

from claimscan.neurosymbolic_layer.natlog.spans import multi_granular, split_spans


@pytest.mark.parametrize("claim, spans", [
  ("Barack Obama was born in Hawaii.", ["Barack Obama", "was born", "in Hawaii"]),
  ("Nikolaj Coster-Waldau did not work with Fox.", ["Nikolaj Coster-Waldau", "did not work", "with Fox"]),
  ("Tom Hanks has never won an Oscar.", ["Tom Hanks", "has never won", "an Oscar"]),
  ("Stranger Things isn't set in Bloomington, Indiana.", ["Stranger Things", "isn't set", "in Bloomington", "Indiana"]),
  ("The Godfather was directed by Francis Ford Coppola.", ["The Godfather", "was directed", "by Francis Ford Coppola"]),
  # "is" has no content word, so it is merged into the next span
  ("Adrienne Bailon is an accountant.", ["Adrienne Bailon", "is an accountant"]),
  ("Sarah Hyland plays Haley Dunphy on Modern Family.", ["Sarah Hyland", "plays", "Haley Dunphy", "on Modern Family"]),
])
def test_split_spans(claim, spans):
  assert split_spans(claim) == spans


@pytest.mark.parametrize("claim", [
  "The Eiffel Tower is located in Paris, France.",
  "It has never been to the Moon.",             # all stopwords: merged into one span, "never" kept
  "Murda Beatz's real name is Marshall Mathers.",
])
def test_spans_cover_the_claim(claim):
  spans = split_spans(claim)
  assert spans and all(s.strip() for s in spans)
  assert re.sub(r"[\s.,]", "", " ".join(spans)) == re.sub(r"[\s.,]", "", claim)


def test_negation_is_never_dropped():
  for claim in ["Fox did not hire him.", "Fox didn't hire him.", "She has never acted.", "It has never been to the Moon."]:
    assert any(neg in s for s in split_spans(claim) for neg in ("not", "n't", "never")), claim


def test_empty_text():
  assert split_spans("") == [] and split_spans(" . ") == []


def test_multi_granular():
  assert multi_granular(["Barack Obama", "was born", "in Hawaii"]) == [
    "Barack Obama", "was born", "in Hawaii", "Barack Obama was born", "was born in Hawaii",
    "Barack Obama was born in Hawaii"]
  assert multi_granular(["a", "b", "c", "d", "e"], max_join=2) == ["a", "b", "c", "d", "e", "a b", "b c", "c d", "d e"]
  assert multi_granular(["a", "a"]) == ["a", "a a"]
  assert multi_granular([]) == []
