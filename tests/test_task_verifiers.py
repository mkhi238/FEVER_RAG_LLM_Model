"""Per-task NLI (claimscan/verifier/tasks.py), its data prep (task_data.py) and adapter training loop
(train_task_adapter.py). Fake models and a tiny random DeBERTa only: no downloads, no artifacts."""
import json
import random

import pytest
import torch

from claimscan.verifier.task_data import load_wordnet, paws_pairs, squad_pairs, wordnet_pairs
from claimscan.verifier.tasks import TASK_NAMES, TASKS, TaskVerifiers, _View, hypothesis


def test_templates():
  assert set(TASKS) == set(TASK_NAMES)
  assert hypothesis("define", term="A poodle", definition="a dog") == "A poodle is a dog"
  assert hypothesis("question", question="Who?", answer="Obama") == "The answer to \"Who?\" is Obama."
  with pytest.raises(ValueError, match="concept"):
    hypothesis("example", instance="Paris")


class FakeVerifier:
  """Entails when every hypothesis word is in the premise; records the name it was loaded as."""
  def __init__(self, name):
    self.name = name

  def encode_text(self, claim, evidence):
    return {"input_ids": (" ".join(evidence), claim), "attention_mask": None}

  def __call__(self, input_ids, attention_mask):
    premise, hyp = input_ids
    ok = all(w in premise.lower() for w in hyp.lower().rstrip(".").split())
    return torch.tensor([[4.0, 0.0, 0.0]] if ok else [[0.0, 0.0, 4.0]])


def test_task_verifiers_zero_shot_and_separate():
  loaded = []
  def loader(name):
    loaded.append(name)
    return FakeVerifier(name)
  settings = {"tasks": {"question": {"model": "some/other-nli"}}}
  tv = TaskVerifiers(settings, "base-nli", loader=loader, nei_prob_threshold=0.2)
  r = tv.check("define", ["A poodle is a dog with curly hair."], term="A poodle", definition="a dog")
  assert (r.label, r.task_label, r.op, r.success) == ("TRUE", "DEFINED", "⊑", True)
  r = tv.check("example", ["Paris is the capital of France."], instance="Paris", concept="a river")
  assert (r.task_label, r.success) == ("UNKNOWN", False)
  assert tv.model("verify").verifier is tv.model("define").verifier                # one shared base
  assert tv.model("question").verifier.name == "some/other-nli"                   # the config switch
  assert loaded == ["base-nli", "some/other-nli"]
  with pytest.raises(ValueError):
    tv.model("summarise")


def test_explain_is_bidirectional():
  tv = TaskVerifiers({}, "base", loader=FakeVerifier)
  same = tv.check("explain", ["Rex is a dog"], original="Rex is a dog")
  more = tv.check("explain", ["Rex is a big dog"], original="Rex is a dog")
  assert same.op == "≡" and more.op == "⊑" and same.success and more.success
  assert same.task_label == "FAITHFUL"


# ---- a tiny random DeBERTa-v2 + LoRA: adapter switching and the training loop ----
def tiny_deberta():
  from transformers import DebertaV2Config, DebertaV2ForSequenceClassification
  torch.manual_seed(0)
  cfg = DebertaV2Config(vocab_size=50, hidden_size=16, num_hidden_layers=1, num_attention_heads=2,
                        intermediate_size=32, max_position_embeddings=32, num_labels=3,
                        id2label={0: "entailment", 1: "neutral", 2: "contradiction"},
                        label2id={"entailment": 0, "neutral": 1, "contradiction": 2})
  return DebertaV2ForSequenceClassification(cfg).eval()


class TinyVerifier(torch.nn.Module):
  """HFNLIVerifier's shape around the tiny model (fixed token ids instead of a tokenizer)."""
  def __init__(self):
    super().__init__()
    self.nli = tiny_deberta()

  def encode_text(self, claim, evidence):
    return {"input_ids": torch.tensor([[1, 5, 7, 2]]), "attention_mask": torch.ones(1, 4, dtype=torch.long)}

  def forward(self, input_ids, attention_mask):
    return self.nli(input_ids=input_ids, attention_mask=attention_mask).logits


def test_lora_adapters_switch_on_one_base(tmp_path):
  pytest.importorskip("peft")
  from claimscan.verifier.train_task_adapter import label_ids, train_loop, wrap_lora
  base = tiny_deberta()
  assert label_ids(base.config) == {"entailment": 0, "neutral": 1, "contradiction": 2}
  for task, seed in (("define", 1), ("example", 2)):       # two adapters, trained a few steps each
    torch.manual_seed(seed)
    m = wrap_lora(tiny_deberta(), r=2, targets=["query_proj", "value_proj"])
    batch = {"input_ids": torch.tensor([[1, 5, 7, 2], [1, 9, 3, 2]]), "attention_mask": torch.ones(2, 4, dtype=torch.long),
             "labels": torch.tensor([seed % 3, 0])}
    log = train_loop(m, iter([batch] * 5), max_steps=5, lr=0.05, device="cpu", log_every=5)
    assert log[-1]["step"] == 5
    m.save_pretrained(tmp_path / task)
  tv = TaskVerifiers({"tasks": {"define": {"adapter": str(tmp_path / "define")},
                                "example": {"adapter": str(tmp_path / "example")}}}, "tiny",
                     loader=lambda name: TinyVerifier())
  zero = tv.model("verify")                                  # zero-shot view made before any adapter loads
  enc = zero.encode_text("h", ["p"])
  with torch.no_grad():
    z0 = zero(**enc)
    d = tv.model("define")(**enc)
    e = tv.model("example")(**enc)
    z1 = zero(**enc)
  assert torch.allclose(z0, z1)                              # adapters are disabled for the zero-shot task
  assert not torch.allclose(d, e) and not torch.allclose(d, z0)
  assert tv.model("define").verifier is tv.model("verify").verifier


# ---- data prep on hand-made inputs ----
def _syn(sid, member, gloss, rels=()):
  return json.dumps({"@id": sid, "@members": member, "@partOfSpeech": "n", "Definition": gloss,
                     "SynsetRelation": [{"@relType": t, "@target": g} for t, g in rels]})


WORDNET = [
  _syn("a", "oewn-animal-n", "a living organism"),
  _syn("c", "oewn-canine-n", "a dog-like mammal", [("hypernym", "a")]),
  _syn("f", "oewn-feline-n", "a cat-like mammal", [("hypernym", "a")]),
  _syn("d", "oewn-dog-n", "a domesticated canine", [("hypernym", "c")]),
  _syn("l", "oewn-location-n", "a point or extent in space"),
  _syn("p", "oewn-big_city-n", "a large town", [("hypernym", "l")]),
  _syn("x", "oewn-Paris-n", "the capital of France", [("instance_hypernym", "p")]),
]


def test_wordnet_pairs():
  syn = load_wordnet(WORDNET)
  assert syn["p"]["lemma"] == "big city" and syn["x"]["inst"] == ["p"]
  rows = wordnet_pairs(syn, "define", random.Random(0), None)
  dog = [r for r in rows if r["source"] == "wordnet:d"]
  assert {"premise": "Dog: a domesticated canine.", "hypothesis": "Dog is a canine", "label": "entailment",
          "source": "wordnet:d"} in dog
  assert any(r["label"] == "contradiction" and r["hypothesis"] == "Dog is a feline" for r in dog)
  ex = wordnet_pairs(syn, "example", random.Random(0), None)
  assert any(r["hypothesis"] == "Paris is an example of a big city" and r["label"] == "entailment" for r in ex)
  assert any(r["hypothesis"] == "Dog is an example of a canine" for r in ex)      # has a grand-hypernym
  assert not any(r["source"] == "wordnet:c" for r in ex)                           # canine -> animal only: skipped


def test_paws_and_squad_pairs():
  pd = pytest.importorskip("pandas")
  paws = pd.DataFrame({"id": [1, 2], "sentence1": ["a b", "c d"], "sentence2": ["b a", "d c"], "label": [1, 0]})
  assert [r["label"] for r in paws_pairs(paws, None)] == ["entailment", "neutral"]
  ctx = "Obama was born in Hawaii in 1961."
  squad = pd.DataFrame({"context": [ctx] * 3, "question": ["Where?", "When?", "Why?"],
                        "answers": [{"text": ["Hawaii"]}, {"text": ["1961"]}, {"text": []}]})
  rows = squad_pairs(squad, random.Random(0), None)
  labels = [(r["hypothesis"], r["label"]) for r in rows]
  assert ("The answer to \"Where?\" is Hawaii.", "entailment") in labels
  assert ("The answer to \"Where?\" is 1961.", "contradiction") in labels
  assert sum(l == "neutral" for _, l in labels) == 1 and all(r["premise"] == ctx for r in rows)
