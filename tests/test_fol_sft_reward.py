"""Decomposer SFT data prep (fol/sft_data.py) and the GRPO reward proposal (fol/reward.py). No models, no downloads."""
import json

from claimscan.instruct.front_decomposer import FakeClient
from claimscan.instruct.prompts import FEW_SHOTS
from claimscan.neurosymbolic_layer.fol.check import UnitCheck
from claimscan.neurosymbolic_layer.fol.decomposition import parse_output
from claimscan.neurosymbolic_layer.fol.reward import fol_reward
from claimscan.neurosymbolic_layer.fol.sft_data import convert, folio_pairs, malls_pairs, silver_records, to_target


def checker_from(truth):
  """TRUE for hypotheses in `truth`, FALSE for ones starting with 'not:' in it, else NEI."""
  def check(hyp, evidence):
    if hyp in truth:
      return UnitCheck("TRUE", "⊑", 0)
    if "not:" + hyp in truth:
      return UnitCheck("FALSE", "¬", 0)
    return UnitCheck("NOT ENOUGH INFO", "#")
  return check


HINTON_REPLY = json.dumps(FEW_SHOTS[0][1])
HINTON_TRUE = {"S. E. Hinton is a fantasy writer", "S. E. Hinton is American", "S. E. Hinton was born in 1950"}


def test_reward():
  ok = fol_reward(HINTON_REPLY, ["e"], "TRUE", checker_from(HINTON_TRUE))
  assert ok.valid and ok.label == "TRUE" and ok.reward == 1.2
  wrong = fol_reward(HINTON_REPLY, ["e"], "FALSE", checker_from(HINTON_TRUE))
  assert wrong.valid and wrong.reward == 0.2
  bad = fol_reward("not json", ["e"], "TRUE", checker_from(HINTON_TRUE))
  assert not bad.valid and bad.reward == 0.0 and bad.error


def test_reward_joins_claims_with_and():
  reply = json.dumps(FEW_SHOTS[2][1])          # two claims: Fox (true here) and the Oscar (refuted here)
  truth = {"Nikolaj Coster-Waldau worked with the Fox Broadcasting Company", "not:Nikolaj Coster-Waldau has won an Oscar"}
  r = fol_reward(reply, ["e"], "FALSE", checker_from(truth))
  assert r.label == "FALSE" and r.reward == 1.2


def test_to_target_and_convert():
  t = to_target("All dogs are animals.", "∀x (Dog(x) → Animal(x))")
  [d] = parse_output(t)
  assert d.formula.text == "All dogs are animals." and d.entities == {} and d.queries == []
  t = to_target("Bonnie is in this club.", "InThisClub(bonnie)")
  assert t["claims"][0]["entities"] == {"bonnie": "bonnie"}
  pairs = malls_pairs([{"NL": "All dogs are animals.", "FOL": "∀x (Dog(x) → Animal(x))"},
                       {"NL": "Broken.", "FOL": "∀x (Dog(x) → "},
                       {"NL": "All dogs are animals.", "FOL": "∀x (Dog(x) → Animal(x))"}])
  recs, stats = convert(pairs)
  assert stats["kept"] == 1 and stats["duplicate"] == 1 and sum(stats["dropped"].values()) == 1
  msgs = recs[0]["messages"]
  assert [m["role"] for m in msgs] == ["system", "user", "assistant"]
  assert parse_output(msgs[2]["content"])[0].claim == "All dogs are animals."


def test_folio_pairs():
  line = json.dumps({"story_id": 7, "premises": "All dogs are animals.\nRex is a dog.",
                     "premises-FOL": "∀x (Dog(x) → Animal(x))\nDog(rex)", "conclusion": "Rex is an animal.",
                     "conclusion-FOL": "Animal(rex)"})
  pairs = folio_pairs([line])
  assert [p[1] for p in pairs] == ["∀x (Dog(x) → Animal(x))", "Dog(rex)", "Animal(rex)"]
  assert all(p[2] == "folio:7" for p in pairs)


def test_silver_records_keep_only_matching_verdicts():
  client = FakeClient(lambda text: HINTON_REPLY if "Hinton" in text else "nonsense")
  items = [("S. E. Hinton is an American fantasy writer born in 1950.", ["e"], "TRUE"),
           ("S. E. Hinton is an American fantasy writer born in 1950.", ["e"], "FALSE"),
           ("Rex is a dog.", ["e"], "TRUE")]
  recs = silver_records(items, client, checker_from(HINTON_TRUE))
  assert len(recs) == 1 and recs[0]["source"] == "fever_silver"
