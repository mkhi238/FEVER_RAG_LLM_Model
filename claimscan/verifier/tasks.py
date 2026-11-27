"""Per-task DeBERTa NLI checks for the 5 actions (Define, Explain, Verify, Question, Example), docs/NATLOG_FOL.md.

Every task is NLI: premise = what the text is checked against, hypothesis = a task template filled with the slots.
  task      premise            hypothesis template                                 success ops   labels
  verify    evidence           {claim}                                             ≡ ⊑           TRUE / FALSE / NOT ENOUGH INFO
  define    evidence           {term} is {definition}                              ≡ ⊑           DEFINED / WRONG / NOT FOUND
  example   evidence           {instance} is an example of {concept}               ≡ ⊑           EXAMPLE / NOT AN EXAMPLE / UNKNOWN
  explain   the explanation    {original}  (each atom of it, with FOL)             ≡ ⊑           FAITHFUL / CONTRADICTS / INCOMPLETE
  question  evidence           The answer to "{question}" is {answer}.             ≡ ⊑           ANSWERED / CONTRADICTED / NOT IN EVIDENCE
The decision per hypothesis is the Verify decision as is (fol/check.py VerifierChecker: pipeline.verify, argmax,
NEI if max prob < 0.5 or P(NEI) >= verifier.nei_prob_threshold). Explain also runs NLI the other way, so its operator
can be ≡ (both entail) rather than ⊑ (the explanation adds something).

Models (natlog.tasks in default.yaml), one shared deberta-v3 base by default (RAM 15 GB / VRAM 6 GB):
  tasks.<task>.adapter: null   zero-shot: the base NLI model (natlog.base_model, null = verifier.inference_model)
  tasks.<task>.adapter: <dir>  a LoRA adapter (train_task_adapter.py) on the shared base; switching tasks switches
                               the active adapter, the base weights are loaded once
  tasks.<task>.model: <name>   a fully separate model for that task (HF name or dir): the config switch to
                               "5 different DeBERTa models"
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from claimscan.neurosymbolic_layer.fol.check import LABELS, VerifierChecker

TASK_NAMES = ("verify", "define", "explain", "question", "example")


@dataclass(frozen=True)
class TaskSpec:
  name: str
  premise: str                    # what the premise is (for docs / the API)
  template: str                   # str.format template over the slots
  labels: tuple[str, str, str]    # task names for TRUE, FALSE, NOT ENOUGH INFO
  success_ops: tuple[str, ...] = ("≡", "⊑")
  bidirectional: bool = False     # also run NLI hypothesis -> premise (≡ vs ⊑)


TASKS: dict[str, TaskSpec] = {
  "verify": TaskSpec("verify", "evidence", "{claim}", ("TRUE", "FALSE", "NOT ENOUGH INFO")),
  "define": TaskSpec("define", "evidence", "{term} is {definition}", ("DEFINED", "WRONG", "NOT FOUND")),
  "example": TaskSpec("example", "evidence", "{instance} is an example of {concept}",
                      ("EXAMPLE", "NOT AN EXAMPLE", "UNKNOWN")),
  "explain": TaskSpec("explain", "explanation", "{original}", ("FAITHFUL", "CONTRADICTS", "INCOMPLETE"),
                      bidirectional=True),
  "question": TaskSpec("question", "evidence", "The answer to \"{question}\" is {answer}.",
                       ("ANSWERED", "CONTRADICTED", "NOT IN EVIDENCE")),
}


def hypothesis(task: str, **slots) -> str:
  spec = TASKS[task]
  try:
    return spec.template.format(**slots).strip()
  except KeyError as e:
    raise ValueError(f"task {task!r} needs slot {e.args[0]!r} (template {spec.template!r})") from None


@dataclass
class TaskResult:
  task: str
  hypothesis: str
  label: str                      # TRUE | FALSE | NOT ENOUGH INFO (the shared states)
  task_label: str                 # the task's name for it, e.g. DEFINED
  op: str
  success: bool                   # op in the task's success_ops
  evidence_idx: int | None
  probs: dict | None


class _View:
  """One task's model with HFNLIVerifier's interface (encode_text, __call__ -> logits in LABEL_MAP order). With a
  shared PEFT base, it activates its adapter (or disables adapters for zero-shot) before each forward pass."""
  def __init__(self, verifier, peft=None, adapter: str | None = None):
    self.verifier, self.peft, self.adapter = verifier, peft, adapter

  def encode_text(self, claim, evidence):
    return self.verifier.encode_text(claim, evidence)

  def __call__(self, input_ids, attention_mask):
    if self.peft is None:
      return self.verifier(input_ids, attention_mask)
    if self.adapter is None:
      with self.peft.disable_adapter():
        return self.verifier(input_ids, attention_mask)
    self.peft.set_adapter(self.adapter)
    return self.verifier(input_ids, attention_mask)


def _default_loader(name: str):
  from claimscan.verifier.hf_nli import HFNLIVerifier
  return HFNLIVerifier(name).eval()


class TaskVerifiers:
  """Loads lazily: the base on first use, each adapter / separate model on first use of its task.

  settings: the natlog: section (fol/verify.py settings()); base_model: the shared base name; artifacts_root: where
  relative adapter / model dirs live. loader(name) -> HFNLIVerifier-like, for tests."""
  def __init__(self, settings: dict, base_model: str, artifacts_root: str | Path = ".", loader=None,
               nei_prob_threshold: float | None = None):
    self.settings, self.base_name = settings, settings.get("base_model") or base_model
    self.root, self.loader = Path(artifacts_root), loader or _default_loader
    self.nei_cut = nei_prob_threshold
    self._base, self._peft, self._separate, self._views = None, None, {}, {}

  @classmethod
  def from_config(cls, cfg=None, loader=None):
    from claimscan.neurosymbolic_layer.fol.verify import _cfg, settings
    cfg = _cfg(cfg)
    return cls(settings(cfg), cfg.verifier_inference_model, cfg.artifacts_root, loader, cfg.verifier_nei_prob_threshold)

  def task_cfg(self, task: str) -> dict:
    if task not in TASKS:
      raise ValueError(f"unknown task {task!r}; choose from {TASK_NAMES}")
    return (self.settings.get("tasks") or {}).get(task) or {}

  def _path(self, p) -> str:
    p = Path(p)
    return str(p if p.is_absolute() else self.root / p)

  def _shared_base(self):
    if self._base is None:
      self._base = self.loader(self.base_name)
    return self._base

  def _attach(self, task: str, adapter_dir: str):
    from peft import PeftModel
    base = self._shared_base()
    if self._peft is None:
      self._peft = PeftModel.from_pretrained(base.nli, adapter_dir, adapter_name=task).eval()
      base.nli = self._peft
    elif task not in self._peft.peft_config:
      self._peft.load_adapter(adapter_dir, adapter_name=task)
    # zero-shot views made before the first adapter was attached must now disable it
    for v in self._views.values():
      if v.verifier is base:
        v.peft = self._peft

  def model(self, task: str):
    """HFNLIVerifier-like model for `task` (usable by VerifierChecker and natlog ops.nli_from_verifier)."""
    if task in self._views:
      return self._views[task]
    c = self.task_cfg(task)
    if c.get("model"):
      name = self._path(c["model"]) if Path(self._path(c["model"])).exists() else c["model"]
      if name not in self._separate:
        self._separate[name] = self.loader(name)
      view = _View(self._separate[name])
    elif c.get("adapter"):
      self._attach(task, self._path(c["adapter"]))
      view = _View(self._shared_base(), self._peft, task)
    else:
      view = _View(self._shared_base(), self._peft, None)
    self._views[task] = view
    return view

  def check(self, task: str, evidence: list[str], **slots) -> TaskResult:
    """One hypothesis from the task template vs the premise (evidence sentences, or [explanation] for explain)."""
    spec = TASKS[task]
    hyp = hypothesis(task, **slots)
    model = self.model(task)
    c = VerifierChecker(model, 0.5, self.nei_cut)(hyp, evidence)
    op = c.op
    if spec.bidirectional and c.label == "TRUE" and evidence:
      back = VerifierChecker(model, 0.5, self.nei_cut, attribute=False)(" ".join(evidence[:3]), [hyp])
      op = "≡" if back.label == "TRUE" else "⊑"
    return TaskResult(task, hyp, c.label, spec.labels[LABELS.index(c.label)], op, op in spec.success_ops,
                      c.evidence_idx, c.probs)
