# ClaimScan (RAG + RL)

Right now this is a FEVER fact-checker that works. You give it a claim, it pulls evidence from a 25.2M-sentence Wikipedia corpus, scores that evidence, and a DeBERTa NLI verifier says TRUE, FALSE or NOT ENOUGH INFO. A Gemma rear aggregator then explains the verdict.

I'm still going to do RAG + RL, but I really am going to pivot the idea now a bit towards something I can actually use. I find often when I'm reading papers, I want quick definitions over highlighted words with AI. Like 1-2 sentence definitions. I highlight over the word, ask for a definition/description, and it gives it to me. That would make reading papers SO easy for me. It's also the perfect project for me to practice my agentic coding + workflow + sys design.

## What works now

```
claim
  -> retriever: hybrid FAISS (bge-small) + BM25 (Lucene, no RM3), fused with RRF (k=60)
  -> reranker: Cohere rerank-v3.5 (top 10 of 100)
  -> neurosymbolic layer: retrieval rank prior x claim-type rules
  -> verifier: DeBERTa-v3 NLI (MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli) -> TRUE / FALSE / NOT ENOUGH INFO
  -> rear aggregator: Gemma-3-4b (LoRA SFT + DPO), writes the explanation
```

- **Neurosymbolic layer** (`claimscan/neurosymbolic_layer/`): whole-word rules (hedge words, word overlap, page-title bonus, claim-type rules), clamped so no single keyword can sink or dominate a sentence. The goal was for it to work decently well without being too restrictive or too useless either.
- **NEI rule:** if the verifier's P(NEI) >= 0.2, a TRUE/FALSE becomes NOT ENOUGH INFO (`verifier.nei_prob_threshold`). We should strictly be using the confidence the verifier gives on NEI directly.
- **Rear aggregator:** we need an LLM in the back to pick up the responses and give an answer. The rear takes from the verifier the softmax output and the top k evidence used to come to that conclusion. It used to have veto power (it could only turn TRUE/FALSE into NEI), but it vetoed too many correct verdicts, so I took away its veto power. Now the LLM only explains. It's off by default in the API (`aggregator.enabled: false`) and fails open: if the server is down you just get the verifier's verdict.

**Numbers** (`/fever-full-test 300`, 300 FEVER validation claims, k=30): label accuracy 0.67 with the NEI rule (0.65 without it, FEVER score 0.467). Per gold label the verifier gets TRUE 91%, FALSE 76%, NEI 28% (before the NEI rule), so most errors are NEI claims called TRUE/FALSE. Retrieval isn't the limit. If data is the issue, it's data everywhere; in the retriever and in the fine tuning stages. We probably need to train on a LOT more data.

## Where this is going

The pipeline is simple:

```
prompt
  -> Qwen decomposer        (the shepherd: decompose to claims, write the queries)
  -> retriever              (data facts are stored here)
  -> reranker               (you don't need DeBERTa for the reranker)
  -> loop until we determine it is ok   (enough evidence? change k or search again, max 3 loops)
  -> verifier               (optional: only if the Qwen decomposer determined it to be a verification task)
  -> Gemma rear aggregator  (takes in all and outputs a response)
```

Gemma for back, Qwen for front. The Qwen decomposer can still just be a small Qwen agent, just the 4B model inheriting more tasks. This makes sense since it's not executing complex tasks, and we know small models can work with RL anyways. This also acts a lot more like a traditional RAG pipeline, and it also enables us to scale more data (papers, definitions, data, knowledge, all sorts of things, not just FEVER).

We are pivoting this towards **explanation** rather than claim verification.

One thing I really wanted to do was to have more semantic ideas in this. The neurosymbolic layer is a start, but I wanted to explore how semanticity/NLP ideas can also be used:
- **Natural logic + FOL, both, layered.** Natural logic in RAG, i.e. like formal logic, first order models, or learning semantic meaning of words. The claim breaks down into formal logic, and we see if the evidence supports that. We train a DeBERTa to predict the logical relations (same, entailment, negation, unrelated), which is smarter than my own neurosymbolic layer, and it can be extended to other semantic meanings (quantifiers and set theory). Plan: `docs/OVERHAUL.md` (on the `overhaul` branch, not pushed yet).
- **RL as a librarian.** My own idea: the Qwen model learns to write the searches, rewarded by the FEVER metric, and I suspect it can LEARN real semantic meaning by reasoning. LoRA SFT first, then GRPO. I may do this project in a hackathon where I have a lot more compute, so scaling this up is the next step.

## Layout

```
claimscan/
  data/                 FEVER SQLite store (fever_db.py), post-training data
  retrieval/            FAISS + BM25 retrievers, RRF hybrid, index builders
  rerank/               Cohere / cross-encoder / OpenAI rerankers
  neurosymbolic_layer/  claim-type rules and evidence scoring
  evidence/             evidence selection + SQLite sentence store
  verifier/             DeBERTa verifiers (HF NLI default, our old checkpoint), training
  instruct/             front decomposer + rear aggregator (LangChain), posttraining (SFT + DPO)
  eval/                 recall@k, retriever benchmark, FEVER metric
  graph/                makes the plots in graphs/
  api/                  FastAPI service
  pipeline.py           load_services(), fact_check_detailed()
configs/default.yaml    all settings
llamacpp/               llama.cpp build/serve scripts, one config per LLM role
docker/                 Docker image for the API
```

## Run it

`requirements.txt` is just what the API needs; `requirements-dev.txt` adds training, eval, BM25, graphs and tests. BM25 (pyserini) also needs a **Java 21 JDK**: `sudo apt install openjdk-21-jdk-headless`.

```bash
git clone https://github.com/mkhi238/FEVER_RAG_LLM_Model.git && cd FEVER_RAG_LLM_Model
source scripts/env.sh                   # caches and temp on D:/E:, not C:
uv python install 3.12 && uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements-dev.txt
uv pip install --python .venv/bin/python --no-deps -e .
```

Data, indexes and models live outside the repo, under `artifacts_root` in `configs/default.yaml` (override with `CLAIMSCAN_ARTIFACTS=/path`). The Cohere key goes in the repo-root `.env` (`COHERE_API_KEY=...`, gitignored).

```bash
source .venv/bin/activate && source scripts/env.sh
uvicorn claimscan.api.main:app        # POST /fact-check {"claim": "..."}, GET /health
python -m pytest -q                   # unit tests (also run by CI on every push)
python scripts/verifier_smoke.py      # just the verifier
```

In Claude Code: `/unit-tests` checks the code is correct, `/fever-full-test 300` checks how accurate the system is. The API runs the same `fact_check_detailed` as the benchmark (k=30), so benchmark numbers are what the API does. Benchmark and training runs are logged to MLflow:

```bash
mlflow ui --backend-store-uri sqlite:////mnt/d/crisis-claim-analysis/artifacts/mlflow/mlflow.db
```

Docker: `docker compose up --build -d`, then see [docker/README.md](docker/README.md). The llama.cpp side is walked through in [llamacpp/README.md](llamacpp/README.md).

## Graphs

![Retriever recall_set@k](graphs/retriever_recall_set.png)

Share of the 200 SUPPORTS/REFUTES claims with a full gold evidence set in the top k, which is what the FEVER score needs. From the benchmark run before the full FAISS index (shards 0-174 only). All graphs: [`graphs/`](graphs/README.md), rebuilt with `/graphs`.

## Known limits

- NEI is the weak spot (28% before the NEI rule, 39% after). Several "NEI" claims are really supported by the retrieved sentences (FEVER label noise).
- The full FAISS index (39 GB, on disk) is slow cold through the API: 30-110 s per query from D:.
- The trained Gemma DPO adapter copies its SFT targets (its confidence is always 0.95/0.05).
