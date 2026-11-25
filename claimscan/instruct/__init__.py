"""Instruct LLMs around the verifier: the front (decomposer, Qwen3.5-9B, not built yet) and the rear
(aggregator, Gemma). Both run on local llama-servers configured in llamacpp/roles.yaml.

Where each layer lives:
  llamacpp/roles.yaml + llamacpp/scripts/serve.sh   llama.cpp: the servers (config only, no Python)
  model.py                                          LangChain: one ChatOpenAI client per role, front and rear
  schemas.py                                        pydantic shapes passed between the stages
  prompts.py                                        system prompts
  front_decomposer.py / rear_aggregator.py          logic; each takes the LLM as an argument
"""
