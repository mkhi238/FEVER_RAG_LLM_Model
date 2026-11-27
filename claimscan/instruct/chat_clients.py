"""LangChain clients for the instruct roles (front_facer, decomposer, aggregator), one per role in llamacpp/roles.yaml.

This file only connects to the servers. It runs no model (llama-server does that) and holds no prompts or logic
(those live in prompts.py, front_decomposer.py and rear_aggregator.py).

A llamacpp role is reached at http://127.0.0.1:<port>/v1, and a role with `shares:` uses the other role's server.
llama-server ignores the API key, so any string works. Pass base_url to override roles.yaml: the API does this so
CLAIMSCAN_AGGREGATOR_URL / docker-compose can point at the `llama` service.

Usage:
  chat = build_chat("aggregator")                                          # plain ChatOpenAI
  llm = chat.with_structured_output(VetoOutput, method="json_schema")      # what aggregate() takes
Use method="json_schema": llama-server turns the schema into a grammar, so the JSON always parses.
Gemma's tool calling (the default method) is unreliable.
"""
import os

import yaml
from langchain_openai import ChatOpenAI

ROLES_FILE = os.path.join(os.path.dirname(__file__), "..", "..", "llamacpp", "roles.yaml")


def build_chat(role: str, base_url: str | None = None, timeout: float = 30.0) -> ChatOpenAI:
  """ChatOpenAI for `role` in roles.yaml (follows `shares:`), temperature 0, no retries (the caller fails open).

  base_url, if given, wins over roles.yaml.
  
  roles = {
      "aggregator": {
          "port": 8083,
          "model": "gemma"
      },
      "front_facer": {
          "shares": "aggregator"
      },
      "decomposer": {
          "shares": "aggregator"
      }
  }
  """
  with open (ROLES_FILE, encoding='utf-8') as f:
    roles = yaml.safe_load(f)["roles"]

  config = roles[role] #role = "front_facer" or "decomposer" or "aggregator"
  shares = config.get("shares")
  if shares:
    config = roles[shares]
  model = config["model"]
  url = base_url or f"http://127.0.0.1:{config['port']}/v1"
  llm = ChatOpenAI(model=model, base_url=url, api_key="not-needed",
                    temperature=0, max_retries=0, timeout=timeout)
  return llm
  
