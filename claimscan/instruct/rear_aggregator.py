"""Rear: combine per-claim verdicts + evidence from the LangGraph loop into the final answer."""
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable
