"""Front: reads the user's text, writes a short reply, and splits the text into atomic, checkable claims.

Output: FrontOutput (schemas.py) via llm.with_structured_output(FrontOutput, method="json_schema").
Each claim then goes through the verifier on its own. Not built yet, and not on the API path.
"""
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable
