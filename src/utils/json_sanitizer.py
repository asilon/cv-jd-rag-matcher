import json
import re
from typing import Any, Dict

def extract_json(text: str) -> Dict[str, Any]:
    """
    Extract the first top-level JSON object from text and parse it.
    Robust against model adding prose before/after.
    """
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in LLM output.")
    raw = text[start:end+1]
    # Remove trailing commas in arrays/objects (common LLM hiccup)
    raw = re.sub(r",\s*([}\]])", r"\1", raw)
    return json.loads(raw)

