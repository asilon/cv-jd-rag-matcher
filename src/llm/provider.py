from typing import Dict, Any
from fastapi import HTTPException
import requests
from requests.adapters import HTTPAdapter, Retry

from ..config import (
    LLM_PROVIDER, OPENAI_API_KEY, OPENAI_MODEL,
    OLLAMA_BASE_URL, OLLAMA_MODEL, TRANSFORMERS_MODEL
)

def call_llm(prompt: str) -> str:
    try:
        if LLM_PROVIDER == "openai":
            if not OPENAI_API_KEY:
                raise HTTPException(status_code=400, detail="OPENAI_API_KEY not set but LLM_PROVIDER=openai.")
            return _call_openai(prompt)
        elif LLM_PROVIDER == "ollama":
            return _call_ollama(prompt)
        elif LLM_PROVIDER == "transformers":
            return _call_transformers(prompt)
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported LLM_PROVIDER={LLM_PROVIDER}")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"LLM provider '{LLM_PROVIDER}' failed: {type(e).__name__}: {e}")

def _session_with_retries():
    s = requests.Session()
    retries = Retry(
        total=2, backoff_factor=1.0,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["POST","GET"])
    )
    s.mount("http://", HTTPAdapter(max_retries=retries))
    s.mount("https://", HTTPAdapter(max_retries=retries))
    return s

def _call_ollama(prompt: str) -> str:
    sess = _session_with_retries()
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "format": "json",          # force JSON
        "keep_alive": "5m",
        "options": {
            "temperature": 0.1,
            "num_ctx": 8192       # if your model supports larger context
        }
    }
    try:
        r = sess.post(f"{OLLAMA_BASE_URL}/api/generate", json=payload, timeout=300)
        r.raise_for_status()
        data = r.json()
        return data.get("response", "")
    except requests.exceptions.ReadTimeout as e:
        raise HTTPException(status_code=502, detail=f"Ollama timed out after 300s. Try smaller batches or a smaller model.")
    except requests.exceptions.ConnectionError:
        raise HTTPException(status_code=502, detail=f"Could not connect to Ollama at {OLLAMA_BASE_URL}. Is it running and is model '{OLLAMA_MODEL}' pulled?")
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Ollama error: {e}")

def _call_openai(prompt: str) -> str:
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)
    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role":"system","content":"You are a precise evaluator. Return STRICT JSON only."},
            {"role":"user","content":prompt}
        ],
        response_format={"type":"json_object"},
        temperature=0.2,
    )
    return resp.choices[0].message.content

def _call_transformers(prompt: str) -> str:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
    model = AutoModelForCausalLM.from_pretrained(
        TRANSFORMERS_MODEL,
        device_map="auto",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
    tok = AutoTokenizer.from_pretrained(TRANSFORMERS_MODEL)
    inp = tok(prompt, return_tensors="pt")
    if torch.cuda.is_available():
        inp = {k:v.cuda() for k,v in inp.items()}
    out = model.generate(**inp, max_new_tokens=600, do_sample=False)
    text = tok.decode(out[0], skip_special_tokens=True)
    return text
