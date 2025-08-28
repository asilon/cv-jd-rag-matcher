from fastapi import FastAPI
from pydantic import BaseModel
from ..pipeline.llm_only import run_match_llm
from ..pipeline.match_pipeline import run_match
from ..config import LLM_PROVIDER, OPENAI_MODEL, OLLAMA_MODEL, TRANSFORMERS_MODEL

app = FastAPI(title="CV-JD RAG Matcher", version="1.0")

class MatchRequest(BaseModel):
    cv_text: str
    jd_text: str
    top_k: int = 3

@app.get("/")
def root():
    return {"ok": True, "service": "cv-jd-rag-matcher", "version": "1.0"}

@app.get("/health")
def health():
    return {"status": "ok"}




class MatchRequest(BaseModel):
    cv_text: str
    jd_text: str
    top_k: int = 3

@app.get("/config")
def config():
    model = None
    if LLM_PROVIDER == "openai":
        model = OPENAI_MODEL
    elif LLM_PROVIDER == "ollama":
        model = OLLAMA_MODEL
    elif LLM_PROVIDER == "transformers":
        model = TRANSFORMERS_MODEL
    return {"llm_provider": LLM_PROVIDER, "model": model}

@app.post("/score_llm")
def score_llm(req: MatchRequest):
    # top_k ignored here; kept for UI compatibility
    return run_match_llm(req.cv_text, req.jd_text)

# keep old endpoint if you still want it
@app.post("/score")
def score(req: MatchRequest):
    return run_match(req.cv_text, req.jd_text, top_k=req.top_k)
