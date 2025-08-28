# CV ↔ JD RAG Matcher (LLM-Scored)

A Retrieval-Augmented Generation pipeline that evaluates a CV against a Job Description using:
- **FAISS + Sentence-Transformers** for semantic retrieval
- **LLM** for **scoring, strengths, weaknesses, missing skills, and suggestions**
- **FastAPI** backend + **Streamlit** UI
- Pluggable LLM providers: **OpenAI**, **Ollama**, **Transformers**

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
cp .env.example .env
# set your keys/provider in .env

# Run API
uvicorn src.api.main:app --reload --port 8000

# In another terminal (UI)
export API_URL=http://localhost:8000
streamlit run src/ui/app.py
# cv-jd-rag-matcher
```

Or with Docker:
```
cp .env.example .env
# edit .env (OpenAI or point Ollama)
docker-compose up --build
```

Then open: http://localhost:8501

## How it works

- Parse & Chunk CV and JD text

- Embed + Index CV chunks via FAISS

- Retrieve top-k CV evidence for each JD requirement

- LLM Scoring Prompt (with evidence) → STRICT JSON

- UI renders scores, good matches, missing requirements/skills, suggestions

## Notes

- The LLM provides the final score (0–100) and explanations.

- We add a light heuristic for extra missing skills to help completeness.

- Works with OpenAI, or locally with Ollama/Transformers.
