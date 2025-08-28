from typing import Dict, List, Tuple
import numpy as np
from fastapi import HTTPException

from ..ingest.chunking import chunk_text
from ..ingest.jd_parser import extract_requirements
from ..rag.embedder import Embedder
from ..rag.store import FaissStore
from ..llm.provider import call_llm
from ..utils.json_sanitizer import extract_json

# --- Tunables to keep prompts small/fast ---
MAX_REQ_PER_BATCH = 10          # handle at most 10 requirements per LLM call
MAX_TOTAL_REQ     = 60          # hard cap total reqs considered
MAX_SNIPPET_CHARS = 400         # truncate evidence
MAX_REQ_CHARS     = 250         # truncate requirement line
MAX_CV_CHUNKS     = 160         # cap CV length (prevents huge indexes)

PROMPT_TEMPLATE = """You are a senior technical recruiter.
You will receive a list of JOB REQUIREMENTS (bullets) and EVIDENCE snippets retrieved from the candidate's CV for each requirement.
Evaluate only with the provided evidence. Return STRICT JSON:

{{
  "overall_score": <0-100 integer>,
  "section_scores": {{
    "hard_skills": <0-100>,
    "experience": <0-100>,
    "soft_skills": <0-100>
  }},
  "good_matches": [{{"requirement": "str", "evidence": "str", "reason": "str"}}],
  "missing_requirements": ["str"],
  "missing_skills": ["str"],
  "improvement_suggestions": ["str"]
}}

JOB_REQUIREMENTS:
{requirements}

EVIDENCE_BY_REQUIREMENT:
{evidence}

Return ONLY JSON.
"""


def build_indexes(cv_chunks: List[str], embedder: Embedder) -> FaissStore:
    emb = embedder.encode(cv_chunks)
    store = FaissStore(dim=emb.shape[1])
    store.add(emb, cv_chunks)
    return store

def retrieve_evidence(requirements: List[str], store: FaissStore, embedder: Embedder, k:int=2) -> List[Tuple[str, List[Tuple[float,str]]]]:
    req_emb = embedder.encode(requirements)
    out = []
    for i, r in enumerate(requirements):
        q = req_emb[i:i+1]
        hits = store.search(q, k=k)
        out.append((r, hits))
    return out

def _format_evidence(evidence: List[Tuple[str, List[Tuple[float,str]]]]) -> str:
    lines = []
    for req, hits in evidence:
        req_short = (req[:MAX_REQ_CHARS] + "…") if len(req) > MAX_REQ_CHARS else req
        lines.append(f"- {req_short}")
        for score, snippet in hits:
            cleaned = snippet.replace("\n", " ").strip()
            if len(cleaned) > MAX_SNIPPET_CHARS:
                cleaned = cleaned[:MAX_SNIPPET_CHARS] + "…"
            lines.append(f"  • score={round(score,3)} | {cleaned}")
    return "\n".join(lines)

def _safe_extract_json(raw: str) -> Dict:
    try:
        return extract_json(raw)
    except Exception as e:
        # print a slice to server logs for debugging
        print("LLM RAW OUTPUT START =====")
        print(raw[:3000])
        print("LLM RAW OUTPUT END   =====")
        raise HTTPException(status_code=502, detail=f"LLM did not return valid JSON: {type(e).__name__}: {e}")

def _merge_batch_results(results: List[Dict]) -> Dict:
    if not results:
        return {
            "overall_score": 0,
            "section_scores": {"hard_skills":0,"experience":0,"soft_skills":0},
            "good_matches": [], "missing_requirements": [], "missing_skills": [],
            "improvement_suggestions": []
        }

    # Average numeric scores; concat lists, then deduplicate a bit.
    overall = int(round(sum(r.get("overall_score",0) for r in results)/len(results)))
    hs = int(round(sum(r.get("section_scores",{}).get("hard_skills",0) for r in results)/len(results)))
    exp = int(round(sum(r.get("section_scores",{}).get("experience",0) for r in results)/len(results)))
    soft = int(round(sum(r.get("section_scores",{}).get("soft_skills",0) for r in results)/len(results)))

    def _dedup(seq):
        seen = set(); out=[]
        for x in seq:
            key = str(x)
            if key not in seen:
                seen.add(key); out.append(x)
        return out

    good = _dedup([gm for r in results for gm in r.get("good_matches",[])])
    miss_req = _dedup([m for r in results for m in r.get("missing_requirements",[])])
    miss_sk = _dedup([m for r in results for m in r.get("missing_skills",[])])
    sugg = _dedup([s for r in results for s in r.get("improvement_suggestions",[])])

    return {
        "overall_score": overall,
        "section_scores": {"hard_skills":hs, "experience":exp, "soft_skills":soft},
        "good_matches": good,
        "missing_requirements": miss_req,
        "missing_skills": miss_sk,
        "improvement_suggestions": sugg
    }

def run_match(cv_text: str, jd_text: str, top_k: int = 2) -> Dict:
    # Cap CV size to keep retrieval fast
    cv_chunks = chunk_text(cv_text, max_chars=800)[:MAX_CV_CHUNKS]
    if not cv_chunks:
        cv_chunks = [cv_text.strip()]

    # Extract + cap requirements
    all_requirements = extract_requirements(jd_text)[:MAX_TOTAL_REQ]
    if not all_requirements:
        raise HTTPException(status_code=400, detail="No requirements detected in JD text.")

    # Build RAG index
    embedder = Embedder()
    store = build_indexes(cv_chunks, embedder)

    # Batch over requirements
    batch_results = []
    for i in range(0, len(all_requirements), MAX_REQ_PER_BATCH):
        batch_reqs = all_requirements[i:i+MAX_REQ_PER_BATCH]
        evidence = retrieve_evidence(batch_reqs, store, embedder, k=top_k)
        prompt = PROMPT_TEMPLATE.format(
            requirements="\n".join([f"- {r}" for r in batch_reqs]),
            evidence=_format_evidence(evidence)
        )
        raw = call_llm(prompt)
        data = _safe_extract_json(raw)
        batch_results.append({
            "overall_score": int(data.get("overall_score", 0)),
            "section_scores": data.get("section_scores", {"hard_skills":0,"experience":0,"soft_skills":0}),
            "good_matches": data.get("good_matches", []),
            "missing_requirements": data.get("missing_requirements", []),
            "missing_skills": data.get("missing_skills", []),
            "improvement_suggestions": data.get("improvement_suggestions", []),
        })

    merged = _merge_batch_results(batch_results)

    # Light heuristic augmentation: add naive missing-tokens pass
    cv_lower = " ".join(cv_chunks).lower()
    extra_missing = []
    for r in all_requirements:
        for w in [w.strip(",.():").lower() for w in r.split() if len(w) > 2]:
            if w.isalpha() and w not in cv_lower:
                extra_missing.append(w)
    # de-dup and clip
    seen=set(); extra=[w for w in extra_missing if not (w in seen or seen.add(w))]
    merged["missing_skills"] = list(dict.fromkeys([*merged.get("missing_skills",[]), *extra[:20]]))

    return merged
