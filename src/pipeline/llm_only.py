# src/pipeline/llm_only.py
import re
from typing import List, Dict
from fastapi import HTTPException
from ..llm.provider import call_llm
from ..utils.json_sanitizer import extract_json

# --------- simple helpers (no FAISS / no RAG) ---------
BULLET_RE = re.compile(r"^\s*(?:[-•*·]|\d+[.)])\s+(.*)$")

def extract_bullets(text: str) -> List[str]:
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    bullets = []
    for l in lines:
        m = BULLET_RE.match(l)
        if m:
            bullets.append(m.group(1).strip())
    if bullets:
        return bullets
    # fallback: heuristic "sentences"
    rough = re.split(r"(?<=[.!?])\s+\n?|\n{2,}", text)
    return [s.strip() for s in rough if len(s.strip()) > 20][:120]

def tokenize_words(text: str) -> List[str]:
    # lowercase tokens, letters+digits only, len>=3
    return re.findall(r"[A-Za-z0-9][A-Za-z0-9\-\+_.]{2,}", text.lower())

def sanitize_list_str(items: List[str]) -> List[str]:
    cleaned = []
    seen = set()
    for it in items:
        it = (it or "").strip()
        if not it:
            continue
        # remove placeholder-y garbage the model might emit
        if it.lower() in {"str", "...", "[omitted]", "n/a"}:
            continue
        key = it.lower()
        if key not in seen:
            seen.add(key)
            cleaned.append(it)
    return cleaned

# --------- LLM prompt (NO {type: str} placeholders) ---------
PROMPT = """You are an expert recruiter. You will receive two inputs:

1) JOB_DESCRIPTION_BULLETS: a list of job requirements or responsibilities (each item is a single bullet).
2) CV_BULLETS: a list of candidate CV lines (each item is a single bullet or sentence from the CV).

Using ONLY these bullets (do not invent content), produce STRICT JSON with this exact structure and keys:

{
  "overall_score": 0-100,
  "section_scores": {
    "hard_skills": 0-100,
    "experience": 0-100,
    "soft_skills": 0-100
  },
  "good_matches": [
    { "jd_bullet": "<exact JD bullet>", "cv_evidence": "<exact CV bullet or sentence>", "reason": "<short why this matches>" }
  ],
  "missing_requirements": ["<JD bullet not covered>", "..."],
  "missing_skills": ["<skills present in JD but missing in CV>", "..."],
  "improvements": ["<short actionable suggestions>", "..."]
}

Rules:
- Use exact substrings from JD/CV lists only. Quote them verbatim for jd_bullet and cv_evidence.
- Do NOT output placeholder words like "str" or "...".
- "missing_requirements" must be a subset of JOB_DESCRIPTION_BULLETS.
- "missing_skills" must be concrete tokens or phrases that appear in the JD bullets but not in the CV bullets.
- Keep lists concise (max ~10 items each). Keep "reason" short.
- Return ONLY the JSON object, nothing else.

JOB_DESCRIPTION_BULLETS:
<<JD>>

CV_BULLETS:
<<CV>>
"""


def _clip_list(xs: List[str], max_len: int, max_chars: int) -> List[str]:
    out = []
    for x in xs[:max_len]:
        x = x.replace("\n", " ").strip()
        if len(x) > max_chars:
            x = x[:max_chars] + "…"
        out.append(x)
    return out

def _postprocess(data: Dict, jd_bullets: List[str], cv_bullets: List[str]) -> Dict:
    jd_set = {b.strip() for b in jd_bullets}
    cv_text = " ".join(cv_bullets).lower()

    # sanitize arrays
    data["good_matches"] = [
        gm for gm in data.get("good_matches", [])
        if isinstance(gm, dict)
        and gm.get("jd_bullet") in jd_set
        and any(gm.get("cv_evidence","") in c for c in cv_bullets)
        and gm.get("jd_bullet").strip().lower() not in {"str","..."}
        and gm.get("cv_evidence","").strip().lower() not in {"str","..."}
    ]

    # keep only JD bullets we actually provided
    data["missing_requirements"] = [
        b for b in sanitize_list_str(data.get("missing_requirements", []))
        if b in jd_set
    ]

    # keep only skills appearing in JD and NOT in CV
    jd_tokens = set(tokenize_words(" ".join(jd_bullets)))
    cv_tokens = set(tokenize_words(" ".join(cv_bullets)))
    miss = []
    for s in sanitize_list_str(data.get("missing_skills", [])):
        toks = set(tokenize_words(s))
        # accept if at least one token appears in JD tokens, and none appear in CV tokens
        if toks & jd_tokens and not (toks & cv_tokens):
            miss.append(s)
    # de-dup
    seen = set(); miss2 = []
    for s in miss:
        key = s.lower()
        if key not in seen:
            seen.add(key); miss2.append(s)
    data["missing_skills"] = miss2[:15]

    # clip improvements
    data["improvements"] = sanitize_list_str(data.get("improvements", []))[:10]

    # numeric guards
    ss = data.get("section_scores", {})
    data["section_scores"] = {
        "hard_skills": int(ss.get("hard_skills", 0)),
        "experience": int(ss.get("experience", 0)),
        "soft_skills": int(ss.get("soft_skills", 0)),
    }
    data["overall_score"] = int(data.get("overall_score", 0))
    return data

def run_match_llm(cv_text: str, jd_text: str) -> Dict:
    jd_bullets = extract_bullets(jd_text)
    cv_bullets = extract_bullets(cv_text)
    if not jd_bullets:
        raise HTTPException(status_code=400, detail="No requirements detected in JD.")
    if not cv_bullets:
        raise HTTPException(status_code=400, detail="No content detected in CV.")

    # keep prompts tight to be fast & avoid timeouts
    jd_bullets_small = _clip_list(jd_bullets, max_len=40, max_chars=220)
    cv_bullets_small = _clip_list(cv_bullets, max_len=120, max_chars=220)

    prompt = (PROMPT
              .replace("<<JD>>", "\n".join(f"- {b}" for b in jd_bullets_small))
              .replace("<<CV>>", "\n".join(f"- {b}" for b in cv_bullets_small)))
    # then call the LLM as before
    raw = call_llm(prompt)
    try:
        data = extract_json(raw)
    except Exception as e:
        # show first 2k chars in server log to debug the LLM
        print("LLM RAW OUTPUT START =====")
        print(raw[:2000])
        print("LLM RAW OUTPUT END   =====")
        raise HTTPException(status_code=502, detail=f"LLM did not return valid JSON: {type(e).__name__}: {e}")

    return _postprocess(data, jd_bullets_small, cv_bullets_small)
