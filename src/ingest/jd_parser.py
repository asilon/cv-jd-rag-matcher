import re
from typing import List

BULLET_RE = re.compile(r"^\s*(?:[-•*·]|\d+[.)])\s+(.*)$")

def extract_requirements(jd_text: str) -> List[str]:
    """
    Extract requirement bullets from a JD. Falls back to sentence-ish lines if no bullets.
    """
    lines = [l.strip() for l in jd_text.splitlines() if l.strip()]
    bullets = []
    for l in lines:
        m = BULLET_RE.match(l)
        if m:
            bullets.append(m.group(1))
    if bullets:
        return bullets
    # fallback: take non-trivial lines
    return [l for l in lines if len(l) > 20]

