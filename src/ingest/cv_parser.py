import re
from typing import List

BULLET_RE = re.compile(r"^\s*(?:[-•*·]|\u2022)\s+(.*)$")

def extract_cv_bullets(cv_text: str) -> List[str]:
    """
    Extract bullet-like lines from a CV to use as experience evidence.
    """
    lines = [l.strip() for l in cv_text.splitlines() if l.strip()]
    bullets = []
    for l in lines:
        m = BULLET_RE.match(l)
        if m:
            bullets.append(m.group(1))
    # fallback: long lines as "bullets"
    if not bullets:
        bullets = [l for l in lines if len(l) > 30]
    return bullets

