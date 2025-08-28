import re
from typing import List

def chunk_text(text: str, max_chars: int = 800) -> List[str]:
    """Simple paragraph-based chunking with length guard."""
    text = text.replace("\r", "")
    paras = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    chunks, buf = [], ""
    for p in paras:
        if len(buf) + len(p) + 1 <= max_chars:
            buf = f"{buf}\n{p}".strip()
        else:
            if buf: chunks.append(buf)
            buf = p
    if buf: chunks.append(buf)
    return chunks

