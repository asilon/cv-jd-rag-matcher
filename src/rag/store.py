import faiss
import numpy as np
from typing import List, Tuple

class FaissStore:
    def __init__(self, dim: int):
        # Cosine via dot product on normalized vectors (IP)
        self.index = faiss.IndexFlatIP(dim)
        self.texts: List[str] = []

    def add(self, embeddings: np.ndarray, texts: List[str]):
        assert embeddings.shape[0] == len(texts)
        self.index.add(embeddings)
        self.texts.extend(texts)

    def search(self, query_vec: np.ndarray, k: int = 5) -> List[Tuple[float, str]]:
        D, I = self.index.search(query_vec, k)
        out = []
        for score, idx in zip(D[0], I[0]):
            if idx < 0:
                continue
            out.append((float(score), self.texts[idx]))
        return out

