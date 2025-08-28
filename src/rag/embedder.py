from sentence_transformers import SentenceTransformer
import numpy as np

class Embedder:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def encode(self, texts):
        emb = self.model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
        return np.array(emb, dtype="float32")

