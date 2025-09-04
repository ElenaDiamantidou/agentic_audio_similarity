import numpy as np

from data.loader import DaTACOSDataLoader
from src.tools.metadata import create_metadata_embedding

class SimilarityTool:
    def compute(self, perf_id1, perf_id2):
        raise NotImplementedError

    def explain(self, perf_id1, perf_id2):
        raise NotImplementedError


class HPCPSimilarity(SimilarityTool):
    def __init__(self,
                 loader: DaTACOSDataLoader):
        self.loader = loader

    def compute(self, perf_id1, perf_id2):
        h1 = self.loader.get_hpcp_embedding(perf_id1)
        h2 = self.loader.get_hpcp_embedding(perf_id2)
        return cosine_similarity(h1, h2)

    def explain(self, perf_id1, perf_id2):
        score = self.compute(perf_id1, perf_id2)
        return f"HPCP similarity score: {score:.2f}"


class CENSSimilarity(SimilarityTool):
    def __init__(self,
                 loader: DaTACOSDataLoader):
        self.loader = loader

    def compute(self, perf_id1, perf_id2):
        c1 = self.loader.get_cens_embedding(perf_id1)
        c2 = self.loader.get_cens_embedding(perf_id2)
        return cosine_similarity(c1, c2)

    def explain(self, perf_id1, perf_id2):
        score = self.compute(perf_id1, perf_id2)
        return f"CENS similarity score: {score:.2f}"

class SemanticMetadataSimilarity(SimilarityTool):
    def __init__(self,
                 loader: DaTACOSDataLoader):
        self.loader = loader
        self.fields = ["perf_artist", "work_artist", "work_title", "release_year"]

    def compute(self, perf_id1, perf_id2):
        m1 = self.loader.get_metadata(perf_id1)
        m2 = self.loader.get_metadata(perf_id2)
        sims = []
        for field in self.fields:
            txt1, txt2 = m1.get(field, ""), m2.get(field, "")
            if txt1 and txt2:
                emb1, emb2 = create_metadata_embedding(txt1, txt2)
                sims.append(cosine_similarity(emb1, emb2))
        return np.mean(sims) if sims else 0.0

    def explain(self, perf_id1, perf_id2):
        m1 = self.loader.get_metadata(perf_id1)
        m2 = self.loader.get_metadata(perf_id2)
        report = []
        for field in self.fields:
            txt1, txt2 = m1.get(field, ""), m2.get(field, "")
            if txt1 and txt2:
                emb1, emb2 = create_metadata_embedding(txt1, txt2)
                sim = cosine_similarity(emb1, emb2)
                verdict = "high" if sim > 0.8 else "medium" if sim > 0.5 else "low"
                report.append(f"{field}: {sim:.2f} ({verdict})")
            else:
                report.append(f"{field}: [missing value]")
        return "\n".join(report)


def cosine_similarity(vector_1: np.ndarray,
                      vector_2: np.ndarray,
                      eps: float = 1e-8) -> float:
    """
    Calculate cosine similarity between two vectors.

    Args:
        vector_1: First vector
        vector_2: Second vector
        eps: Small epsilon value to avoid division by zero

    Returns:
        Cosine similarity value between -1 and 1
    """
    if vector_1.shape != vector_2.shape:
        raise ValueError(f"Vector shapes must match: {vector_1.shape} != {vector_2.shape}")

    norm1 = np.linalg.norm(vector_1)
    norm2 = np.linalg.norm(vector_2)

    # Use epsilon to avoid division by zero
    return float(np.dot(vector_1, vector_2) / (max(norm1, eps) * max(norm2, eps)))
