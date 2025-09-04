from sentence_transformers import SentenceTransformer


def compare_strings(a, b):
    """
    Case-insensitive, trimmed string comparison, handles None
    """
    if not a or not b:
        return False
    return a.strip().lower() == b.strip().lower()


def compare_years(y1, y2, tolerance=2):
    """
    Compare years within tolerance
    """
    try:
        y1, y2 = int(y1), int(y2)
        return abs(y1 - y2) <= tolerance
    except Exception:
        return False

def metadata_similarity(meta1: dict, meta2: dict) -> float:
    score, total = 0, 3

    if compare_strings(meta1.get("perf_artist"), meta2.get("perf_artist")):
        score += 1
    if compare_strings(meta1.get("work_artist"), meta2.get("work_artist")):
        score += 1
    if compare_years(meta1.get("release_year"), meta2.get("release_year")):
        score += 1
    else:
        total -= 1  # Don't penalize if year missing

    return score / total if total else 0


def load_sentence_model(model_name: str = "all-MiniLM-L6-v2"):
    return SentenceTransformer(model_name)


def create_metadata_embedding(txt1: str,
                              txt2: str) -> tuple:
    model = load_sentence_model()

    emb1 = model.encode(txt1)
    emb2 = model.encode(txt2)

    return emb1, emb2
