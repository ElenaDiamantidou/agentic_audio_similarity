def aggregate_and_explain(scores, explanations, weights=None):
    """
    Combines similarity scores from multiple tools using specified weights,
    returns both the combined similarity score and a detailed explanation.
    """
    if weights is None:
        # Default: Equal weights
        weights = {k: 1/len(scores) for k in scores}
        print(weights)

    sim_score = sum(scores[k] * weights.get(k, 0) for k in scores)
    summary = []
    for k, v in scores.items():
        if v > 0.7:
            summary.append(f"{k} high")
        elif v > 0.5:
            summary.append(f"{k} medium")
    summary_text = ", ".join(summary) if summary else "no strong similarity"
    fields_expl = "\n".join(f"{k}: {v}" for k, v in explanations.items())
    explanation = (
        f"Weighted score: {sim_score:.2f}\n"
        f"Summary: {summary_text}\n"
        f"Feature details:\n{fields_expl}"
    )
    return {"similarity_score": sim_score, "explanation": explanation}
