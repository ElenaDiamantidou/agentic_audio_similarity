from src.agent import MusicSimilarityAgent


if __name__ == "__main__":
    agent = MusicSimilarityAgent()
    inputs = {
        "perf_id1": "P_747569",
        "perf_id2": "P_708875"
    }
    output = agent.invoke(inputs)
    print(f"Similarity Score: {output['similarity_score']:.4f}")
    print(f"Explanation: {output['explanation']}")
