from typing import Dict, Any, ClassVar, List

from langchain.chains.base import Chain

from data.loader import DaTACOSDataLoader
from src.tools.aggregator import aggregate_and_explain

from src.tools.similarity import HPCPSimilarity, CENSSimilarity, SemanticMetadataSimilarity

# Initialize loader with paths to your downloaded data folders
loader = DaTACOSDataLoader(
    meta_path="data/da-tacos/da-tacos_metadata/da-tacos_benchmark_subset_metadata.json",
    features_cens_path="data/da-tacos/da-tacos_benchmark_subset_cens",
    features_hpcp_path="data/da-tacos/da-tacos_benchmark_subset_hpcp"
)


class MusicSimilarityAgent(Chain):
    @property
    def input_keys(self):
        return ["perf_id1", "perf_id2"]

    @property
    def output_keys(self):
        return ["similarity_score", "explanation"]

    def _call(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        pid1 = inputs["perf_id1"]
        pid2 = inputs["perf_id2"]

        # compute scores and explanation here
        result = self.predict_similarity(pid1, pid2)
        return result

    def compute_all_similarities(self,
                                 perf_id1: str,
                                 perf_id2: str):
        similarity_tools = {
            "hpcp": HPCPSimilarity(loader),
            "cens": CENSSimilarity(loader),
            "meta_sem": SemanticMetadataSimilarity(loader)
            # Add more tools as needed
        }
        meta1 = loader.get_metadata(perf_id1)
        meta2 = loader.get_metadata(perf_id2)
        print("-------- Metadata for performance_id1 --------")
        print(f"title: {meta1.get('work_title')}")
        print(f"artist: {meta1.get('work_artist')}")
        print(f"year: {meta1.get('release_year')}")

        print("-------- Metadata for performance_id2 --------")
        print(f"title: {meta2.get('work_title')}")
        print(f"artist: {meta2.get('work_artist')}")
        print(f"year: {meta2.get('release_year')}\n")

        scores, explanations = {}, {}
        for name, tool in similarity_tools.items():
            scores[name] = tool.compute(perf_id1, perf_id2)
            explanations[name] = tool.explain(perf_id1, perf_id2)
        return scores, explanations


    def predict_similarity(self, perf_id1, perf_id2):
        scores, explanations = self.compute_all_similarities(perf_id1, perf_id2)
        weights = {"hpcp": 0.35, "cens": 0.35, "meta_sem": 0.3}
        result = aggregate_and_explain(scores, explanations, weights=weights)
        return result