import json
import h5py
import numpy as np
from typing import Dict


def load_metadata(meta_path: str) -> Dict:
    """
    Load DaTACOS metadata from a JSON file.

    Args:
        meta_path: Path to the metadata JSON file

    Returns:
        Dictionary containing the metadata
    """

    with open(meta_path, "r") as f:
        return json.load(f)


def flatten_metadata(da_tacos_meta: Dict) -> Dict[str, Dict]:
    """
    Flatten the nested DaTACOS metadata structure for easier access.

    Args:
        da_tacos_meta: Raw DaTACOS metadata with work_id -> performances structure

    Returns:
        Flattened metadata dictionary with performance IDs as keys
    """

    flat = {}
    for work_id, performances in da_tacos_meta.items():
        for perf_id, meta in performances.items():
            meta["work_id"] = work_id
            flat[perf_id] = meta
    return flat


class DaTACOSDataLoader:
    def __init__(self,
                 meta_path: str,
                 features_cens_path: str,
                 features_hpcp_path: str):
        """
        Args:
            meta_path: Path to the metadata JSON file
            features_cens_path: Base path for CENS feature files
            features_hpcp_path: Base path for HPCP feature files
        """
        self.metadata_raw = load_metadata(meta_path)
        self.metadata = flatten_metadata(self.metadata_raw)
        self.features_cens_path = features_cens_path
        self.features_hpcp_path = features_hpcp_path

    def get_metadata(self,
                     perf_id: str) -> Dict:
        """
        Get metadata for a specific performance ID.

        Args:
            perf_id: Performance ID to retrieve metadata for

        Returns:
            Dictionary containing metadata for the performance
        """

        return self.metadata.get(perf_id, {})

    def load_h5_feature(self, base_path: str, perf_id: str, suffix: str) -> np.ndarray:
        """
        Load feature embedding from H5 file

        Args:
            base_path: Base directory for feature files
            perf_id: Performance ID
            suffix: Feature type suffix ('cens' or 'hpcp')

        Returns:
            Feature embedding as numpy array
        """
        meta = self.get_metadata(perf_id)

        work_id = meta.get("work_id")
        if not work_id:
            return np.zeros(12)
        file_path = f"{base_path}/{work_id}_{suffix}/{perf_id}_{suffix}.h5"
        try:
            with h5py.File(file_path, "r") as f:
                # Try to fetch dataset; keys should match feature type
                # For CENS or HPCP, keys are generally "chroma_cens" or "hpcp"
                for key in f.keys():
                    if key in ["chroma_cens", "hpcp"]:
                        emb = np.array(f[key])
                        if emb.ndim > 1:
                            emb = np.mean(emb, axis=0)
                        return emb
            # fallback if no expected keys found
            return np.zeros(12)
        except Exception as e:
            print(f"Failed to load {perf_id} at {file_path}: {e}")
            return np.zeros(12)

    def get_cens_embedding(self, perf_id: str) -> np.ndarray:
        return self.load_h5_feature(self.features_cens_path, perf_id, "cens")

    def get_hpcp_embedding(self, perf_id: str) -> np.ndarray:
        return self.load_h5_feature(self.features_hpcp_path, perf_id, "hpcp")
