from itertools import combinations
import random

import pandas as pd
from tqdm import tqdm
import multiprocessing as mp
from functools import partial

from data.loader import DaTACOSDataLoader
from src.tools.similarity import cosine_similarity
from src.tools.metadata import metadata_similarity


def load_data(loader, num_samples=5000, random_seed=42):
    """Load and sample data from the dataset."""
    random.seed(random_seed)
    all_keys = list(loader.metadata.keys())
    sampled_keys = random.sample(all_keys, num_samples) if num_samples < len(all_keys) else all_keys

    print(f"Loading data for {len(sampled_keys)} samples...")

    data = {}
    for k in tqdm(sampled_keys, desc="Loading features"):
        data[k] = {
            "hpcp": loader.get_hpcp_embedding(k),
            "cens": loader.get_cens_embedding(k),
            "meta_features": loader.get_metadata(k)
        }

    return sampled_keys, data


def process_pair(pair, data):
    """
    Process a single pair of tracks to compute similarity metrics and label.

    Args:
        pair (tuple): A tuple of (track_id1, track_id2) to compare
        data (dict): Dictionary containing features and metadata for all tracks

    Returns:
        tuple: (track_id1, track_id2, sim_hpcp, sim_cens, sim_meta, label) where:
            - track_id1, track_id2: IDs of the compared tracks
            - sim_hpcp: HPCP-based similarity score
            - sim_cens: CENS-based similarity score
            - sim_meta: Metadata-based similarity score
            - label: Binary label (1 if different performances of same work, 0 otherwise)
    """
    i, j = pair
    sim_hcpc = cosine_similarity(data[i]["hpcp"], data[j]["hpcp"])
    sim_cens = cosine_similarity(data[i]["cens"], data[j]["cens"])
    sim_meta = metadata_similarity(data[i]["meta_features"], data[j]["meta_features"])

    # Label: different performances of the same work
    label = int((data[i]["meta_features"]["work_id"] == data[j]["meta_features"]["work_id"]) and
                (data[i]["meta_features"]["perf_id"] != data[j]["meta_features"]["perf_id"]))

    return (i, j, sim_hcpc, sim_cens, sim_meta, label)


def create_dataset(num_samples=5000, chunk_size=10000, n_jobs=None):
    """
    Create a training dataset for the similarity aggregator model.

    This function processes the DA-TACOS dataset to create pairs of tracks with
    similarity metrics and binary labels for training an aggregator model.
    Processing is done in chunks with multiprocessing for efficiency.

    Args:
        num_samples (int, optional): Number of tracks to sample from dataset.
            Defaults to 5000.
        chunk_size (int, optional): Number of pairs to process in each chunk.
            Helps manage memory usage. Defaults to 10000.
        n_jobs (int, optional): Number of parallel workers. If None, uses
            CPU count - 1. Defaults to None.

    Notes:
        The resulting dataset is saved to "da-tacos/train_set_for_aggregator_model.csv".
        Intermediate results are saved every 5 chunks to prevent memory overflow.
    """
    n_jobs = n_jobs or max(1, mp.cpu_count() - 1)

    # Initialize loader
    loader = DaTACOSDataLoader(
        meta_path="da-tacos/da-tacos_metadata/da-tacos_benchmark_subset_metadata.json",
        features_cens_path="da-tacos/da-tacos_benchmark_subset_cens",
        features_hpcp_path="da-tacos/da-tacos_benchmark_subset_hpcp"
    )

    # Load data
    sampled_keys, data = load_data(loader, num_samples)

    # Generate all pairs
    all_pairs = list(combinations(sampled_keys, 2))
    total_pairs = len(all_pairs)
    print(f"Processing {total_pairs} pairs with {n_jobs} workers...")

    # Process in chunks to manage memory
    chunks = [all_pairs[i:i + chunk_size] for i in range(0, total_pairs, chunk_size)]
    results = []

    # Set up parallel processing
    worker_func = partial(process_pair, data=data)

    # Process each chunk
    for i, chunk in enumerate(chunks):
        print(f"Processing chunk {i + 1}/{len(chunks)}")
        with mp.Pool(processes=n_jobs) as pool:
            chunk_results = list(tqdm(
                pool.imap(worker_func, chunk),
                total=len(chunk),
                desc="Computing similarities"
            ))
        results.extend(chunk_results)

        # Optional: write intermediate results to save memory
        if (i + 1) % 5 == 0 or i == len(chunks) - 1:
            temp_df = pd.DataFrame(results,
                                   columns=["perf_id1", "perf_id2", "sim_hpcp", "sim_centroid", "sim_metadata", "label"])
            mode = "w" if i == 0 else "a"
            header = i == 0
            temp_df.to_csv("da-tacos/train_set_for_aggregator_model.csv",
                           mode=mode,
                           header=header,
                           index=False)
            results = []  # Free memory

    print("Dataset creation completed!")


if __name__ == "__main__":
    create_dataset(num_samples=100)