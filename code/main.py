from typing import Any
import gc
import os
import random
import numpy as np
import psutil
from tqdm.auto import tqdm

import torch
import torch.backends.cudnn as cudnn

from datasets.configs import get_dataset_configs
from datasets.register import load_datasets_parallel

from experiment.experiment import run_experiment
from export.export import export_regraph_items

from metrics.community_aware import community_interlink_amplification
from metrics.community_distance import distance_based_evaluation
from metrics.community_pagerank import community_rank_based_weighting
from metrics.pagerank_hits import run_baselines_parallel


def get_memory_usage() -> float:
    """Return current process memory usage in GB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 ** 3)


def cleanup_memory() -> None:
    """Force full Python + CUDA cleanup to prevent fragmentation."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def setup_random_seeds(seed: int = 42) -> None:
    """Ensures full reproducibility (as much as CUDA allows)."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.deterministic = True
    cudnn.benchmark = True
    cudnn.enabled = True


def main():
    setup_random_seeds(42)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tqdm.write(f"Using device: {device}")

    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        tqdm.write(f"GPU: {props.name}")
        tqdm.write(f"GPU Memory: {props.total_memory / 1e9:.1f} GB")

    dataset_configs = get_dataset_configs()
    tqdm.write(f"Found {len(dataset_configs)} datasets")
    tqdm.write(f"Initial memory: {get_memory_usage():.2f} GB")

    tqdm.write("\nLoading datasets...")
    loaded_graphs = load_datasets_parallel(dataset_configs)

    valid_datasets: list[tuple[dict[str, Any], Any]] = []
    for config, graph in zip(dataset_configs, loaded_graphs):
        if graph and graph.number_of_nodes() > 0:
            tqdm.write(
                f"  [SUCCESS] {config['name']}: "
                f"{graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges"
            )
            valid_datasets.append((config, graph))
        else:
            tqdm.write(f"  [FAILED] {config['name']}: Could not load dataset")

    tqdm.write(f"\nStarting experiments on {len(valid_datasets)} datasets")

    experiment_results = []

    for config, graph in valid_datasets:
        tqdm.write("\n" + "=" * 60)
        tqdm.write(f"Processing dataset: {config['title']}")
        tqdm.write("=" * 60)

        tqdm.write(
            f"Graph: {graph.number_of_nodes()} nodes, "
            f"{graph.number_of_edges()} edges"
        )
        tqdm.write(f"Memory before: {get_memory_usage():.2f} GB")

        community_method = config["community_method"]
        community_kwargs = config["community_kwargs"]

        graph_smoothing_function = config["graph_smoothing_function"]
        damping_factor = config["damping_factor"]
        max_iterations = config["max_iterations"]
        tolerance = config["tolerance"]

        (
            salsa_hubs,
            salsa_authorities,
            hits_hubs,
            hits_auths,
            pagerank_scores,
            community_assignments,
        ) = run_experiment(
            name=config["title"],
            graph=graph,
            use_weights=True,
            community_method=community_method,
            community_kwargs=community_kwargs,
            graph_smoothing_function=graph_smoothing_function,
            damping_factor=damping_factor,
            max_iterations=max_iterations,
            tolerance=tolerance,
            verbose=True,
        )

        export_regraph_items(
            graph,
            salsa_hubs,
            salsa_authorities,
            hits_hubs,
            hits_auths,
            pagerank_scores,
            community_assignments,
            output_file=config["outfile"],
        )
        tqdm.write(f"Exported results to: {config['outfile']}")

        # ----- Community-aware metrics -----
        community_graph = community_interlink_amplification(graph, community_assignments)
        tqdm.write(f"  Community Interlink: {community_graph.number_of_edges()} edges")

        rank_graph = community_rank_based_weighting(graph, community_assignments)
        tqdm.write(f"  Community Rank: {rank_graph.number_of_edges()} edges")

        # ----- Distance-based evaluation -----
        dist_graph = distance_based_evaluation(graph)
        tqdm.write(f"  Distance-Based: {dist_graph.number_of_edges()} edges")

        # ----- Baseline centralities -----
        baseline_pr, baseline_hubs, baseline_auths = run_baselines_parallel(
            graph, 
            hits_params={
                "max_iter": config["max_iterations"],
                "tol": config["tolerance"],
                "use_weight": True,
            }, 
            pr_params={
                "max_iter": config["max_iterations"],
                "tol": config["tolerance"],
                "use_weight": True,
            }
        )
        tqdm.write(f"  Baseline PageRank: {len(baseline_pr)} scores")
        tqdm.write(f"  Baseline HITS: {len(baseline_hubs)} hubs, {len(baseline_auths)} authorities")

        experiment_results.append(
            {
                "dataset_name": config["name"],
                "status": "SUCCESS",
                "node_count": graph.number_of_nodes(),
                "edge_count": graph.number_of_edges(),
                "metrics": {
                    "community_interlink_edges": community_graph.number_of_edges(),
                    "community_rank_edges": rank_graph.number_of_edges(),
                    "distance_based_edges": dist_graph.number_of_edges(),
                    "pagerank_scores": len(baseline_pr),
                    "hubs_scores": len(baseline_hubs),
                    "authorities_scores": len(baseline_auths),
                },
            }
        )

        tqdm.write(f"Memory after: {get_memory_usage():.2f} GB")

        del graph
        cleanup_memory()

    tqdm.write("\n" + "=" * 60)
    tqdm.write("EXPERIMENT SUMMARY")
    tqdm.write("=" * 60)

    for result in experiment_results:
        tqdm.write(f"\nDataset: {result['dataset_name']}")
        tqdm.write(f" Nodes: {result['node_count']}, Edges: {result['edge_count']}")
        tqdm.write(" Metrics:")
        for key, value in result["metrics"].items():
            tqdm.write(f"    {key}: {value}")

    tqdm.write(f"\nCompleted {len(experiment_results)}/{len(dataset_configs)} datasets")
    tqdm.write(f"Final memory usage: {get_memory_usage():.2f} GB")


if __name__ == "__main__":
    main()
