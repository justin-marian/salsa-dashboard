from tqdm import tqdm
from typing import Any

from .rank import topk


def report_results(
    name: str,
    node_count: int,
    edge_count: int,
    elapsed_time: float,
    device: str,
    salsa_hubs: dict[Any, float],
    salsa_authorities: dict[Any, float],
    hits_hubs: dict[Any, float],
    hits_authorities: dict[Any, float],
    pagerank_scores: dict[Any, float],
    k: int = 10,
) -> None:
    """
    Print a compact summary of an experiment: speed + top-k centralities.

    For a single run, this reports:
        - basic graph stats (nodes, edges),
        - runtime and per-node time (μs/node),
        - top-k SALSA hubs / authorities,
        - top-k HITS hubs / authorities (if available),
        - top-k PageRank scores (if available).

    Parameters
    ----------
    name:
        Dataset / experiment name for logging.
    node_count:
        Number of nodes in the (cleaned) graph.
    edge_count:
        Number of edges in the (cleaned) graph.
    elapsed_time:
        Total runtime of the experiment in seconds.
    device:
        String describing the compute device (e.g. "cpu", "cuda").
    salsa_hubs:
        Node → SALSA hub score mapping.
    salsa_authorities:
        Node → SALSA authority score mapping.
    hits_hubs:
        Node → HITS hub score mapping (may be empty if not computed).
    hits_authorities:
        Node → HITS authority score mapping (may be empty).
    pagerank_scores:
        Node → PageRank score mapping (may be empty).
    k:
        Number of top nodes to display for each centrality.
    """
    # Microseconds per node for a quick “speed sense”
    time_per_node = elapsed_time / max(node_count, 1) * 1e6

    tqdm.write(
        f"[{name}] ULTRA-FAST GPU | nodes={node_count} edges={edge_count} "
        f"| time={elapsed_time:.2f}s ({device}) | {time_per_node:.1f} μs/node"
    )

    # SALSA should always exist in this pipeline, but be defensive anyway
    if salsa_hubs:
        tqdm.write(f"  Top {k} SALSA hubs: {topk(salsa_hubs, k)}")
    else:
        tqdm.write(f"  Top {k} SALSA hubs: []")

    if salsa_authorities:
        tqdm.write(f"  Top {k} SALSA authorities: {topk(salsa_authorities, k)}")
    else:
        tqdm.write(f"  Top {k} SALSA authorities: []")

    if hits_hubs:
        tqdm.write(f"  Top {k} HITS hubs: {topk(hits_hubs, k)}")
        tqdm.write(f"  Top {k} HITS authorities: {topk(hits_authorities, k)}")

    if pagerank_scores:
        tqdm.write(f"  Top {k} PageRank: {topk(pagerank_scores, k)}")
