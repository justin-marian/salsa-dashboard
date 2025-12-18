import networkx as nx
from tqdm.auto import tqdm
import time

from .dfs_minbd import DFSMinBD


def minimal_back_distance(
    graph: nx.DiGraph,
    max_distance: int = 30,
    verbose: bool = False,
) -> dict[tuple, int | None]:
    """
    Compute Minimum Backward Distance (MinBD) for every edge in a directed graph.

    Definition
    ----------
    For a directed edge (u → v), the Minimum Backward Distance is:

        MinBD(u → v) = length of the shortest directed path v ⇒ u
                (capped at `max_distance`, or None if no path exists)

    Intuition
    ---------
    - Small MinBD: (u → v) lies in a short directed cycle; v can return to u quickly.
    - Large / None: (u → v) leads into regions with poor or no return paths,
    indicating more “one-way” structure.

    Parameters
    ----------
    graph:
        Input directed graph.
    max_distance:
        Maximum backward distance to track; any longer distances are clamped
        to this value to avoid unbounded growth in large graphs.
    verbose:
        If True, prints progress and summary statistics via `tqdm`.

    Returns
    -------
    distances:
        A dictionary mapping each edge (u, v) to:
        - an integer distance in [1, max_distance], or
        - None if no return path v ⇒ u was discovered.
    """
    analyzer = DFSMinBD(graph, max_distance, verbose)

    if verbose:
        tqdm.write(
            f"Starting MinBD computation for graph with "
            f"{graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges"
        )

    started = time.time()
    distances = analyzer.minimal_backward_distance()
    elapsed = time.time() - started

    if verbose:
        finite_count = sum(1 for distance in distances.values() if distance is not None)
        total_edges = len(distances)
        tqdm.write(f"MinBD computation completed in {elapsed:.2f} seconds")
        tqdm.write(f"Results: {finite_count}/{total_edges} edges have finite MinBD")

        stats = analyzer.get_statistics()
        if stats:
            tqdm.write(f"Statistics: {stats}")

    return distances
