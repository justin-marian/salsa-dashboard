import gc
import os
import networkx as nx
from tqdm import tqdm
from typing import Any
import concurrent.futures as cf

from .conversion import load_unidir_graph
from .loader import load_snap, load_dir_graog


def load_wikivote() -> nx.Graph:
    """
    Load the Wiki-Vote directed network as a cleaned NetworkX graph.

    Source
    ------
    SNAP dataset: https://snap.stanford.edu/data/wiki-Vote.html

    Steps
    -----
    1. Download the `wiki-Vote.txt.gz` file if it is not already cached.
    2. Parse edges via `load_snap` into a NumPy edge array.
    3. Pass edges through `load_unidir_graph`, which:
        - removes self-loops,
        - optionally keeps only the largest (weakly) connected component,
        - relabels nodes to dense integers 0..N-1,
        - constructs a directed NetworkX graph.

    Returns
    -------
    G:
        Directed NetworkX graph representing the Wikipedia voting network.
    """
    local = "../public/data/WikiVote/wiki-Vote.txt.gz"
    edges = load_snap(
        "https://snap.stanford.edu/data/wiki-Vote.txt.gz",
        local,
        comment_prefix=("#",),
    )
    return load_unidir_graph(
        edges,
        keep_lcc=True,
        relabel_to_int=True,
        remove_self_loops=True,
        directed=True,
    )


def load_ego_facebook() -> nx.Graph:
    """
    Load the ego-Facebook social network as a cleaned undirected graph.

    Source
    ------
    SNAP dataset: https://snap.stanford.edu/data/ego-Facebook.html

    The file `facebook_combined.txt.gz` is an undirected graph. We use
    `load_dir_graog` to:
        - download the file,
        - read edges,
        - expand them into explicit (u, v) and (v, u) pairs,
    and then feed those edges into `load_unidir_graph` with `directed=False`.

    Returns
    -------
    G:
        Undirected NetworkX graph representing the combined Facebook ego network.
    """
    local = "../public/data/EgoFacebook/facebook_combined.txt.gz"
    edges = load_dir_graog(
        "https://snap.stanford.edu/data/facebook_combined.txt.gz",
        local,
        comment_prefix=("#",),
    )
    return load_unidir_graph(
        edges,
        keep_lcc=True,
        relabel_to_int=True,
        remove_self_loops=True,
        directed=False,
    )


def load_soc_sign_bitcoin_otc() -> nx.Graph:
    """
    Load the soc-sign-bitcoin-otc trust network as a cleaned directed graph.

    Source
    ------
    SNAP dataset: https://snap.stanford.edu/data/soc-sign-bitcoinotc.html

    The raw CSV has the format:

        source, target, rating, time

    We only need the topology (source, target), so we:
        - download and parse via `load_snap`,
        - keep only the first 2 columns if extra columns exist,
        - pass the resulting edge pairs into `load_unidir_graph`.

    Returns
    -------
    G:
        Directed NetworkX graph where edges represent trust ratings
        (weights/signs are discarded in this loader).
    """
    local = "../public/data/BitcoinOTC/soc-sign-bitcoinotc.csv.gz"
    edges = load_snap(
        "https://snap.stanford.edu/data/soc-sign-bitcoinotc.csv.gz",
        local,
        comment_prefix=("#",),
    )
    # Drop rating/time columns if present; keep only (src, dst)
    if hasattr(edges, "shape") and edges.shape[1] > 2:
        edges = edges[:, :2]

    return load_unidir_graph(
        edges,
        keep_lcc=True,
        relabel_to_int=True,
        remove_self_loops=True,
        directed=True,
    )


def load_datasets_parallel(dataset_configs: list[dict[str, Any]]) -> list[nx.Graph]:
    """
    Load multiple datasets in parallel based on a list of config dictionaries.

    Each config is expected to contain a ``"name"`` key, matching one of the
    supported dataset identifiers:

        - "wiki-vote"
        - "ego-facebook"
        - "soc-sign-bitcoin-otc"

    Behavior
    --------
    1. Build a mapping from dataset names to loader functions.
    2. Filter requested names against a whitelist of SAFE_DATASETS
    (only those are allowed to be executed).
    3. Use a ThreadPoolExecutor to call the appropriate loader for each
    requested dataset, with at most ``min(#safe_datasets, cpu_count)`` workers.
    4. Return a list of graphs aligned with the input order of configs.
    For any config whose name is not recognized, the corresponding entry
    in the result list remains an empty DiGraph (as initialized).

    Parameters
    ----------
    dataset_configs:
        List of dictionaries, each describing a dataset. At minimum, each
        config must contain:
            - "name": str  (e.g. "wiki-vote", "ego-facebook"...)

    Returns
    -------
    graphs:
        List of NetworkX graphs (Directed or Undirected, depending on loader),
        in the same order as ``dataset_configs``. If a dataset name is not in
        the loaders mapping or SAFE_DATASETS, the corresponding entry is an
        empty DiGraph.
    """
    loaders: dict[str, Any] = {
        "wiki-vote": load_wikivote,
        "ego-facebook": load_ego_facebook,
        "soc-sign-bitcoin-otc": load_soc_sign_bitcoin_otc,
    }
    # Whitelist of dataset names that are allowed to be loaded in parallel
    SAFE_DATASETS = {"wiki-vote", "ego-facebook", "soc-sign-bitcoin-otc"}

    # Normalize requested dataset names
    requested_names: list[str] = [cfg.get("name", "").lower() for cfg in dataset_configs]

    # Initialize result list with empty graphs (one per config)
    results: list[nx.Graph] = [nx.DiGraph() for _ in requested_names]

    future_to_idx: dict[cf.Future, int] = {}
    safe_count = sum(1 for name in requested_names if name in SAFE_DATASETS)

    if safe_count > 0:
        cpu_count = os.cpu_count() or 1
        max_workers = min(safe_count, cpu_count)

        with cf.ThreadPoolExecutor(max_workers=max_workers) as pool:
            # Submit loader jobs for each safe dataset
            for idx, ds_name in enumerate(requested_names):
                fn = loaders.get(ds_name)
                if fn and ds_name in SAFE_DATASETS:
                    # Wrap loader to avoid capturing mutable loop variable
                    def _runner(wrapped_fn=fn):
                        return wrapped_fn()

                    fut = pool.submit(_runner)
                    future_to_idx[fut] = idx

            # Collect results as they complete, with a nice tqdm progress bar
            for fut in tqdm(
                cf.as_completed(list(future_to_idx.keys())),
                total=len(future_to_idx),
                desc="Loading datasets (threaded)",
            ):
                gc.collect()
                idx = future_to_idx[fut]
                results[idx] = fut.result()
                gc.collect()

    return results
