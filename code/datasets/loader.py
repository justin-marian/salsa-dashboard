import os
import torch
import numpy as np
from typing import Any

from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.datasets import Planetoid

from .polardf import read_edge
from .download import download_file_parallel
from .transform import df_to_np_edges, make_bidir_edges


def load_pyg(name: str) -> np.ndarray | None:
    """
    Try to load a Planetoid-style dataset via PyTorch Geometric and return edges.

    This is the fastest / cleanest path for citation-style datasets that are
    already supported by `torch_geometric.datasets.Planetoid` (e.g. Cora,
    Citeseer, Pubmed).

    Steps
    -----
    1. Instantiate `Planetoid(root=..., name=...)`.
    2. Grab the `edge_index` tensor from the first (and only) graph.
    3. Move `edge_index` to CPU if needed and convert to a NumPy array.
    4. Return edges as a 2D array of shape `(E, 2)` with dtype int64.

    Parameters
    ----------
    name:
        Dataset name as expected by `Planetoid` (e.g. "Cora").

    Returns
    -------
    edges:
        NumPy array of shape `(E, 2)` containing (u, v) pairs, or `None`
        if `edge_index` is missing (unlikely for Planetoid datasets).
    """
    dataset = Planetoid(root=f"../public/data/{name}", name=name)
    # PyG data object; Planetoid returns a single graph as dataset[0]
    graph_data = dataset[0]

    if hasattr(graph_data, "edge_index"):
        edge_index = graph_data.edge_index

        # Efficient GPU → CPU transfer if needed
        if getattr(edge_index, "is_cuda", False):
            edge_index = edge_index.cpu()

        edge_index_numpy = edge_index.numpy()
        # edge_index is shape (2, E); convert to (E, 2)
        return np.vstack((edge_index_numpy[0], edge_index_numpy[1])).T.astype(np.int64)

    return None


def load_linqs(
    root_dir: str,
    linqs_filename: str,
) -> np.ndarray:
    """
    Load a LINQS-format graph file and return edges as a NumPy array.

    This is a generic loader for datasets hosted in the LINQS repository
    (e.g., Cora, Citeseer in their original text format).

    Steps
    -----
    1. Download the file from the standard LINQS URL if it's not already local.
    2. Read the edge list into a DataFrame (`read_edge`).
    3. Convert to a `(E, 2)` NumPy array of integer node IDs.

    Parameters
    ----------
    root_dir:
        Root directory where the dataset should live (we use `root_dir/raw`).
    linqs_filename:
        Filename as hosted on the LINQS server (e.g. "cora.cites").

    Returns
    -------
    edges:
        NumPy array of shape `(E, 2)` with integer (u, v) pairs.
    """
    linqs_url = f"https://linqs-data.soe.ucsc.edu/public/lbc/{linqs_filename}"
    local = os.path.join(root_dir, "raw", linqs_filename)
    download_file_parallel(linqs_url, local)
    df = read_edge(local, comment_char="#")
    return df_to_np_edges(df)


def load_planetoid(
    name: str,
    root_dir: str,
    linqs_filename: str,
) -> np.ndarray:
    """
    Unified loader for Planetoid citation datasets (Cora, Citeseer, Pubmed).

    Strategy
    --------
    1. First try to load via PyTorch Geometric's `Planetoid` class
    (`load_pyg`), which is usually the most convenient and robust.
    2. If that fails (or returns `None`), fall back to reading the original
    LINQS-format file via `load_linqs`.

    Parameters
    ----------
    name:
        Dataset name as expected by `Planetoid` (e.g. "Cora").
    root_dir:
        Root directory for storing local copies of the dataset.
    linqs_filename:
        Filename of the LINQS edge-list file (e.g. "cora.cites").

    Returns
    -------
    edges:
        NumPy array of shape `(E, 2)` with integer edge pairs.
    """
    edges = load_pyg(name)
    if edges is not None:
        return edges
    return load_linqs(root_dir, linqs_filename)


def load_snap(
    url: str,
    root_path: str,
    comment_prefix: tuple[str, ...] = ("#",),
) -> np.ndarray:
    """
    Load a SNAP (Stanford Network Analysis Project) dataset and return edges.

    SNAP datasets are usually provided as plain text edge lists with comment
    lines starting with "#" and whitespace-separated integer node IDs.

    Steps
    -----
    1. Download the file from `url` to `root_path` if not already present.
    2. Use `read_edge` to parse the file into a DataFrame, skipping comment
    lines based on `comment_prefix`.
    3. Convert the DataFrame to a `(E, 2)` NumPy edge array.

    Parameters
    ----------
    url:
        Remote SNAP URL (e.g. "https://snap.stanford.edu/data/wiki-Vote.txt.gz").
    root_path:
        Local file path where the dataset will be stored.
        (Despite the name `root_path`, this is the full file path, not just a directory.)
    comment_prefix:
        Tuple of characters indicating comment lines. Only the first element
        is used as `comment_char` in `read_edge`.

    Returns
    -------
    edges:
        NumPy array of shape `(E, 2)` containing integer (u, v) pairs.
    """
    comment = comment_prefix[0] if comment_prefix else None
    download_file_parallel(url, root_path)
    edge_dataframe = read_edge(root_path, comment_char=comment)
    return df_to_np_edges(edge_dataframe)


def load_dir_graog(
    url: str,
    local: str,
    comment_prefix: tuple[str, ...] = ("#",),
) -> np.ndarray:
    """
    Load an undirected graph file and expand it into bidirectional edges.

    This is useful when:
    - The raw dataset is undirected (edge (u, v) implicitly means (v, u)).
    - Downstream processing expects an explicitly directed edge list where
    both directions are present.

    Steps
    -----
    1. Download the file from `url` to `local` if necessary.
    2. Read edges into a DataFrame with `read_edge`.
    3. Call `make_bidir_edges` to add reversed edges for each (u, v).
    4. Convert to a `(E, 2)` NumPy array.

    Parameters
    ----------
    url:
        Remote URL of the text edge list.
    local:
        Local file path where the data will be stored.
    comment_prefix:
        Comment prefixes used to skip metadata/header lines.

    Returns
    -------
    edges:
        NumPy array of shape `(E, 2)` with both (u, v) and (v, u) included.
    """
    comment = comment_prefix[0] if comment_prefix else None
    download_file_parallel(url, local)
    df = read_edge(local, comment_char=comment)
    df_bidir = make_bidir_edges(df)
    return df_to_np_edges(df_bidir)


def load_ogb(name: str, root_dir: str) -> dict[str, Any] | None:
    """
    Load an OGB node property prediction dataset via PyG and return basic graph info.

    OGB datasets are typically loaded through `PygNodePropPredDataset`. However,
    recent versions of PyTorch and torch.load include a `weights_only` flag that
    can cause issues when loading older checkpoints. To guard against this, we
    temporarily wrap `torch.load` to force `weights_only=False`.

    Steps
    -----
    1. Monkey-patch `torch.load` to inject `weights_only=False` (if omitted).
    2. Instantiate `PygNodePropPredDataset(name=name, root=root_dir)`.
    3. Extract `edge_index` and `num_nodes` from the first graph.
    4. Convert `edge_index` (shape (2, E)) to a NumPy array of shape `(E, 2)`.
    5. Restore the original `torch.load` in a `finally` block.

    Parameters
    ----------
    name:
        OGB dataset name (e.g., "ogbn-arxiv", "ogbn-products").
    root_dir:
        Root directory where OGB will cache the dataset files.

    Returns
    -------
    payload:
        Dictionary with keys:
            - 'num_nodes' : int, number of nodes
            - 'num_edges' : int, number of edges
            - 'edges'     : np.ndarray of shape (E, 2), edge list
        or `None` if loading fails (exception would be raised in that case).

    Notes
    -----
    - This function intentionally returns a plain dict rather than a NetworkX
    graph, since OGB is often consumed by PyTorch Geometric / GNN pipelines.
    - Always restores `torch.load` even if an exception occurs while loading.
    """
    original_torch_load = torch.load

    # Wrap torch.load to ensure weights_only=False (for compatibility).
    torch.load = lambda *args, **kwargs: original_torch_load(
        *args, **{**kwargs, "weights_only": False}
    )

    try:
        os.makedirs(root_dir, exist_ok=True)
        dataset = PygNodePropPredDataset(name=name, root=root_dir)
        graph_data = dataset[0]

        edge_index = graph_data.edge_index.numpy()  # shape (2, E)
        num_nodes = graph_data.num_nodes

        # Transpose to shape (E, 2)
        edges = edge_index.T

        return {
            "num_nodes": int(num_nodes),
            "num_edges": len(edges),
            "edges": edges,
        }
    finally:
        # Always restore original torch.load, even if an error occurs.
        torch.load = original_torch_load
