import numpy as np
import polars as pl


def df_to_np_edges(df: pl.DataFrame) -> np.ndarray:
    """
    Convert a Polars DataFrame with ``src`` and ``dst`` columns to a NumPy edge array.

    Parameters
    ----------
    df:
        Polars DataFrame expected to contain at least columns:
            - "src"
            - "dst"

    Returns
    -------
    edges:
        NumPy array of shape (E, 2) with dtype int64, where each row is (src, dst).
        If the DataFrame is empty, returns an empty array with shape (0, 2).
    """
    if df.is_empty():
        return np.empty((0, 2), dtype=np.int64)

    # Use PyArrow-backed conversion (faster / more memory-efficient for large columns)
    source_nodes = df["src"].to_numpy(use_pyarrow=True)
    target_nodes = df["dst"].to_numpy(use_pyarrow=True)
    return np.column_stack([source_nodes, target_nodes]).astype(np.int64, copy=False)


def remap_labels(edges: np.ndarray) -> np.ndarray:
    """
    Remap arbitrary node labels to dense integer IDs 0..N-1.

    This takes an edge array (whatever its original dtype) and compresses all
    unique labels into a contiguous integer range suitable for array-based
    graph algorithms.

    Example
    -------
    edges = [[10, 42],
             [42, 99]]
    → unique labels {10, 42, 99} → mapped to {0, 1, 2}:
      [[0, 1],
       [1, 2]]

    Parameters
    ----------
    edges:
        Array-like of shape (E, 2) or compatible; will be flattened then
        reshaped to (E, 2) after remapping.

    Returns
    -------
    remapped:
        NumPy array of shape (E, 2) with dtype int64 and labels in [0, N-1].
        If ``edges`` is empty, returns an empty (0, 2) array.
    """
    if edges.size == 0:
        return edges.reshape(-1, 2).astype(np.int64)

    # Flatten, compute unique labels + inverse mapping, then reshape back to (E, 2)
    _, inverse_mapping = np.unique(edges.reshape(-1), return_inverse=True)
    return inverse_mapping.reshape(-1, 2).astype(np.int64)


def remap_edges(edges: np.ndarray) -> np.ndarray:
    """
    Normalize an edge list to int64 node IDs, remapping labels when needed.

    Behavior
    --------
    - If the input is already numeric, it is cast (cheaply) to int64.
    - If the input is non-numeric (e.g. strings, objects), it is passed through
    :func:`remap_labels` to obtain dense integer IDs 0..N-1.

    Parameters
    ----------
    edges:
        Array-like of shape (E, 2) or similar, containing node labels.

    Returns
    -------
    normalized_edges:
        NumPy array of shape (E, 2) with dtype int64.
        If the input is empty, returns an empty (0, 2) array.
    """
    edges_array = np.atleast_2d(np.asarray(edges))
    if edges_array.size == 0:
        return edges_array.reshape(0, 2).astype(np.int64)

    # Fast path: already numeric → just cast to int64 without unnecessary copies
    if np.issubdtype(edges_array.dtype, np.number):
        return edges_array.astype(np.int64, copy=False)

    # Slow path: non-numeric labels need remapping
    return remap_labels(edges_array)


def remove_self_loops(edges: np.ndarray) -> np.ndarray:
    """
    Remove self-loop edges (u, u) from an edge list.

    Parameters
    ----------
    edges:
        NumPy array of shape (E, 2) containing edge endpoints.

    Returns
    -------
    filtered_edges:
        Edge array of shape (E', 2) without any self-loops.
        If ``edges`` is empty, it is returned unchanged.
    """
    if edges.size == 0:
        return edges

    # Keep only edges where source != target
    non_self_loop_mask = edges[:, 0] != edges[:, 1]
    return edges[non_self_loop_mask]


def make_bidir_edges(df: pl.DataFrame) -> pl.DataFrame:
    """
    Construct an explicitly bidirectional edge list from an undirected graph DataFrame.

    Given a DataFrame of directed edges (u, v), this:
        1. Drops self-loops.
        2. Creates reverse edges (v, u) for each remaining edge.
        3. Concatenates original + reverse edges.
        4. Drops duplicates.

    Result: For every undirected edge {u, v} you end up with both (u, v) and (v, u).

    Parameters
    ----------
    df:
        Polars DataFrame with columns "src" and "dst".

    Returns
    -------
    df_bidir:
        Polars DataFrame with the same schema, containing both original
        and reverse edges, with duplicates removed. If the input is empty
        (or all self-loops), returns an empty DataFrame.
    """
    if df.is_empty():
        return df

    # Drop self-loops to avoid creating duplicated self-loops (u, u) twice.
    clean_dataframe = df.filter(pl.col("src") != pl.col("dst"))
    if clean_dataframe.is_empty():
        return clean_dataframe

    # Create reverse edges by swapping src and dst
    reverse_edges = clean_dataframe.select(
        [
            pl.col("dst").alias("src"),
            pl.col("src").alias("dst"),
        ]
    )

    # Combine original + reverse and drop duplicates
    return pl.concat([clean_dataframe, reverse_edges], how="vertical").unique()
