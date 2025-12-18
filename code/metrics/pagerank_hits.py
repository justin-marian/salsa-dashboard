"""
GPU-accelerated baselines: HITS and PageRank.

- **HITS** (Hyperlink-Induced Topic Search):
    Iteratively reinforces:
        • "good hubs point to good authorities"
        • "good authorities are pointed to by good hubs"
    Produces two score vectors:
        • hub scores
        • authority scores

- **PageRank**:
    Computes the stationary distribution of a random surfer:
        • with probability `alpha`, follow an outgoing link (weighted transition)
        • with probability `1 - alpha`, teleport uniformly to any node.

Weight semantics
----------------
- The *attribute name* used for weighted edges in this module is **'weights'**.
- If `use_weight=True`, we look at edge_data['weights'] (defaulting to 1.0).
- If `use_weight=False`, we ignore edge weights and treat all edges as weight 1.
- If your graph has unweighted edges, you can either:
    • omit the 'weights' attribute, or
    • call with `use_weight=False`.

Note
----
Other parts of your pipeline (e.g. SALSA, community methods) typically use
the edge attribute name `'weight'`. This module intentionally uses `'weights'`
to keep baselines separate, and the caller is responsible for populating
or ignoring this attribute as needed.
"""

import time
from typing import Any

import networkx as nx
import torch
from tqdm.auto import tqdm

from experiment.utils import (
    get_device,
    graph_to_adjacency_tensors,
    normalize_rows,
    sparse_matmul,
    batched_power_iteration,
)


class GraphScorer:
    """
    Base class for GPU-based graph scorers (PageRank / HITS).

    - Manages the compute device (via `get_device()`).
    - Provides a small logging helper respecting the `verbose` flag.
    """

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.device = get_device()

    def log(self, message: str) -> None:
        """Log a message only when `verbose=True`."""
        if self.verbose:
            tqdm.write(message)


########################################################################################################################


class PageRankScorer(GraphScorer):
    """
    GPU-based PageRank scorer using power iteration on a row-stochastic adjacency.

    Parameters
    ----------
    alpha:
        Probability of following an outgoing edge (1 - alpha is teleportation).
    max_iter:
        Maximum number of power-iteration steps.
    tol:
        L1 convergence tolerance on successive iterates.
    batch_size:
        Number of vectors to try in parallel for robust convergence
        (passed to `batched_power_iteration`).
    verbose:
        If True, prints timing and iteration info via `tqdm.write`.
    """

    def __init__(
        self,
        alpha: float = 0.85,
        max_iter: int = 100,
        tol: float = 1e-9,
        batch_size: int = 1024,
        verbose: bool = False,
    ):
        super().__init__(verbose)
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.batch_size = batch_size

    def score(self, graph: nx.DiGraph, weights: str | None = "weights") -> dict[Any, float]:
        """
        Compute PageRank scores for all nodes in `graph`.

        Parameters
        ----------
        graph:
            Directed graph on which PageRank is computed.
        weights:
            Name of the edge attribute used as weights, or None to ignore weights.
            When None, all edges are treated as having weight 1.0.

        Returns
        -------
        scores:
            Mapping node -> PageRank score (float).
        """
        self.log(f"[PageRank] Starting computation with alpha={self.alpha}, max_iter={self.max_iter}")
        started = time.perf_counter()

        # Build sparse adjacency tensors on the target device
        edge_index, edge_weight, nodes, node_count = graph_to_adjacency_tensors(
            graph,
            weights or "weights",
            self.device,
        )

        # If no weights are used, override the loaded edge weights with 1's
        if weights is None:
            edge_weight = torch.ones_like(edge_weight)

        # Row-normalize to obtain a stochastic transition matrix
        edge_weight = normalize_rows(edge_index, edge_weight, node_count)

        # Power iteration with teleportation via damping_factor = 1 - alpha
        pr_scores = batched_power_iteration(
            edge_index=edge_index,
            edge_weight=edge_weight,
            node_count=node_count,
            max_iterations=self.max_iter,
            tolerance=self.tol,
            damping_factor=1 - self.alpha,
            batch_size=self.batch_size,
            verbose=self.verbose,
        )

        elapsed = time.perf_counter() - started
        self.log(f"[PageRank] Completed in {elapsed:.3f}s")

        return {nodes[i]: pr_scores[i].item() for i in range(node_count)}


def compute_pagerank(
    graph: nx.DiGraph,
    alpha: float = 0.85,
    max_iter: int = 100,
    tol: float = 1e-9,
    batch_size: int = 1024,
    weights: str | None = "weights",
    verbose: bool = False,
) -> dict[Any, float]:
    """
    Convenience wrapper to compute PageRank in a single call.
    """
    scorer = PageRankScorer(
        alpha=alpha,
        max_iter=max_iter,
        tol=tol,
        batch_size=batch_size,
        verbose=verbose,
    )
    return scorer.score(graph, weights)


########################################################################################################################


def pagerank_scores(
    graph: nx.DiGraph,
    alpha: float = 0.85,
    max_iter: int = 100,
    tol: float = 1e-9,
    use_weight: bool | None = None,
    verbose: bool = True,
    **kwargs: Any,
) -> dict[object, float]:
    """
    High-level PageRank interface used by the experiment pipeline.

    Interprets `use_weight` and `kwargs["weights"]` to decide if edge
    weights should be used and which attribute name to read.

    Logic
    -----
    - If `use_weight` is None and `'weights'` is in kwargs:
        `use_weight = bool(kwargs["weights"])`
    - If `use_weight` is still None:
        default to True.
    - If `use_weight` is True:
        use edge attribute name `'weights'`.
    - If `use_weight` is False:
        ignore edge weights (treat all edges as weight 1).

    Returns
    -------
    Mapping node -> PageRank score.
    """
    if use_weight is None and "weights" in kwargs:
        use_weight = bool(kwargs["weights"])
    if use_weight is None:
        use_weight = True  # default behavior

    started = time.perf_counter()
    weight_key = "weights" if use_weight else None

    pr = compute_pagerank(
        graph=graph,
        alpha=alpha,
        max_iter=max_iter,
        tol=tol,
        batch_size=kwargs.get("batch_size", 1024),
        weights=weight_key,  # str name or None
        verbose=verbose,
    )
    elapsed = time.perf_counter() - started
    tqdm.write(
        f"[PageRank] alpha={alpha}, iter<={max_iter}, "
        f"tol={tol:.1e}, weighted={use_weight} | time={elapsed:.3f}s"
    )
    return pr


########################################################################################################################


class HITSScorer(GraphScorer):
    """
    GPU-based HITS scorer for hub and authority scores.

    Parameters
    ----------
    max_iter:
        Maximum number of HITS iterations.
    tol:
        L1 convergence tolerance between successive hub/authority vectors.
    verbose:
        If True, logs per-10-iteration diffs and convergence info.
    """

    def __init__(self, max_iter: int = 100, tol: float = 1e-9, verbose: bool = False):
        super().__init__(verbose)
        self.max_iter = max_iter
        self.tol = tol

    def score(self, graph: nx.DiGraph, weights: str | None = "weights") -> tuple[dict[Any, float], dict[Any, float]]:
        """
        Compute HITS hub and authority scores for all nodes.

        Parameters
        ----------
        graph:
            Input directed graph.
        weights:
            Name of the edge attribute used as weights, or None to ignore weights.

        Returns
        -------
        hubs, authorities:
            Two dictionaries mapping node -> score.
        """
        self.log(f"[HITS] Starting computation with max_iter={self.max_iter}")
        started = time.perf_counter()

        edge_index, edge_weight, nodes, node_count = graph_to_adjacency_tensors(
            graph,
            weights or "weights",
            self.device,
        )
        if weights is None:
            edge_weight = torch.ones_like(edge_weight)

        # Normalize outgoing weights to make adjacency row-stochastic
        edge_weight = normalize_rows(edge_index, edge_weight, node_count)
        edge_index_t = torch.stack([edge_index[1], edge_index[0]])

        hubs, authorities = self.init_scores(edge_index, edge_weight, node_count)
        hubs, authorities = self.iter_updates(
            edge_index=edge_index,
            edge_index_t=edge_index_t,
            edge_weight=edge_weight,
            hubs=hubs,
            authorities=authorities,
            count_nodes=node_count,
        )

        hubs_dict = {nodes[i]: hubs[i].item() for i in range(node_count)}
        auth_dict = {nodes[i]: authorities[i].item() for i in range(node_count)}

        elapsed = time.perf_counter() - started
        self.log(f"[HITS] Completed in {elapsed:.3f}s")
        return hubs_dict, auth_dict

    def init_scores(
        self,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        n: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Degree-based initialization for hubs and authorities.

        - Hubs start proportional to out-degree.
        - Authorities start proportional to in-degree.
        - If all degrees are zero, fall back to uniform distribution.
        """
        device = self.device
        one = torch.ones_like(edge_weight)

        out_deg = torch.zeros(n, device=device).scatter_add_(0, edge_index[0], one)
        in_deg = torch.zeros(n, device=device).scatter_add_(0, edge_index[1], one)

        eps = 1e-12
        if out_deg.sum() <= eps:
            hubs = torch.full((n,), 1.0 / max(n, 1), device=device)
        else:
            hubs = out_deg / out_deg.sum()

        if in_deg.sum() <= eps:
            authorities = torch.full((n,), 1.0 / max(n, 1), device=device)
        else:
            authorities = in_deg / in_deg.sum()

        return hubs, authorities

    def iter_updates(
        self,
        edge_index: torch.Tensor,
        edge_index_t: torch.Tensor,
        edge_weight: torch.Tensor,
        hubs: torch.Tensor,
        authorities: torch.Tensor,
        count_nodes: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Perform iterative HITS updates until convergence or max_iter.

        Updates
        -------
        - next_authorities = Aᵀ · hubs
        - next_hubs        = A   · authorities

        with L1-normalization and stability checks at each step.
        """
        iterator = range(self.max_iter)
        if self.verbose:
            iterator = tqdm(iterator, desc="HITS iterations", leave=False)

        eps = 1e-6

        for iteration in iterator:
            # Authority: incoming from hub neighbors
            next_authorities = sparse_matmul(edge_index_t, edge_weight, hubs, count_nodes)
            # Hub: outgoing to authority neighbors
            next_hubs = sparse_matmul(edge_index, edge_weight, authorities, count_nodes)

            # Clamp negatives due to numeric noise
            next_authorities = torch.clamp(next_authorities, min=0)
            next_hubs = torch.clamp(next_hubs, min=0)

            # L1 normalize with robust fallback to uniform
            a_norm = next_authorities.norm(p=1).clamp_min(eps)
            h_norm = next_hubs.norm(p=1).clamp_min(eps)

            if a_norm <= eps:
                next_authorities = torch.full_like(next_authorities, 1.0 / max(count_nodes, 1))
            else:
                next_authorities = next_authorities / a_norm

            if h_norm <= eps:
                next_hubs = torch.full_like(next_hubs, 1.0 / max(count_nodes, 1))
            else:
                next_hubs = next_hubs / h_norm

            # Numerical safety
            if (
                torch.isnan(next_authorities).any()
                or torch.isinf(next_authorities).any()
                or torch.isnan(next_hubs).any()
                or torch.isinf(next_hubs).any()
            ):
                self.log("[HITS] Numerical instability encountered; stopping early.")
                break

            # Convergence diagnostics
            a_diff = torch.sum(torch.abs(next_authorities - authorities)).item()
            h_diff = torch.sum(torch.abs(next_hubs - hubs)).item()

            if self.verbose and iteration % 10 == 0:
                self.log(f"[HITS] Iteration {iteration}: a_diff={a_diff:.2e}, h_diff={h_diff:.2e}")

            if a_diff < self.tol and h_diff < self.tol:
                self.log(f"[HITS] Converged at iteration {iteration}")
                authorities, hubs = next_authorities, next_hubs
                break

            authorities, hubs = next_authorities, next_hubs

        return hubs, authorities


def compute_hits(
    graph: nx.DiGraph,
    max_iter: int = 100,
    tol: float = 1e-9,
    weights: str | None = "weights",
    verbose: bool = False,
) -> tuple[dict[object, float], dict[object, float]]:
    """
    Convenience wrapper for HITS that returns dict-based scores.
    """
    scorer = HITSScorer(max_iter=max_iter, tol=tol, verbose=verbose)
    return scorer.score(graph, weights)


########################################################################################################################


def hits_scores(
    graph: nx.DiGraph,
    max_iter: int = 100,
    tol: float = 1e-9,
    use_weight: bool | None = None,
    verbose: bool = True,
    **kwargs: Any,
) -> tuple[dict[object, float], dict[object, float]]:
    """
    High-level HITS interface used by the experiment pipeline.

    HITS iteratively alternates:
    - authority <- sum of hub scores of in-neighbors
    - hub       <- sum of authority scores of out-neighbors

    followed by normalization each step until convergence.

    Notes
    -----
    - `use_weight` controls whether edge attribute `'weights'` is used.
    - On certain directed structures, HITS can oscillate; this implementation
    relies on normalization and tolerance checks to stop in a stable state.
    - Disconnected / isolated nodes tend to get small or uniform scores,
    depending on the garph structure and normalization.
    """
    if use_weight is None and "weights" in kwargs:
        use_weight = bool(kwargs["weights"])
    if use_weight is None:
        use_weight = True

    started = time.perf_counter()
    weight_key = "weights" if use_weight else None

    hubs, auths = compute_hits(
        graph=graph,
        max_iter=max_iter,
        tol=tol,
        weights=weight_key,  # str name or None
        verbose=verbose,
    )
    elapsed = time.perf_counter() - started
    tqdm.write(
        f"[HITS] iter<={max_iter}, tol={tol:.1e}, "
        f"weighted={use_weight} | time={elapsed:.3f}s"
    )
    return hubs, auths


########################################################################################################################


def run_baselines_parallel(
    graph: nx.DiGraph,
    hits_params: dict[str, Any] | None = None,
    pr_params: dict[str, Any] | None = None,
) -> tuple:
    """
    Run HITS and PageRank with (potentially) different parameter settings.

    Notes
    -----
    - Currently both methods are run on the same directed graph.
    If you ever want HITS on a symmetrized/undirected version, you
    can transform the graph before passing it here.
    - `hits_params` and `pr_params` are forwarded to `hits_scores` and
    `pagerank_scores` respectively; they typically contain:
    • max_iter, tol, use_weight, etc.
    """
    if hits_params is None:
        hits_params = {}
    if pr_params is None:
        pr_params = {}

    hubs, auths = hits_scores(graph, **hits_params)
    pr = pagerank_scores(graph, **pr_params)
    return pr, hubs, auths
