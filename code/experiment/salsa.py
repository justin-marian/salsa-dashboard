import torch
import networkx as nx
from tqdm.auto import tqdm
from typing import Any

from .utils import (
    graph_to_gpu_tensors,
    sparse_matmul,
    normalize_rows,
    get_device,
)


class SalsaSparse:
    """
    GPU-accelerated SALSA (Stochastic Approach for Link-Structure Analysis).
    This class implements SALSA on top of sparse GPU/CPU tensors:

        - Hub update:        h_{t+1} = A · a_t
        - Authority update:  a_{t+1} = A_t · h_t

    with:
        - row-normalized adjacency (stochastic matrix),
        - optional damping towards the uniform distribution,
        - L1-based convergence checks,
        - a simple Krasnosel'skii-style averaging step when residuals get worse.

    The final outputs are:
        - `hub_scores`:  dict[node, float]
        - `auth_scores`: dict[node, float]
        both L1-normalized and rounded to 6 decimal places.

    Attributes
    ----------
    max_iterations:
        Maximum number of SALSA iterations allowed.
    tolerance:
        L1 residual threshold for convergence.
    damping_factor:
        Damping parameter in [0, 1]. If not None and > 0, each update is
        blended with a uniform vector:

            x_next ← (1 - damping) * x_next + damping * (1 / N)

    use_gpu:
        If True, uses CUDA if available (via `get_device()`); otherwise falls
        back to CPU.
    verbose:
        If True, shows a per-iteration tqdm bar and convergence logs.
    nodes:
        List of nodes in the order they correspond to tensor indices.
    hub_scores:
        Mapping node → final SALSA hub score (float).
    auth_scores:
        Mapping node → final SALSA authority score (float).
    iterations:
        Number of iterations actually performed.
    converged:
        True if convergence was detected before hitting `max_iterations`.
    """

    def __init__(
        self,
        max_iter: int = 200,
        tol: float = 1e-9,
        damping: float | None = None,
        use_gpu: bool = True,
        verbose: bool = True,
    ) -> None:
        self.max_iterations = int(max_iter)
        self.tolerance = float(tol)
        self.damping_factor = damping
        self.use_gpu = use_gpu
        self.verbose = verbose

        self.nodes: list[Any] = []
        self.hub_scores: dict[Any, float] = {}
        self.auth_scores: dict[Any, float] = {}
        self.iterations: int = 0
        self.converged: bool = False

    @staticmethod
    def l1_normalize(vector: torch.Tensor) -> torch.Tensor:
        """
        Normalize a vector by its L1 norm, with a safe fallback for zero vectors.

        If the L1 norm is zero (e.g. all entries are zero), we fall back to a
        uniform distribution over entries instead of dividing by zero.
        """
        vector_sum = torch.sum(torch.abs(vector))
        if vector_sum <= 0:
            return torch.ones_like(vector) / vector.numel()
        return vector / vector_sum

    def fit(self, graph: nx.DiGraph, weight: str | None = None) -> "SalsaSparse":
        """
        Run SALSA on the given directed graph.

        Parameters
        ----------
        graph:
            Directed NetworkX graph to analyze.
        weight:
            Name of the edge attribute storing weights. If None or empty,
            the graph is treated as unweighted.

        Returns
        -------
        self:
            The fitted instance, with `hub_scores`, `auth_scores`,
            `converged`, and `iterations` populated.
        """
        # Select device (CPU/GPU) for tensor computations
        device = get_device() if self.use_gpu else "cpu"

        # Convert graph to tensor representation on the chosen device.
        # edge_index: (2, E) long tensor of source/target indices
        # edge_weight: (E,) float tensor of edge weights
        # node_list: list of original node labels (index → node)
        # node_count: number of nodes
        edge_index, edge_weight, node_list, node_count = graph_to_gpu_tensors( graph, weight or "", device)
        self.nodes = node_list

        # Row-normalize the adjacency for stochastic behavior
        edge_weight = normalize_rows(edge_index, edge_weight, node_count)

        # Initialize hub and authority scores to uniform distributions
        authority_scores = torch.ones(node_count, device=device) / node_count
        hub_scores = torch.ones(node_count, device=device) / node_count

        # Precompute transposed edge_index for Aᵀ · h updates
        transposed_edge_index = torch.stack([edge_index[1], edge_index[0]])

        # Iteration loop configuration (with optional tqdm progress bar)
        iteration_range = range(self.max_iterations)
        if self.verbose:
            iteration_range = tqdm(
                iteration_range, desc="SALSA iterations", leave=False
            )

        previous_residual = None
        for current_iteration in iteration_range:
            # Core SALSA updates
            # a_{t+1} = Aᵀ · h_t
            next_authority = sparse_matmul(
                transposed_edge_index, edge_weight, hub_scores, node_count
            )
            # h_{t+1} = A · a_{t+1}
            next_hub = sparse_matmul(
                edge_index, edge_weight, next_authority, node_count
            )

            # Dmping to reduce bipartite oscillations / sinks
            if self.damping_factor and self.damping_factor > 0.0:
                uniform_value = 1.0 / node_count
                next_authority = (
                    (1 - self.damping_factor) * next_authority
                    + self.damping_factor * uniform_value
                )
                next_hub = (
                    (1 - self.damping_factor) * next_hub
                    + self.damping_factor * uniform_value
                )

            # Numerical guards + L1 normalization
            next_authority = torch.clamp(next_authority, min=0)
            next_hub = torch.clamp(next_hub, min=0)
            next_authority = self.l1_normalize(next_authority)
            next_hub = self.l1_normalize(next_hub)

            # Convergence monitoring
            authority_residual = torch.norm(next_authority - authority_scores, p=1)
            hub_residual = torch.norm(next_hub - hub_scores, p=1)
            current_residual = torch.maximum(authority_residual, hub_residual)

            # If residual worsens, apply mild averaging (Krasnosel'skii step)
            if (
                previous_residual is not None
                and current_residual > previous_residual
            ):
                # The previous step made things worse; average the scores
                # This helps stabilize oscillations in bipartite structures
                next_authority = 0.5 * (next_authority + authority_scores)
                next_hub = 0.5 * (next_hub + hub_scores)
                # Re-normalize after averaging (to ensure stochasticity)
                next_authority = self.l1_normalize(next_authority)
                next_hub = self.l1_normalize(next_hub)
                current_residual = torch.maximum(
                    torch.norm(next_authority - authority_scores, p=1),
                    torch.norm(next_hub - hub_scores, p=1),
                )

            # Check convergence against L1 tolerance
            # Useful for checking if the method has stabilized enough
            if current_residual < self.tolerance:
                self.iterations = current_iteration + 1
                self.converged = True
                authority_scores, hub_scores = next_authority, next_hub
                if self.verbose:
                    tqdm.write(
                        f"[SALSA] Converged in {self.iterations} iterations "
                        f"(tolerance={self.tolerance:.1e})"
                    )
                break

            # Prepare for next iteration
            authority_scores, hub_scores = next_authority, next_hub
            previous_residual = current_residual
        else:
            # Loop exhausted without breaking → no convergence within max_iterations
            self.iterations = self.max_iterations
            self.converged = False
            if self.verbose:
                tqdm.write(
                    f"[SALSA] Maximum iterations ({self.max_iterations}) reached "
                    f"without convergence"
                )

        # Map back from tensor indices to original nodes and round for readability
        self.auth_scores = {
            node_list[i]: round(authority_scores[i].item(), 6)
            for i in range(node_count)
        }
        self.hub_scores = {
            node_list[i]: round(hub_scores[i].item(), 6)
            for i in range(node_count)
        }
        return self
