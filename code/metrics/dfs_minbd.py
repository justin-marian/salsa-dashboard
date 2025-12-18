from enum import Enum
from typing import Any
import networkx as nx
from tqdm.auto import tqdm


class NodeVisitStatus(Enum):
    """Tracks node visitation state during DFS traversal."""
    UNVISITED = 0   # Not yet visited
    VISITING = 1    # Currently on the DFS stack (in active path)
    VISITED = 2     # Fully processed, all descendants explored


class DFSMinBD:
    """
    Minimum Backward Distance (MinBD) via DFS-based cycle analysis.

    Goal
    ----
    For every directed edge (u → v) in a graph, we want the *shortest*
    number of steps it takes to get from v back to u (if possible).
    This is the "minimal backward distance" of that edge.

        MinBD(u → v) = length of the shortest directed path v ⇒ u
                    (capped at `max_distance`, or None if unreachable)

    Intuition
    ---------
    - Edges that participate in short directed cycles (u → v → ... → u)
    get small MinBD, meaning they are embedded in tight feedback loops.
    - Edges that lead into large or no-return regions get large MinBD or None.

    Approach
    --------
    We run a DFS over the graph and:
      * Use an explicit stack (iterative DFS) to avoid recursion depth issues.
      * Track the current DFS path (`current_dfs_path`) and a `path_index`
        (node → index in that path) for O(1) ancestor checks.
      * When we encounter:
        - a back edge (to a node in VISITING state), we have a simple cycle
        and can update distances along that cycle.
        - cross/forward edges connected to previously discovered cycles, we
        can combine distances from the current path to those cycle ancestors.

    All computed distances are stored in:
        minimal_backward_distances[(u, v)] → int | None

    and clamped at `max_distance` to limit blowup on large graphs.
    """

    def __init__(self, graph: nx.DiGraph, max_distance: int = 30, verbose: bool = False):
        self.graph = graph
        self.max_distance = max_distance
        self.verbose = verbose

        # Per-node DFS state
        self.node_status: dict[Any, NodeVisitStatus] = {}
        self.current_dfs_path: list[Any] = []      # active DFS path (stack of nodes)
        self.path_index: dict[Any, int] = {}       # node → index in `current_dfs_path`
        self.node_depths: dict[Any, int] = {}      # optional depth info
        self.parent_nodes: dict[Any, Any] = {}     # node → parent in DFS tree

        # Per-edge and per-node-cycle information
        self.minimal_backward_distances: dict[tuple[Any, Any], int | None] = {}
        # For each node, we can record distances to cycle ancestors:
        #     cycle_ancestor_distances[x][a] = minimal distance from x to ancestor a
        self.cycle_ancestor_distances: dict[Any, dict[Any, int]] = {}
        self.initialize_structs()

    def minimal_backward_distance(self) -> dict[tuple[Any, Any], int | None]:
        """
        Compute MinBD for all edges in the graph.

        Returns
        -------
        dict[(u, v), int | None]
            Mapping each edge to its minimal backward distance:
            - finite int ≤ max_distance if there is a return path v ⇒ u,
            - None if no such path was discovered.
        """
        self.initialize_structs()
        for node in self.graph.nodes():
            if self.node_status[node] == NodeVisitStatus.UNVISITED:
                self.dfs_cycle_detection(node)
        self.finalize_results()
        return self.minimal_backward_distances

    def get_statistics(self) -> dict[str, float | int]:
        """
        Summary statistics over the current MinBD results.

        Returns a dict with:
            - total_edges
            - edges_with_finite_distance
            - edges_with_infinite_distance
            - finite_ratio / infinite_ratio
            - min_distance / max_distance / average_distance (for finite ones)
            - edges_within_max_distance / within_max_ratio
        """
        if not self.minimal_backward_distances:
            return {}

        total = len(self.minimal_backward_distances)
        finite = [d for d in self.minimal_backward_distances.values() if d is not None]
        finite_n = len(finite)  # count of finite distances (ints)
        inf_n = total - finite_n  # count of None distances (infinite)

        stats: dict[str, float | int] = {
            "total_edges": total,
            "edges_with_finite_distance": finite_n,
            "edges_with_infinite_distance": inf_n,
            "finite_ratio": (finite_n / total) if total else 0.0,
            "infinite_ratio": (inf_n / total) if total else 0.0,
        }

        if finite_n:
            within_max = sum(1 for d in finite if d <= self.max_distance)
            stats.update({
                "min_distance": min(finite),
                "max_distance": max(finite),
                "average_distance": sum(finite) / finite_n,
                "edges_within_max_distance": within_max,
                "within_max_ratio": within_max / finite_n,
            })

        return stats

    def reset_computation(self) -> None:
        """
        Reset all computation state for a new run, but keep the same graph.
        This clears:
            - DFS visitation state
            - recorded backward distances
            - path/parent/ancestor structures
        """
        for node in self.graph.nodes():
            self.node_status[node] = NodeVisitStatus.UNVISITED
            self.cycle_ancestor_distances[node] = {}

        for edge in self.graph.edges():
            self.minimal_backward_distances[edge] = None

        self.current_dfs_path.clear()
        self.path_index.clear()
        self.node_depths.clear()
        self.parent_nodes.clear()

    def update_path_cycle(self, cycle_start: Any, cycle_end: Any, cycle_length: int) -> None:
        """
        Update backward distances along a detected cycle path.

        We assume that there is a DFS path:
            cycle_start ⇒ ... ⇒ cycle_end

        with length `cycle_length`, and that from `cycle_end` we saw an edge
        that closes the cycle back to `cycle_start`.

        We walk backwards from `cycle_end` to `cycle_start`, updating:
        - minimal_backward_distances for edges along this cycle,
        - cycle_ancestor_distances to help future cross-edge updates.
        """
        current = cycle_end
        dist = cycle_length

        # Ensure entries exist for endpoints in the ancestor map
        self.cycle_ancestor_distances.setdefault(cycle_end, {})
        self.cycle_ancestor_distances.setdefault(cycle_start, {})

        while current != cycle_start:
            prev = self.parent_nodes.get(current)
            if prev is None:
                # Broken parent chain (e.g., root or inconsistency)
                break

            edge = (prev, current)
            cur_val = self.minimal_backward_distances.get(edge)
            if (cur_val is None) or (dist < cur_val):
                # Clamp to max_distance to avoid unbounded growth
                self.minimal_backward_distances[edge] = min(dist, self.max_distance)

            # Record ancestor distance for cross-edge processing
            cad = self.cycle_ancestor_distances.setdefault(current, {})
            anc_val = cad.get(cycle_start)
            if (anc_val is None) or (dist < anc_val):
                cad[cycle_start] = dist

            current = prev
            dist += 1

        # Optionally also update the edge into cycle_start if it has a parent
        prev_start = self.parent_nodes.get(cycle_start)
        if prev_start is not None:
            final_edge = (prev_start, cycle_start)
            cur_val = self.minimal_backward_distances.get(final_edge)
            if (cur_val is None) or (dist < cur_val):
                self.minimal_backward_distances[final_edge] = min(dist, self.max_distance)

    def dfs_cycle_detection(self, root: Any, depth: int = 0) -> None:
        """
        Iterative DFS to detect cycles and compute minimal backward distances.

        We maintain:
            - an explicit `stack` of (node, neighbor-iterator),
            - `current_dfs_path` to know which nodes form the current root→…→node path,
            - `path_index` for O(1) “is this node an ancestor on the current path?”

        When we discover:
            - a node in UNVISITED: it's a tree edge → extend DFS.
            - a node in VISITING: it's a back edge → direct cycle to an ancestor.
            - a node in VISITED but with known cycle ancestors: treat as cross/forward
            edge and update MinBD using previously recorded ancestor distances.
        """
        self.node_status[root] = NodeVisitStatus.VISITING
        self.node_depths[root] = depth
        self.parent_nodes[root] = self.parent_nodes.get(root, None)
        self.current_dfs_path.append(root)
        self.path_index[root] = 0

        if self.verbose and depth == 0:
            visited_count = sum(1 for s in self.node_status.values() if s != NodeVisitStatus.UNVISITED)
            tqdm.write(f"DFS from root {root}, progress: {visited_count}/{len(self.node_status)} nodes")

        # Stack items: (current_node, iterator over its neighbors)
        stack: list[tuple[Any, Any]] = [(root, iter(self.graph.neighbors(root)))]

        while stack:
            node, it = stack[-1]
            try:
                nbr = next(it)
            except StopIteration:
                # Finished exploring all neighbors of `node` → backtrack
                stack.pop()
                if self.current_dfs_path:
                    self.current_dfs_path.pop()
                self.path_index.pop(node, None)
                self.node_status[node] = NodeVisitStatus.VISITED
                continue

            status = self.node_status.get(nbr, NodeVisitStatus.UNVISITED)

            if status == NodeVisitStatus.UNVISITED:
                # Tree edge discovery: extend DFS into `nbr`
                self.parent_nodes[nbr] = node
                self.node_status[nbr] = NodeVisitStatus.VISITING
                self.node_depths[nbr] = self.node_depths[node] + 1
                self.current_dfs_path.append(nbr)
                self.path_index[nbr] = len(self.current_dfs_path) - 1
                stack.append((nbr, iter(self.graph.neighbors(nbr))))

            elif status == NodeVisitStatus.VISITING:
                # Back edge to an ancestor on the current path → direct cycle
                idx = self.path_index.get(nbr)
                if idx is not None:
                    cycle_len = len(self.current_dfs_path) - idx
                    if cycle_len > 0:
                        self.update_path_cycle(nbr, node, cycle_len)

            else:
                # status == VISITED: potential cross/forward edge
                # If `nbr` knows about cycle ancestors, we can combine distances.
                cad = self.cycle_ancestor_distances.get(nbr)
                if cad:
                    for ancestor, dist in cad.items():
                        a_idx = self.path_index.get(ancestor)
                        if a_idx is not None:  # ancestor currently on the path
                            total_dist = dist + 1 + (len(self.current_dfs_path) - a_idx)
                            self.update_path_cycle(ancestor, node, total_dist)

    def proc_back_edges(self, source: Any, target: Any) -> None:
        """
        Compatibility stub for back-edge handling using `path_index`.

        Provided for API similarity with other MinBD implementations.
        Currently, `dfs_cycle_detection` already calls `update_path_cycle`
        directly for back edges, so this is a thin wrapper.
        """
        idx = self.path_index.get(target)
        if idx is not None:
            cycle_len = len(self.current_dfs_path) - idx
            if cycle_len > 0:
                self.update_path_cycle(target, source, cycle_len)

    def proc_cross_edges(self, source: Any, target: Any) -> None:
        """
        Compatibility stub for cross/forward edges using cycle ancestor info.

        If `target` has known cycle ancestors, and some ancestor is on the
        current DFS path in VISITING state, we can combine distances and
        update MinBD using `update_path_cycle`.
        """
        cad = self.cycle_ancestor_distances.get(target)
        if not cad:
            return
        for ancestor, dist in cad.items():
            a_idx = self.path_index.get(ancestor)
            if a_idx is not None and self.node_status.get(ancestor) == NodeVisitStatus.VISITING:
                total_dist = dist + 1 + (len(self.current_dfs_path) - a_idx)
                self.update_path_cycle(ancestor, source, total_dist)

    def initialize_structs(self) -> None:
        """
        Initialize/clear all data structures for a fresh run.

        This is called in __init__ and before each new MinBD computation.
        """
        self.current_dfs_path.clear()
        self.path_index.clear()
        self.node_depths.clear()
        self.parent_nodes.clear()
        self.minimal_backward_distances.clear()
        self.cycle_ancestor_distances.clear()

        for node in self.graph.nodes():
            self.node_status[node] = NodeVisitStatus.UNVISITED
            self.cycle_ancestor_distances[node] = {}

        for u, v in self.graph.edges():
            self.minimal_backward_distances[(u, v)] = None

    def finalize_results(self) -> None:
        """
        Clamp all finite distances at `max_distance`.

        This ensures that extremely long inferred cycles don't produce
        arbitrarily large MinBD values; they are all capped.
        """
        for e, d in list(self.minimal_backward_distances.items()):
            if isinstance(d, int) and d > self.max_distance:
                self.minimal_backward_distances[e] = self.max_distance
