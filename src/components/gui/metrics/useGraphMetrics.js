import { useMemo } from "react";
import { computeGraphMetrics } from "./computeGraphMetrics";
import { makeNodeColorEncoder } from "../encoders/nodeColorEncoder";
import { makeLinkColorEncoder } from "../encoders/linkColorEncoder";
import { makeLinkWidthEncoder } from "../encoders/linkWidthEncoder";
import { makeNodeSizeEncoder } from "../encoders/nodeSizeEncoder";

/**
 * useGraphMetrics
 *
 * @param {Array} nodes
 * @param {Array} edges
 * @param {Set<number>} selectedCommunities
 * @param {string|number|null} selectedNodeId
 * @param {Object|null} scoreRanges - Optional research filters
 *
 * @returns {Object}
 *  - filteredNodes
 *  - filteredEdges
 *  - nodeMetrics (augmented with encoder functions)
 *  - communities
 *  - communityColors
 */
export const useGraphMetrics = (
  nodes,
  edges,
  selectedCommunities,
  selectedNodeId = null,
  scoreRanges = null
) => {
  /* ============================================================
   * BASE METRICS (PURE, MEMOIZED)
   * ============================================================ */
  const base = useMemo(
    () => computeGraphMetrics(nodes, edges, selectedCommunities),
    [nodes, edges, selectedCommunities]
  );

  const {
    filteredNodes: baseNodes,
    filteredEdges: baseEdges,
    nodeMetrics,
    communities,
    communityColors,
    maxImportance,
    maxDegreeTotal,
    sizeStats,
  } = base;

  /* ============================================================
   * SCORE-RANGE FILTERING (RESEARCH MODE)
   * ============================================================ */
  const ranges = scoreRanges || {
    importance: [-Infinity, Infinity],
    pagerank: [-Infinity, Infinity],
    degree: [-Infinity, Infinity],
  };

  /**
   * inRange
   *
   * Defensive numeric range check.
   * Purpose:
   *  - Prevent NaNs from leaking into filters
   */
  const inRange = (v, [mn, mx]) => {
    const x = Number(v);
    if (!Number.isFinite(x)) return false;
    if (Number.isFinite(mn) && x < mn) return false;
    if (Number.isFinite(mx) && x > mx) return false;
    return true;
  };

  const filteredNodes = useMemo(() => {
    return baseNodes.filter((n) => {
      const m = nodeMetrics[String(n.id)];
      if (!m) return false;

      const degTot =
        (m.deg_in ?? m.degreeIn ?? 0) +
        (m.deg_out ?? m.degreeOut ?? 0);

      return (
        inRange(m.importance ?? 0, ranges.importance) &&
        inRange(m.pagerank ?? 0, ranges.pagerank) &&
        inRange(degTot, ranges.degree)
      );
    });
  }, [baseNodes, nodeMetrics, ranges]);

  /* ============================================================
   * EDGE FILTERING (KEEP GRAPH CONSISTENT)
   * ============================================================ */
  const visibleNodeIds = useMemo(
    () => new Set(filteredNodes.map((n) => String(n.id))),
    [filteredNodes]
  );

  const filteredEdges = useMemo(() => {
    return baseEdges.filter((e) => {
      const s = String(e.source);
      const t = String(e.target);
      return visibleNodeIds.has(s) && visibleNodeIds.has(t);
    });
  }, [baseEdges, visibleNodeIds]);

  /* ============================================================
   * NEIGHBOR SET (FOR INTERACTION CONTEXT)
   * ============================================================ */
  const neighborsOfSelected = useMemo(() => {
    if (selectedNodeId == null) return new Set();

    const sid = String(selectedNodeId);
    const neigh = new Set();

    for (const e of filteredEdges) {
      const s = String(e.source);
      const t = String(e.target);
      if (s === sid) neigh.add(t);
      if (t === sid) neigh.add(s);
    }

    return neigh;
  }, [filteredEdges, selectedNodeId]);

  const hasSelection = selectedNodeId != null;

  /* ============================================================
   * VISUAL ENCODERS (PURE FACTORIES)
   * ============================================================ */
  const nodeColor = useMemo(
    () =>
      makeNodeColorEncoder({
        nodeMetrics,
        communityColors,
        maxImportance,
        hasSelection,
        neighborsOfSelected,
        selectedNodeId: hasSelection ? String(selectedNodeId) : null,
      }),
    [
      nodeMetrics,
      communityColors,
      maxImportance,
      hasSelection,
      neighborsOfSelected,
      selectedNodeId,
    ]
  );

  const linkColor = useMemo(
    () =>
      makeLinkColorEncoder({
        nodeMetrics,
        communityColors,
        maxImportance,
        hasSelection,
        selectedNodeId: hasSelection ? String(selectedNodeId) : null,
        neighborsOfSelected,
      }),
    [
      nodeMetrics,
      communityColors,
      maxImportance,
      hasSelection,
      selectedNodeId,
      neighborsOfSelected,
    ]
  );

  const linkWidth = useMemo(
    () =>
      makeLinkWidthEncoder({
        nodeMetrics,
        maxImportance,
        hasSelection,
        selectedNodeId: hasSelection ? String(selectedNodeId) : null,
      }),
    [nodeMetrics, maxImportance, hasSelection, selectedNodeId]
  );

  const nodeSize = useMemo(
    () =>
      makeNodeSizeEncoder({
        nodeMetrics,
        sizeStats,
        maxImportance,
        maxDegreeTotal,
        hasSelection,
        neighborsOfSelected,
        selectedNodeId: hasSelection ? String(selectedNodeId) : null,
      }),
    [
      nodeMetrics,
      sizeStats,
      maxImportance,
      maxDegreeTotal,
      hasSelection,
      neighborsOfSelected,
      selectedNodeId,
    ]
  );

  /* ============================================================
   * RETURN COMPOSED RESULT
   * ============================================================ */
  return {
    filteredNodes,
    filteredEdges,
    nodeMetrics: {
      ...nodeMetrics,
      nodeColor,
      nodeSize,
      linkColor,
      linkWidth,
      maxImportance,
      maxDegreeTotal,
      sizeStats,
    },
    communities,
    communityColors,
  };
};
