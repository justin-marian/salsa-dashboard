import { VISUAL_CONFIG } from "./config";

function makeNodeSizeEncoder({
  nodeMetrics,
  sizeStats,
  maxImportance,
  maxDegreeTotal,
  hasSelection,
  neighborsOfSelected,
  selectedNodeId,
  config = VISUAL_CONFIG
}) {
  const { NODE_SIZE } = config;

  return function nodeSize(node) {
    const metrics = nodeMetrics[node.id];
    if (!metrics) return NODE_SIZE.MIN;

    /* ------------------------------------------------------------
     * Normalize backend-provided size using quantiles
     * Purpose:
     *  - Robust to outliers
     * ------------------------------------------------------------ */
    const rawSize = Number.isFinite(metrics.size)
      ? metrics.size
      : 0;

    let sizeFactor = 0.5;
    if (sizeStats.range > 0) {
      if (rawSize <= sizeStats.q10) sizeFactor = 0;
      else if (rawSize >= sizeStats.q90) sizeFactor = 1;
      else {
        sizeFactor =
          (rawSize - sizeStats.q10) /
          sizeStats.range;
      }
    }

    /* ------------------------------------------------------------
     * Degree metrics
     * ------------------------------------------------------------ */
    const inDegree =
      metrics.deg_in ?? metrics.degreeIn ?? 0;
    const outDegree =
      metrics.deg_out ?? metrics.degreeOut ?? 0;
    const totalDegree = inDegree + outDegree;

    /* ------------------------------------------------------------
     * Normalize importance
     * ------------------------------------------------------------ */
    const importanceFactor = Math.max(
      0,
      (metrics.importance || 0) /
        Math.max(maxImportance, 1)
    );

    /* ------------------------------------------------------------
     * Log-normalized degree (prevents hub domination)
     * ------------------------------------------------------------ */
    let degreeFactor = 0;
    if (maxDegreeTotal > 0) {
      degreeFactor = Math.min(
        1,
        Math.log10(totalDegree + 1) /
          Math.log10(maxDegreeTotal + 1)
      );
    }

    /* ------------------------------------------------------------
     * Weighted fusion
     * ------------------------------------------------------------ */
    const combinedFactor =
      NODE_SIZE.WEIGHTS.SIZE *
        Math.pow(sizeFactor, 0.7) +
      NODE_SIZE.WEIGHTS.DEGREE *
        Math.pow(degreeFactor, 0.6) +
      NODE_SIZE.WEIGHTS.IMPORTANCE *
        Math.pow(importanceFactor, 0.5);

    let size =
      NODE_SIZE.MIN +
      combinedFactor *
        (NODE_SIZE.MAX - NODE_SIZE.MIN);

    /* ------------------------------------------------------------
     * Directionality asymmetry boost
     * Purpose:
     *  - Highlights strong source/sink hubs
     * ------------------------------------------------------------ */
    if (totalDegree > 0) {
      const asymmetry =
        Math.abs(inDegree - outDegree) /
        totalDegree;

      if (Math.max(inDegree, outDegree) >
          totalDegree * 0.75) {
        size +=
          NODE_SIZE.ASYMMETRY_BOOST * asymmetry;
      }
    }

    /* ------------------------------------------------------------
     * Clamp
     * ------------------------------------------------------------ */
    size = Math.max(
      NODE_SIZE.MIN,
      Math.min(size, NODE_SIZE.MAX)
    );

    /* ------------------------------------------------------------
     * Selection effects
     * ------------------------------------------------------------ */
    if (hasSelection) {
      const isSelected =
        node.id === selectedNodeId;
      const isNeighbor =
        neighborsOfSelected.has(node.id);

      if (isSelected) {
        size = Math.max(
          size * NODE_SIZE.SELECTED_MULTIPLIER,
          NODE_SIZE.SELECTED_MIN
        );
      } else if (isNeighbor) {
        size = Math.max(
          size * NODE_SIZE.NEIGHBOR_MULTIPLIER,
          (NODE_SIZE.MIN + NODE_SIZE.MAX) / 1.5
        );
      } else {
        size *= NODE_SIZE.BACKGROUND_MULTIPLIER;
      }
    }

    return size;
  };
}

export { makeNodeSizeEncoder };
