import { useMemo } from "react";

/**
 * getSimulationParams
 *
 * Pure helper that converts simple graph statistics
 * into force-directed simulation parameters.
 *
 * Purpose:
 *  - Automatically adapt layout behavior to graph size & density
 *  - Avoid hardcoded force values
 *  - Keep visual spacing readable under filtering
 *
 * @param {number} nodeCount
 * @param {number} edgeCount
 * @param {number} selectedCommunitiesCount
 * @param {number} totalCommunitiesCount
 * @param {boolean} forceLayout
 *
 * @returns {{
 *   repulsion: number,
 *   gravity: number,
 *   linkDistance: number,
 *   linkSpring: number,
 *   friction: number,
 *   decay: number
 * }}
 */
export const getSimulationParams = (
  nodeCount,
  edgeCount,
  selectedCommunitiesCount,
  totalCommunitiesCount,
  forceLayout
) => {
  const n = Math.max(nodeCount, 1);
  const e = Math.max(edgeCount, 0);

  /**
   * Density proxy:
   *  - Rough indicator of how interconnected the graph is
   */
  const density = e / n;

  /**
   * Complexity:
   *  - Combines size and density
   *  - Grows slowly to avoid extreme jumps
   */
  const sizeScale = Math.log1p(n);
  const densityScale = Math.log1p(density + 1);
  const complexity = sizeScale * (1 + 0.4 * densityScale);

  /**
   * Community selection ratio
   *
   * Purpose:
   *  - Detect filtering
   *  - When many communities are hidden, nodes need more space
   */
  const selectionRatio =
    totalCommunitiesCount > 0
      ? selectedCommunitiesCount / totalCommunitiesCount
      : 1;

  const separationBoost =
    1 + (1 - selectionRatio) * 1.3;

  /**
   * SPACE
   *
   * Controls how “spread out” the layout feels.
   */
  const baseSPACE = 1 + sizeScale * 0.7;
  const densityFactor = 1 + 0.6 * densityScale;

  const SPACE = Math.min(
    14,
    baseSPACE * densityFactor * separationBoost
  );

  /**
   * Repulsion
   *
   * Stronger for:
   *  - large graphs
   *  - dense graphs
   *  - forceLayout mode
   */
  const baseRepulsion = Math.min(
    10,
    1.8 * complexity
  );
  const repulsion = forceLayout
    ? baseRepulsion * 1.5
    : baseRepulsion;

  /**
   * Gravity
   *
   * Pulls nodes back toward center.
   * Reduced as SPACE grows to allow separation.
   */
  const gravity = 0.012 / (SPACE * (1 + 0.3 * densityScale));

  /**
   * Link distance
   *
   * Average preferred edge length.
   * Grows with complexity and filtering.
   */
  const linkDistance = (35 + 12 * complexity) * separationBoost;

  /**
   * Link spring stiffness
   *
   * Softer springs for complex graphs to avoid jitter.
   */
  const linkSpring = 0.25 + 0.2 / (1 + complexity);

  /**
   * Friction & decay
   *
   * Control convergence speed vs stability.
   */
  const friction = 0.88 + 0.08 / (1 + complexity);
  const decay = 0.95 + 0.03 / (1 + complexity);

  return {
    repulsion,
    gravity,
    linkDistance,
    linkSpring,
    friction,
    decay,
  };
};

/**
 * getRenderingParams
 *
 * Computes *visual* rendering limits based on graph size.
 *
 * Purpose:
 *  - Avoid overloading GPU for huge graphs
 *  - Scale quality vs performance automatically
 */
export function getRenderingParams(
  nodeCount,
  edgeCount
) {
  const n = Math.max(nodeCount, 1);
  const e = Math.max(edgeCount, 0);

  const density = e / n;

  const sizeScale = Math.log10(n + 10);
  const densityScale =
    density > 0 ? Math.log10(density + 1) : 0;

  const complexity =
    sizeScale * (1 + 0.3 * densityScale);

  const baseSpace = 8192;

  const spaceSize = Math.min(
    150000,
    Math.max(
      baseSpace,
      Math.round(baseSpace * complexity)
    )
  );

  const pixelRatio = Math.max(
    0.6,
    Math.min(1, 1.1 - 0.15 * complexity)
  );

  const showTopLabels =
    n <= 2000
      ? true
      : n <= 20000
      ? complexity < 2.2
      : false;

  const showTopLabelsLimit = showTopLabels
    ? Math.max(10, Math.round(80 / complexity))
    : 0;

  const renderLinks = n <= 120000;
  const showLinkArrows =
    n <= 15000 && density < 30;

  return {
    spaceSize,
    pixelRatio,
    showTopLabels,
    showTopLabelsLimit,
    renderLinks,
    showLinkArrows,
  };
}

/**
 * useSimulationParams
 *
 * React hook wrapper around getSimulationParams.
 *
 * Purpose:
 *  - Memoize physics parameters
 *  - Recompute only when graph structure or force toggle changes
 */
export const useSimulationParams = (
  nodeCount,
  edgeCount,
  selectedCommunitiesCount,
  totalCommunitiesCount,
  forceLayout
) => {
  return useMemo(
    () =>
      getSimulationParams(
        nodeCount,
        edgeCount,
        selectedCommunitiesCount,
        totalCommunitiesCount,
        forceLayout
      ),
    [
      nodeCount,
      edgeCount,
      selectedCommunitiesCount,
      totalCommunitiesCount,
      forceLayout,
    ]
  );
};
