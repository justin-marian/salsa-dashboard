import { VISUAL_CONFIG } from "./config";

/**
 * makeLinkWidthEncoder
 *
 * Strategy:
 *  - Importance controls baseline thickness
 *  - Selection overrides everything
 *  - Background links collapse to near-invisible
 */
function makeLinkWidthEncoder({
  nodeMetrics,
  maxImportance,
  hasSelection,
  selectedNodeId,
  config = VISUAL_CONFIG
}) {
  const { WIDTH } = config;

  return function linkWidth(link) {
    const s = nodeMetrics[link.source];
    const t = nodeMetrics[link.target];

    if (!s || !t) return WIDTH.MIN;

    const importanceFactor = Math.max(
      (s.importance || 0) / Math.max(maxImportance, 1),
      (t.importance || 0) / Math.max(maxImportance, 1)
    );

    // Base thickness for normal mode
    const baseWidth =
      WIDTH.MIN + importanceFactor * (WIDTH.MAX - WIDTH.MIN);

    if (!hasSelection) {
      return baseWidth;
    }

    const isOutgoing = link.source === selectedNodeId;
    const isIncoming = link.target === selectedNodeId;

    if (isOutgoing) {
      return (
        WIDTH.SELECTED_MIN +
        importanceFactor *
          (WIDTH.SELECTED_MAX - WIDTH.SELECTED_MIN)
      );
    }

    if (isIncoming) {
      return (
        WIDTH.SELECTED_MIN * 0.8 +
        importanceFactor *
          (WIDTH.SELECTED_MAX * 0.8 -
            WIDTH.SELECTED_MIN * 0.8)
      );
    }

    // Background links
    return WIDTH.BACKGROUND;
  };
}

export { makeLinkWidthEncoder };
