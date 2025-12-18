import { hexToRgba } from "../../filters/graph/hextorgba";
import { VISUAL_CONFIG } from "./config";

function makeNodeColorEncoder({
  nodeMetrics,
  communityColors,
  maxImportance,
  hasSelection,
  neighborsOfSelected,
  selectedNodeId,
  config = VISUAL_CONFIG
}) {
  const { COLORS, NODE_COLOR } = config;

  return function nodeColor(node) {
    const metrics = nodeMetrics[node.id];

    if (!metrics) {
      return hexToRgba(
        COLORS.DEFAULT,
        NODE_COLOR.DEFAULT_OPACITY
      );
    }

    const communityId = metrics.community || 0;
    const communityColor =
      communityColors[communityId % communityColors.length] ||
      COLORS.DEFAULT;

    const importanceFactor = Math.max(
      0,
      (metrics.importance || 0) /
        Math.max(maxImportance, 1)
    );

    const isSelected =
      hasSelection && node.id === selectedNodeId;
    const isNeighbor =
      hasSelection && neighborsOfSelected.has(node.id);

    /* ============================================================
     * NORMAL MODE
     * ============================================================ */
    if (!hasSelection) {
      const opacity = Math.min(
        NODE_COLOR.NORMAL_MIN +
          importanceFactor *
            NODE_COLOR.IMPORTANCE_BOOST,
        NODE_COLOR.NORMAL_MAX
      );
      return hexToRgba(communityColor, opacity);
    }

    /* ============================================================
     * SELECTION MODE
     * ============================================================ */
    if (isSelected) {
      return hexToRgba(
        communityColor,
        NODE_COLOR.SELECTED
      );
    }

    if (isNeighbor) {
      return hexToRgba(
        communityColor,
        Math.min(
          NODE_COLOR.NEIGHBOR +
            importanceFactor * 0.15,
          0.95
        )
      );
    }

    return hexToRgba(
      communityColor,
      NODE_COLOR.BACKGROUND * 0.5
    );
  };
}

export { makeNodeColorEncoder };
