import { hexToRgba } from "../../filters/graph/hextorgba";
import { VISUAL_CONFIG } from "./config";

/**
 * makeLinkColorEncoder
 *
 * Factory function that returns a `link => color` encoder.
 * Encapsulates all context needed for color decisions.
 */
function makeLinkColorEncoder({
  nodeMetrics,
  communityColors,
  maxImportance,
  hasSelection,
  selectedNodeId,
  neighborsOfSelected,
  config = VISUAL_CONFIG
}) {
  const { OPACITY, COLORS } = config;

  return function linkColor(link) {
    const s = nodeMetrics[link.source];
    const t = nodeMetrics[link.target];

    // Missing data → safe fallback
    if (!s || !t) {
      return hexToRgba(COLORS.DEFAULT, OPACITY.NORMAL_MIN);
    }

    /* ------------------------------------------------------------
     * Importance factor (0..1)
     * Purpose:
     *  - Shared boost mechanism across encoders
     *  - Prevents dominance by raw values
     * ------------------------------------------------------------ */
    const importanceFactor = Math.max(
      (s.importance || 0) / Math.max(maxImportance, 1),
      (t.importance || 0) / Math.max(maxImportance, 1)
    );

    /* ------------------------------------------------------------
     * Selection state checks
     * ------------------------------------------------------------ */
    const isFromSelected = link.source === selectedNodeId;
    const isToSelected = link.target === selectedNodeId;
    const isSelectedLink = isFromSelected || isToSelected;

    /* ============================================================
     * NORMAL MODE (no selection OR unrelated links)
     * ============================================================ */
    if (!hasSelection || !isSelectedLink) {
      const sameCommunity = s.community === t.community;

      if (sameCommunity) {
        // Same-community links inherit community color
        const color =
          communityColors[s.community % communityColors.length] ||
          COLORS.DEFAULT;

        const opacity = Math.min(
          OPACITY.COMMUNITY_MIN +
            importanceFactor * OPACITY.IMPORTANCE_BOOST,
          OPACITY.COMMUNITY_MAX
        );

        return hexToRgba(color, opacity);
      }

      // Cross-community links are neutral gray
      const opacity = Math.min(
        OPACITY.NORMAL_MIN +
          importanceFactor * OPACITY.IMPORTANCE_BOOST,
        OPACITY.NORMAL_MAX
      );

      return hexToRgba(COLORS.INTER_COMMUNITY, opacity);
    }

    /* ============================================================
     * SELECTION MODE
     * ============================================================ */

    // Directional encoding
    if (isFromSelected) {
      return hexToRgba(COLORS.OUTGOING, OPACITY.SELECTED);
    }

    if (isToSelected) {
      return hexToRgba(COLORS.INCOMING, OPACITY.SELECTED);
    }

    // Neighbor-to-neighbor links
    const inNeighborhood =
      neighborsOfSelected?.has(link.source) &&
      neighborsOfSelected?.has(link.target);

    if (inNeighborhood) {
      const opacity = Math.min(
        OPACITY.NEIGHBOR +
          importanceFactor * OPACITY.IMPORTANCE_BOOST,
        0.95
      );
      return hexToRgba(COLORS.NEIGHBOR, opacity);
    }

    // Everything else fades into the background
    return hexToRgba(COLORS.BACKGROUND, OPACITY.BACKGROUND * 0.3);
  };
}

export { makeLinkColorEncoder };
