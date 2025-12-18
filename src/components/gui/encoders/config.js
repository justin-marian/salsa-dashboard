export const VISUAL_CONFIG = {
  /* ============================================================
    * OPACITY
    * Purpose:
    *  - Control perceptual emphasis via transparency
    *  - Reduce clutter while preserving context
    * ============================================================ */
  OPACITY: {
    SELECTED: 1.0,           // Fully opaque for selected elements
    NEIGHBOR: 0.2,           // Neighbors remain visible but secondary
    COMMUNITY_MIN: 0.1,      // Lower bound for same-community links
    COMMUNITY_MAX: 0.4,      // Upper bound for same-community links
    NORMAL_MIN: 0.05,        // Barely visible links/nodes
    NORMAL_MAX: 0.3,         // Upper bound in non-selected mode
    IMPORTANCE_BOOST: 0.3,   // Extra opacity gained from importance
    BACKGROUND: 0.02         // Almost invisible background elements
  },

  /* ============================================================
   * COLORS
   * Purpose:
   *  - Encode semantic meaning (direction, role, state)
   * ============================================================ */
  COLORS: {
    DEFAULT: "#5B8FC7",          // Fallback blue
    INTER_COMMUNITY: "#B4B8C8",  // Neutral gray for cross-community links
    OUTGOING: "#306cedff",         // Blue: outgoing from selected node
    INCOMING: "#dea112ff",         // Yellow: incoming to selected node
    NEIGHBOR: "#A0AABE",         // Muted highlight for neighbor links
    BACKGROUND: "#646E82",       // Desaturated background color
  },

  /* ============================================================
   * WIDTH
   * Purpose:
   *  - Encode importance and selection in link thickness
   * ============================================================ */
  WIDTH: {
    MIN: 0.35,              // Thin baseline
    MAX: 3.0,               // Upper bound for normal mode
    SELECTED_MIN: 3.2,      // Selected links are always thicker
    SELECTED_MAX: 9.0,      // Maximum emphasis thickness
    BACKGROUND: 0.05,       // Barely visible background links
    SELECTED_BOOST: 4.0,    // Extra thickness for selected context
  },

  /* ============================================================
   * NODE SIZE
   * Purpose:
   *  - Combine multiple signals into a single perceptual radius
   * ============================================================ */
  NODE_SIZE: {
    MIN: 4.0,
    MAX: 30.0,

    // Multipliers applied during selection
    SELECTED_MULTIPLIER: 2.2,
    NEIGHBOR_MULTIPLIER: 1.6,
    BACKGROUND_MULTIPLIER: 0.55,

    SELECTED_MIN: 36,       // Ensures selected node is unmistakable
    ASYMMETRY_BOOST: 2.5,   // Extra size for directional hubs

    // Relative influence of different signals
    WEIGHTS: {
      SIZE: 0.6,            // Backend-provided size
      DEGREE: 0.25,         // Connectivity
      IMPORTANCE: 0.15,     // Semantic importance
    },
  },

  /* ============================================================
   * NODE COLOR
   * Purpose:
   *  - Encode selection and importance via opacity
   * ============================================================ */
  NODE_COLOR: {
    SELECTED: 1.0,
    NEIGHBOR: 0.9,
    NORMAL_MIN: 0.1,
    NORMAL_MAX: 0.5,
    BACKGROUND: 0.03,
    DEFAULT_OPACITY: 0.2,
    IMPORTANCE_BOOST: 0.2
  }
};
