/**
 * @file NormalizeGraph.jsx
 * @description
 * Universal graph normalizer.
 *
 * Purpose:
 *  - Accepts multiple JSON graph schemas
 *  - Converts them into a single canonical shape:
 *
 *    {
 *      nodes: [{ id, label, data }],
 *      edges: [{ source, target, id?, data }],
 *      debug: { reason? }
 *    }
 *
 * Why this exists:
 *  - Real-world graph exports are inconsistent
 *  - Upstream code (metrics, encoders, UI) must NEVER care about format
 *  - All heuristics live here, once
 *
 * This file is intentionally:
 *  - framework-agnostic
 *  - defensive
 *  - tolerant of partial / malformed inputs
 */

export function normalizeGraph(raw) {
  /** Node registry (id → normalized node) */
  const NODES = new Map();

  /** Flat edge list */
  const EDGES = [];

  /**
   * addNode
   *
   * Registers a node if not already present.
   *
   * Responsibilities:
   *  - Normalize node ID to string
   *  - Extract label from multiple possible fields
   *  - Merge metrics + display metadata into `data`
   *
   * @param {string|number} id
   * @param {object} payload
   */
  const addNode = (id, payload = {}) => {
    if (!id && id !== 0) return;

    const key = String(id);
    if (NODES.has(key)) return;

    /**
     * base
     * Metrics often live under:
     *  - payload.properties
     *  - payload.data
     *  - payload directly
     */
    const base = payload.properties ?? payload.data ?? payload;

    /**
     * label
     * Display label resolution priority:
     *  - label.text
     *  - label
     *  - name
     *  - fallback to id
     */
    const label =
      payload.label?.text ??
      payload.label ??
      payload.name ??
      key;

    /**
     * data
     * Contains *everything* panels and encoders might need.
     * This prevents having to look back at raw node objects later.
     */
    const data = {
      ...base,

      // UI-facing extras
      tooltip: payload.tooltip ?? base.tooltip,
      group: payload.group ?? base.group,
      size: payload.size ?? base.size,
      color: payload.color ?? base.color,

      // Consistent title used by tooltips / panels
      title: base.title ?? label,
    };

    NODES.set(key, {
      id: key,
      label,
      data,
    });
  };

  /**
   * addEdge
   *
   * Registers an edge and ensures its endpoints exist.
   *
   * Responsibilities:
   *  - Normalize source/target to strings
   *  - Preserve edge-level metrics in `data`
   *  - Attach width/arrow if present
   *
   * @param {string|number} s
   * @param {string|number} t
   * @param {object} payload
   * @param {string|number} id
   */
  const addEdge = (s, t, payload = {}, id) => {
    if (s == null || t == null) return;

    const source = String(s);
    const target = String(t);

    // Ensure endpoint nodes exist
    addNode(source);
    addNode(target);

    const base = payload.properties ?? payload.data ?? payload;

    const data = {
      ...base,
      width: payload.width ?? base.width,
      arrow: payload.arrow ?? base.arrow,
    };

    EDGES.push({
      id: id != null ? String(id) : undefined,
      source,
      target,
      data,
    });
  };

  /**
   * isEdgeRecord
   *
   * Heuristic check to detect edge-like objects.
   *
   * Supported conventions:
   *  - id1 / id2
   *  - source / target
   */
  const isEdgeRecord = (o) =>
    o &&
    (("id1" in o && "id2" in o) ||
     ("source" in o && "target" in o));

  /**
   * itemsToNT
   *
   * Handles `{ items: { key: node|edge } }` format.
   */
  const itemsToNT = (items) => {
    for (const [key, val] of Object.entries(items)) {
      if (isEdgeRecord(val)) {
        const s = val.id1 ?? val.source;
        const t = val.id2 ?? val.target;
        addEdge(s, t, val, key);
      } else {
        addNode(key, val);
      }
    }
  };

  /**
   * MAIN NORMALIZATION LOGIC
   *
   * Each branch handles a known graph export pattern.
   * Order matters: most explicit → most heuristic.
   */
  try {
    if (raw?.items && typeof raw.items === "object") {
      // { items: { key: node|edge } }
      itemsToNT(raw.items);

    } else if (Array.isArray(raw?.nodes) || Array.isArray(raw?.edges)) {
      // { nodes: [...], edges: [...] }
      (raw.nodes ?? []).forEach((n) => {
        const id = n.id ?? n.key ?? n.name ?? n._id;
        addNode(id, n);
      });

      (raw.edges ?? []).forEach((e, i) => {
        const s = e.source ?? e.id1 ?? e.from ?? e.u;
        const t = e.target ?? e.id2 ?? e.to ?? e.v;
        addEdge(s, t, e, e.id ?? i);
      });

    } else if (Array.isArray(raw)) {
      // Mixed array heuristic
      raw.forEach((it, i) => {
        if (isEdgeRecord(it)) {
          addEdge(it.id1 ?? it.source, it.id2 ?? it.target, it, i);
        } else {
          addNode(it.id ?? it.key ?? it.name ?? i, it);
        }
      });

    } else if (raw && typeof raw === "object") {
      // Flat object with optional edges array
      if (raw.edges && Array.isArray(raw.edges)) {
        Object.keys(raw).forEach((k) => {
          if (k !== "edges") addNode(k, raw[k]);
        });
        raw.edges.forEach((e, i) => {
          addEdge(e.source ?? e.id1, e.target ?? e.id2, e, e.id ?? i);
        });
      } else {
        // Last-resort: treat every entry as node
        Object.entries(raw).forEach(([k, v]) => addNode(k, v));
      }
    }
  } catch (e) {
    return {
      nodes: [],
      edges: [],
      debug: { reason: `Error: normalize ${String(e)}` },
    };
  }

  // Nothing usable found
  if (NODES.size === 0 && EDGES.length === 0) {
    return {
      nodes: [],
      edges: [],
      debug: { reason: "Error: no nodes or edges found" },
    };
  }

  return {
    nodes: [...NODES.values()],
    edges: EDGES,
    debug: {},
  };
}
