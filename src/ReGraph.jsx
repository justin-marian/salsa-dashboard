import { useEffect, useRef, useState, useCallback } from "react";
import ExCosmoGraph from "./components/ExCosmoGraph";
import ValidateJson from "./components/interface/ValidateJson";
import { normalizeGraph } from "./components/interface/NormalizeGraph";

/**
 * ReGraph.jsx
 *
 * Purpose:
 * - This component is the "data ingestion + orchestration" layer.
 * - It fetches graph JSON from `url`, parses it safely, normalizes it, and only then
 *   renders the visualization layer (ExCosmoGraph).
 *
 * Responsibilities:
 * - Network lifecycle (fetch + abort previous requests)
 * - Progressive loading UI (progress bar)
 * - Robust JSON parsing (streaming when supported)
 * - Format normalization (convert arbitrary schemas into {nodes, edges})
 * - Error handling (show friendly failure UI)
 * - Diagnostics (node/edge counts, load + parse timing)
 *
 * What this component does NOT do:
 * - It does not compute metrics (that happens in computeGraphMetrics/useGraphMetrics).
 * - It does not implement visual encoders (nodeColor/nodeSize/linkColor/etc).
 * - It does not contain Cosmograph configuration (handled in ExCosmoGraph).
 */

/* ---------- Helpers for safe JSON detection ---------- */

/**
 * firstNonWsChar
 *
 * Purpose:
 * - Returns the first non-whitespace character from a string.
 * - Used for early detection of HTML responses (often start with "<")
 *   masquerading as JSON (e.g., error pages, redirects, reverse proxy errors).
 *
 * @param {string} s
 * @returns {string} first non-whitespace char or "" if none
 */
function firstNonWsChar(s) {
  for (let i = 0; i < s.length; i++) {
    if (s[i] > " ") return s[i];
  }
  return "";
}

/**
 * stripBOM
 *
 * Purpose:
 * - Some JSON files include a UTF-8 BOM (Byte Order Mark).
 * - JSON.parse can fail if BOM is present in some environments.
 * - This removes BOM defensively.
 *
 * @param {string} s
 * @returns {string}
 */
function stripBOM(s) {
  return s.charCodeAt(0) === 0xfeff ? s.slice(1) : s;
}

/**
 * parseJsonSmart
 *
 * Purpose:
 * - Parse JSON from a fetch Response safely and efficiently.
 * - Uses streaming when available to:
 *    - avoid blocking the main thread for huge files
 *    - allow progress updates (percentage)
 *
 * Safety features:
 * - Detects non-JSON content-type and HTML payloads early
 * - Throws readable errors instead of failing silently
 *
 * @param {Response} response - fetch response
 * @param {(p:number)=>void} onProgress - progress callback (0..100)
 * @returns {Promise<any>} parsed JSON object
 */
async function parseJsonSmart(response, onProgress) {
  // Content-Type detection helps identify bad server responses
  const ct = (response.headers.get("Content-Type") || "").toLowerCase();

  // Accept both common JSON content types
  const isJsonCT =
    ct.includes("application/json") || ct.includes("text/json");

  /**
   * Non-streaming fallback:
   * Purpose:
   * - Some environments / servers may not provide response.body as a readable stream.
   * - In that case, we read the whole response as text.
   */
  if (!response.body) {
    const txt = await response.text();

    // Early HTML detection: HTML pages typically begin with "<"
    if (!isJsonCT || firstNonWsChar(txt) === "<") {
      throw new Error("Expected JSON but received non-JSON content.");
    }

    // BOM-safe parse
    return JSON.parse(stripBOM(txt));
  }

  /**
   * Streaming path:
   * Purpose:
   * - Decode chunks incrementally
   * - Emit progress updates if Content-Length is present
   */
  const reader = response.body.getReader();
  const decoder = new TextDecoder();

  // Total bytes for progress ratio (may be NaN if header missing)
  const total = +response.headers.get("Content-Length");

  let received = 0; // bytes processed
  let chunks = "";  // assembled full JSON text

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    received += value.length;
    const piece = decoder.decode(value, { stream: true });
    chunks += piece;

    // Early HTML detection for streams (prevents waiting to the end)
    if (!isJsonCT && firstNonWsChar(chunks) === "<") {
      throw new Error("Non-JSON stream detected.");
    }

    // Emit progress if total size is known
    if (onProgress && total) {
      onProgress((received / total) * 100);
    }
  }

  // Final parse (BOM-safe)
  return JSON.parse(stripBOM(chunks));
}

/**
 * CitationsReGraph
 *
 * Purpose:
 * - The boundary component between:
 *    - "data world" (fetch/parse/normalize)
 *    - "visualization world" (Cosmograph + UI overlays)
 *
 * @param {Object} props
 * @param {string} props.url - URL to JSON graph export
 * @returns {JSX.Element}
 */
export default function CitationsReGraph({
  url = "/exports/wiki_vote_regraph.json",
}) {
  /* ---------- Data & UI state ---------- */

  /**
   * raw
   * Purpose:
   * - Holds the normalized graph: { nodes, edges }
   * - `null` means "not ready" (loading or not yet fetched)
   */
  const [raw, setRaw] = useState(null);

  /**
   * err
   * Purpose:
   * - Holds user-visible error text for failed fetch/parse/normalize
   */
  const [err, setErr] = useState(null);

  /**
   * progress
   * Purpose:
   * - Drives the progress bar (0..100)
   * - Updated by parseJsonSmart during streaming reads
   */
  const [progress, setProgress] = useState(0);

  /**
   * isLoading
   * Purpose:
   * - Simple loading gate for UI states
   * - Keeps rendering logic clean ("if loading show loader")
   */
  const [isLoading, setIsLoading] = useState(false);

  /**
   * meta
   * Purpose:
   * - Debug/diagnostic metadata displayed in the stats panel
   * - Helps confirm file size/complexity and performance
   */
  const [meta, setMeta] = useState({
    nodes: 0,
    edges: 0,
    reason: "",
    loadTime: 0,
    parseTime: 0,
  });

  /**
   * controllerRef
   * Purpose:
   * - Stores the active AbortController
   * - Allows cancelling in-flight requests when:
   *    - URL changes
   *    - component unmounts
   * - Prevents race conditions (old request finishing after new request)
   */
  const controllerRef = useRef(null);

  /**
   * loadGraphData
   *
   * Purpose:
   * - Executes the full pipeline:
   *    1) abort old request
   *    2) reset UI state
   *    3) fetch JSON
   *    4) parse JSON safely (streaming when possible)
   *    5) normalize graph shape
   *    6) store {nodes, edges} in state
   *
   * Why useCallback:
   * - Stable function identity for useEffect deps
   * - Prevents accidental refetch loops
   */
  const loadGraphData = useCallback(async () => {
    // Abort any previous fetch to avoid overlapping updates
    if (controllerRef.current) controllerRef.current.abort();

    // Create a new controller for the new request
    const controller = new AbortController();
    controllerRef.current = controller;

    // Reset UI for a clean loading state
    setIsLoading(true);
    setErr(null);
    setProgress(0);
    setRaw(null);

    // Start timing the whole pipeline
    const t0 = performance.now();

    /**
     * Cache-bust:
     * Purpose:
     * - Avoid browser/CDN returning stale JSON
     * - Particularly important while iterating on exports
     */
    const urlBust =
      url + (url.includes("?") ? "&" : "?") + "_ts=" + Date.now();

    try {
      /**
       * Fetch:
       * - cache: "no-store" tries to prevent caching
       * - Accept header nudges servers toward JSON
       * - signal enables abort
       */
      const response = await fetch(urlBust, {
        signal: controller.signal,
        cache: "no-store",
        headers: { Accept: "application/json" },
      });

      // HTTP errors should become readable UI errors
      if (!response.ok) {
        throw new Error(`HTTP ${response.status} ${response.statusText}`);
      }

      // Parse JSON (streaming supports progress updates)
      const json = await parseJsonSmart(response, setProgress);

      /**
       * Normalize:
       * Purpose:
       * - Accept multiple graph schemas from exports
       * - Convert into canonical:
       *   nodes: [{id,label,data}], edges: [{source,target,data}]
       */
      const tParse0 = performance.now();
      const { nodes, edges, debug } = normalizeGraph(json);
      const parseTime = performance.now() - tParse0;

      // Store normalized data for visualization layer
      setRaw({ nodes, edges });

      // Store diagnostics for UI (stats panel)
      setMeta({
        nodes: nodes.length,
        edges: edges.length,
        reason: debug?.reason ?? "",
        loadTime: performance.now() - t0,
        parseTime,
      });
    } catch (e) {
      // Ignore abort errors (expected during rapid URL changes/unmount)
      if (e?.name !== "AbortError") {
        setErr(String(e.message || e));
      }
    } finally {
      // Release loading state regardless of success/fail
      setIsLoading(false);
    }
  }, [url]);

  /**
   * Auto-load behavior:
   * Purpose:
   * - Every time URL changes, reload graph
   * - Cleanup abort prevents setting state after unmount
   */
  useEffect(() => {
    loadGraphData();
    return () => controllerRef.current?.abort();
  }, [loadGraphData]);

  /* ---------- Loading state ---------- */
  if (isLoading || !raw) {
    return (
      <div className="loading-container">
        {/* Spinner is purely visual feedback */}
        <div className="loading-spinner" />

        <h3>Loading Graph Data...</h3>

        {/* Progress bar: fill width based on streaming progress */}
        <div className="progress-bar">
          <div
            className="progress-fill"
            style={{ width: `${progress}%` }}
          />
        </div>

        {/* Human-readable percentage */}
        <p>{progress.toFixed(1)}% loaded</p>
      </div>
    );
  }

  /* ---------- Error state ---------- */
  if (err) {
    return (
      <div className="error-container">
        <h3>Failed to load graph</h3>

        {/* Preformatted error text keeps stack/messages readable */}
        <pre>{err}</pre>

        {/* Retry uses same pipeline (no special-case logic) */}
        <button onClick={loadGraphData}>Retry</button>
      </div>
    );
  }

  /* ---------- Success ---------- */
  return (
    <div style={{ height: "100%", position: "relative", flex: 1 }}>
      {/* Small stats + validation panel */}
      <div className="stats-panel">
        {/* Purpose: quick confirmation for dataset + performance */}
        <h4>Graph Stats</h4>

        {/* Nodes count */}
        <div className="stats-item">
          <span>Nodes:</span>
          <span className="stats-value">{meta.nodes.toLocaleString()}</span>
        </div>

        {/* Edges count */}
        <div className="stats-item">
          <span>Edges:</span>
          <span className="stats-value">{meta.edges.toLocaleString()}</span>
        </div>

        {/* Total pipeline time (fetch + parse + normalize) */}
        <div className="stats-item">
          <span>Load Time:</span>
          <span className="stats-value">{meta.loadTime.toFixed(0)}ms</span>
        </div>

        {/* Normalization time only (separate so you can optimize it) */}
        <div className="stats-item">
          <span>Parse Time:</span>
          <span className="stats-value">{meta.parseTime.toFixed(0)}ms</span>
        </div>

        {/* Embedded validator for debugging */}
        <ValidateJson url={url} />
      </div>

      {/* Visualization layer */}
      <ExCosmoGraph
        key={url} // Purpose: forces full remount when dataset changes (avoids stale renderer state)
        nodes={raw.nodes}
        edges={raw.edges}
      />
    </div>
  );
}
