import { useState } from "react";
import CitationsReGraph from "./ReGraph";
import "./style_app.css";

/**
 * Default dataset loaded on startup
 * Keeps the app immediately usable.
 */
const DEFAULT_JSON = "/exports/wiki_vote_regraph.json";

/**
 * Preset demo datasets
 * Used to quickly switch between known graphs.
 */
const PRESETS = [
  "/exports/wiki_vote_regraph.json",
  "/exports/ego_facebook_regraph.json",
  "/exports/bitcoin_otc_regraph.json",
];

/**
 * Inline UI styles for small controls.
 * Avoids creating CSS classes for trivial layout.
 */
const UI = {
  select: {
    padding: "0.5rem 0.75rem",
    border: "1px solid #3973d9ff",
    borderRadius: "6px",
  },
  input: {
    padding: "0.5rem 1rem",
    border: "1px solid #3973d9ff",
    borderRadius: "6px",
    flex: 1,
    minWidth: 240,
    maxWidth: 520,
  },
};

export default function App() {
  /**
   * jsonUrl
   * Single source of truth for the active dataset.
   * Changing this triggers a full reload downstream.
   */
  const [jsonUrl, setJsonUrl] = useState(DEFAULT_JSON);

  /**
   * Feature flag placeholder.
   * Allows switching renderers in the future.
   */
  const [useCosmograph] = useState(true);

  return (
    <div className="app-container">
      {/* =========================
          Header / Controls
         ========================= */}
      <header className="header">
        <div
          style={{
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
            gap: "1rem",
            flexWrap: "wrap",
          }}
        >
          <h1 style={{ margin: 0 }}>
            Graph Visualization Dashboard
          </h1>

          {/* Dataset selection controls */}
          <div
            style={{
              display: "flex",
              gap: "1rem",
              alignItems: "center",
              flexWrap: "wrap",
              justifyContent: "flex-end",
            }}
          >
            {/* Preset selector */}
            <select
              value={PRESETS.includes(jsonUrl) ? jsonUrl : ""}
              onChange={(e) =>
                e.target.value && setJsonUrl(e.target.value)
              }
              style={UI.select}
            >
              <option value="">— Choose preset —</option>
              {PRESETS.map((p) => (
                <option key={p} value={p}>
                  {p}
                </option>
              ))}
            </select>

            {/* Manual URL input */}
            <input
              type="text"
              value={jsonUrl}
              onChange={(e) => setJsonUrl(e.target.value)}
              placeholder="Enter JSON URL..."
              style={UI.input}
            />
          </div>
        </div>
      </header>

      {/* =========================
          Main content area
         ========================= */}
      <main className="content">
        {/* All graph logic lives below this point */}
        <CitationsReGraph
          url={jsonUrl}
          useCosmograph={useCosmograph}
        />
      </main>
    </div>
  );
}
