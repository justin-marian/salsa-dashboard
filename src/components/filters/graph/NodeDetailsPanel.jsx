import { useState } from "react";
import BasePanel from "../BasePanel";
import "./style_node.css";

/**
 * Header for the draggable NodeDetails panel.
 *
 * @param {Object} props
 * @param {() => void} props.onClose - Called when user closes the panel.
 * @returns {JSX.Element}
 */
function Header({ onClose }) {
  return (
    <div className="node-details-header">
      <h3 className="node-details-title">Node Detail</h3>
      <button
        className="node-details-close"
        onClick={onClose}
        aria-label="Close"
      >
        x
      </button>
    </div>
  );
}

/**
 * Generic key/value row renderer.
 * - If `value` is null/empty and no children are provided, it returns null.
 * - Supports monospaced rendering for IDs / codes.
 *
 * @param {Object} props
 * @param {string} props.label
 * @param {any} [props.value]
 * @param {boolean} [props.mono=false]
 * @param {any} [props.children]
 * @returns {JSX.Element|null}
 */
function Row({ label, value, mono = false, children }) {
  if ((value == null || value === "") && !children) return null;

  return (
    <div className="node-details-row">
      <div className="node-details-key">{label}</div>
      <div
        className={
          mono
            ? "node-details-value node-details-mono"
            : "node-details-value"
        }
      >
        {children || value}
      </div>
    </div>
  );
}

/**
 * Small “stat tile” for displaying a value with optional subtext.
 *
 * @param {Object} props
 * @param {string} props.label
 * @param {string|number} props.value
 * @param {string} [props.sub]
 * @returns {JSX.Element}
 */
function ScoreCard({ label, value, sub }) {
  return (
    <div className="node-details-score-box">
      <div className="node-details-score-key">{label}</div>
      <div className="node-details-score-value">{value}</div>
      {sub && <div className="node-stat-subtext">{sub}</div>}
    </div>
  );
}

/**
 * Centrality score section (HITS / SALSA / PageRank).
 * Renders nothing if no supported metrics exist.
 *
 * @param {Object} props
 * @param {Object<string, any>} props.metrics
 * @returns {JSX.Element|null}
 */
function Scores({ metrics }) {
  const hasScores =
    metrics.pagerank != null ||
    metrics.hits_hub != null ||
    metrics.hits_auth != null ||
    metrics.salsa_hub != null ||
    metrics.salsa_auth != null;

  if (!hasScores) return null;

  return (
    <div className="node-details-section">
      <div className="node-details-section-title">Centrality Scores</div>
      <div className="node-details-scores-grid">
        {metrics.hits_hub != null && (
          <ScoreCard
            label="HITS Hub"
            value={metrics.hits_hub.toFixed(8)}
            sub="Hub score"
          />
        )}
        {metrics.hits_auth != null && (
          <ScoreCard
            label="HITS Auth"
            value={metrics.hits_auth.toFixed(8)}
            sub="Authority score"
          />
        )}
        {metrics.salsa_hub != null && (
          <ScoreCard
            label="SALSA Hub"
            value={metrics.salsa_hub.toFixed(8)}
            sub="Hub score"
          />
        )}
        {metrics.salsa_auth != null && (
          <ScoreCard
            label="SALSA Auth"
            value={metrics.salsa_auth.toFixed(8)}
            sub="Authority score"
          />
        )}
        {metrics.pagerank != null && (
          <ScoreCard
            label="PageRank"
            value={`${(metrics.pagerank * 100).toFixed(8)}%`}
            sub="Importance"
          />
        )}
      </div>
    </div>
  );
}

/**
 * Connectivity summary (in/out/total degree).
 * Supports multiple degree naming conventions (deg_in/deg_out or degreeIn/degreeOut).
 *
 * @param {Object} props
 * @param {Object<string, any>} props.metrics
 * @returns {JSX.Element}
 */
function Connectivity({ metrics }) {
  const inDegree = metrics.deg_in ?? metrics.degreeIn ?? 0;
  const outDegree = metrics.deg_out ?? metrics.degreeOut ?? 0;
  const total = inDegree + outDegree;

  return (
    <div className="node-details-section">
      <div className="node-details-section-title">Connections</div>
      <div className="node-details-scores-grid">
        <ScoreCard label="Incoming" value={inDegree} sub="Links to node" />
        <ScoreCard label="Outgoing" value={outDegree} sub="Links from node" />
        <ScoreCard label="Total" value={total} sub="All links" />
      </div>
    </div>
  );
}

/**
 * Community section for the selected node.
 * Includes community id/name, optional size, and a copy-to-clipboard color UI.
 *
 * @param {Object} props
 * @param {Object<string, any>} props.metrics
 * @param {string[]} props.colors
 * @returns {JSX.Element|null}
 */
function Community({ metrics, colors }) {
  if (metrics.community == null) return null;

  const cid = metrics.community;
  const name = metrics.community_name || `Community ${cid}`;
  const size = metrics.community_size;
  const color = colors?.[cid % colors.length] || "#78A3FF";

  return (
    <div className="node-details-section node-details-community-section">
      <div className="node-details-section-title">
        Community
        <div
          className="community-color-preview"
          style={{ backgroundColor: color }}
          title={`Community color: ${color}`}
        />
      </div>
      <Row label="Name" value={name} />
      <Row label="ID" value={`#${cid}`} mono />
      {size != null && <Row label="Size" value={`${size} nodes`} />}
    </div>
  );
}

/**
 * NodeDetailsPanel.
 *
 * Draggable side panel that renders details for a selected graph node.
 * The panel is hidden when `selectedNode` is null.
 *
 * Supported fields include:
 * - Node id
 * - Link (metrics.href)
 * - Description toggle (metrics.desc)
 * - Community info (metrics.community + color palette)
 * - Centrality scores (pagerank / hits / salsa)
 * - Connectivity (in/out degree)
 *
 * @param {Object} props
 * @param {Object|null} props.selectedNode - Selected node object (from graph lib).
 * @param {string} props.selectedNode.id
 * @param {Object} [props.selectedNode.data]
 * @param {(node: null) => void} props.setSelectedNode - Setter used to close the panel.
 * @param {string[]} [props.communityColors=[]] - Color palette indexed by community id.
 * @returns {JSX.Element|null}
 */
export default function NodeDetailsPanel({
  selectedNode,
  setSelectedNode,
  communityColors = [],
}) {
  const [showDesc, setShowDesc] = useState(false);

  // NOTE: `copiedColor` currently never becomes true (setter is intentionally unused).
  // Keeping it as-is to avoid behavior change. If you want the toast, wire the setter.
  const [copiedColor, _] = useState(false);

  if (!selectedNode) return null;

  const metrics = selectedNode.data || {};
  const nodeId = selectedNode.id;

  return (
    <BasePanel
      storageKey="node-details-panel-position"
      defaultPosition={{ x: 20, y: 90 }}
      className="node-details-panel"
      dragHandle={<Header onClose={() => setSelectedNode(null)} />}
    >
      <div className="node-details-panel-content">
        {/* Basic Info */}
        <div className="node-details-section node-details-basic-section">
          <div className="node-details-row node-id-row">
            <div className="node-details-key">Node ID</div>
            <div className="node-details-value node-details-mono">{nodeId}</div>
          </div>

          {metrics.href && (
            <div className="node-details-row">
              <div className="node-details-key">Link</div>
              <div className="node-details-value">
                <a
                  href={metrics.href}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="node-details-link"
                >
                  {metrics.href}
                </a>
              </div>
            </div>
          )}
        </div>

        {/* Description */}
        {metrics.desc && (
          <div className="node-details-section">
            <div className="node-details-section-title">
              Description
              <button
                className="node-toggle-button"
                onClick={() => setShowDesc(!showDesc)}
              >
                {showDesc ? "Hide" : "Show"}
              </button>
            </div>
            {showDesc && <div className="node-description-text">{metrics.desc}</div>}
          </div>
        )}

        <Community metrics={metrics} colors={communityColors} />
        <Scores metrics={metrics} />
        <Connectivity metrics={metrics} />

        {copiedColor && (
          <div className="color-copied-notification">Color copied to clipboard!</div>
        )}
      </div>
    </BasePanel>
  );
}
