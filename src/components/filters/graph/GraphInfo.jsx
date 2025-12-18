import React, { useState } from "react";
import BasePanel from "../BasePanel";
import "./style_graph.css";

/**
 * @typedef {Object} GraphNode
 * @property {string} id
 * @property {Object} [data]
 * @property {number} [data.deg_in]
 * @property {number} [data.deg_out]
 * @property {number} [data.degreeIn]
 * @property {number} [data.degreeOut]
 */

/**
 * @typedef {Object} GraphEdge
 * @property {string|{id:string}} source - Source node (id or object containing id)
 * @property {string|{id:string}} target - Target node (id or object containing id)
 */

/**
 * @typedef {Object<string, any>} CommunityStats
 * Arbitrary key/value statistics for community detection / clustering output.
 */

/**
 * GraphInfo panel.
 *
 * Renders a draggable panel displaying graph statistics for the *currently filtered*
 * graph view (nodes/edges). Includes:
 * - Basic metrics (node count, edge count, average degree, density)
 * - Graph properties (isolated nodes, self-loops)
 * - Optional community stats (collapsible)
 *
 * @param {Object} props
 * @param {GraphNode[]} [props.filteredNodes=[]] - Nodes visible in the current view/filter.
 * @param {GraphEdge[]} [props.filteredEdges=[]] - Edges visible in the current view/filter.
 * @param {CommunityStats} [props.communityStats={}] - Extra stats produced by community algorithms.
 * @param {boolean} [props.forceLayout=false] - Whether a force-directed layout is actively running.
 * @returns {JSX.Element}
 */
function GraphInfo({
  filteredNodes = [],
  filteredEdges = [],
  communityStats = {},
  forceLayout = false,
}) {
  const [showDetails, setShowDetails] = useState(false);

  const nodeCount = filteredNodes.length;
  const edgeCount = filteredEdges.length;

  /**
   * Basic graph density proxy for a filtered view.
   * Note: this is not the classic undirected density formula; it uses edges/nodes
   * which stays stable for large graphs and works well as a quick “connectivity feel”.
   *
   * @type {string|number}
   */
  const density = nodeCount > 0 ? (edgeCount / nodeCount).toFixed(3) : 0;

  /**
   * Average degree assuming an undirected interpretation: 2E / N.
   * Useful even for directed graphs as a quick summary of overall link volume.
   *
   * @type {string|number}
   */
  const avgDegree = nodeCount > 0 ? ((edgeCount * 2) / nodeCount).toFixed(3) : 0;

  /**
   * Counts nodes with zero total degree (in+out == 0).
   *
   * @returns {number}
   */
  const isolatedNodes = filteredNodes.filter((node) => {
    const inDegree = node.data?.deg_in ?? node.data?.degreeIn ?? 0;
    const outDegree = node.data?.deg_out ?? node.data?.degreeOut ?? 0;
    return inDegree + outDegree === 0;
  }).length;

  /**
   * Counts self-loops (edges where source === target).
   * Supports both string ids and {id} node objects.
   *
   * @returns {number}
   */
  const selfLoops = filteredEdges.filter(
    (edge) => edge.source === edge.target || edge.source?.id === edge.target?.id
  ).length;

  /**
   * Drag handle content for BasePanel.
   * Shows title + layout status.
   *
   * @type {JSX.Element}
   */
  const dragHandle = (
    <div className="graph-info-header">
      <h3 className="graph-info-title">Graph Info</h3>
      <span
        className={`graph-status ${
          forceLayout ? "graph-status-force" : "graph-status-stable"
        }`}
      >
        {forceLayout ? "• Force" : "○ Stable"}
      </span>
    </div>
  );

  return (
    <BasePanel
      storageKey="graph-info-panel-position"
      defaultPosition={{ x: 20, y: 20 }}
      className="graph-info-panel"
      dragHandle={dragHandle}
    >
      <div className="graph-info-section">
        <div className="graph-info-section-title">Basic Statistics</div>
        <div className="graph-info-stats">
          <div className="graph-info-stat">
            <span className="graph-stat-label">Nodes</span>
            <span className="graph-stat-value">
              {nodeCount.toLocaleString()}
            </span>
          </div>
          <div className="graph-info-stat">
            <span className="graph-stat-label">Edges</span>
            <span className="graph-stat-value">
              {edgeCount.toLocaleString()}
            </span>
          </div>
          <div className="graph-info-stat">
            <span className="graph-stat-label">Avg Degree</span>
            <span className="graph-stat-value">{avgDegree}</span>
          </div>
          <div className="graph-info-stat">
            <span className="graph-stat-label">Density</span>
            <span className="graph-stat-value">{density}</span>
          </div>
        </div>
      </div>

      <div className="graph-info-section">
        <div className="graph-info-section-title">Graph Properties</div>
        <div className="graph-info-properties">
          <div className="graph-property">
            <span className="graph-property-label">Isolated Nodes:</span>
            <span className="graph-property-value">{isolatedNodes}</span>
          </div>
          <div className="graph-property">
            <span className="graph-property-label">Self-loops:</span>
            <span className="graph-property-value">{selfLoops}</span>
          </div>
        </div>
      </div>

      {/* Community Statistics (if available) */}
      {Object.keys(communityStats).length > 0 && (
        <div className="graph-info-section">
          <button
            className="graph-info-toggle"
            onClick={() => setShowDetails(!showDetails)}
          >
            {showDetails ? "Hide Stats" : "Show Stats"}
          </button>

          {showDetails && (
            <div className="graph-info-details">
              <table className="graph-stats-table">
                <tbody>
                  {Object.entries(communityStats).map(([key, value]) => (
                    <tr key={key}>
                      <td className="graph-stat-key">{key}:</td>
                      <td className="graph-stat-value">
                        {typeof value === "number"
                          ? value.toFixed(4)
                          : String(value)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      )}
    </BasePanel>
  );
}

export default React.memo(GraphInfo);
