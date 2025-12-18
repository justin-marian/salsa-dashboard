/**
 * ExCosmoGraph.jsx
 *
 * Purpose:
 * - This is the visualization "controller" for your graph view.
 * - It binds raw graph data (nodes/edges) - metrics/encoders - Cosmograph renderer.
 * - It also mounts all draggable UI panels (search, filters, zoom, info, node details).
 *
 * Key responsibilities:
 * - Maintain interaction state (selected node, labels, community filtering).
 * - Compute filtered graph + metrics via useGraphMetrics().
 * - Compute physics + rendering params based on graph size.
 * - Connect UI actions (search select, node click) to Cosmograph methods (focus/select).
 */

import { useRef, useState } from "react";
import { CosmographProvider, Cosmograph } from "@cosmograph/react";

import { useGraphMetrics } from "./gui/metrics/useGraphMetrics";
import { useSimulationParams } from "./hooks/useSimulationParams";
import { getRenderingParams } from "./hooks/useSimulationParams";

import SearchPanel from "./filters/search/SearchPanel";
import CommunityFilters from "./filters/search/CommunityFilters";
import ZoomControls from "./filters/zoom/ZoomControls";
import NodeDetailsPanel from "./filters/graph/NodeDetailsPanel";
import GraphInfo from "./filters/graph/GraphInfo";

export default function ExCosmoGraph({
  nodes = [],
  edges = [],
  communityText = {},
  communityStats = {},
}) {
  /* ============================================================
   * IMPERATIVE REFS (bridge to Cosmograph + SearchPanel)
   * ============================================================ */

  // Ref to the Cosmograph instance (imperative API: focusNode, selectNode, etc.)
  const cosmographRef = useRef(null);

  // Ref to the search component (used to clear input/highlights on graph click)
  const searchRef = useRef(null);

  /* ============================================================
   * UI STATE (what user is looking at / toggling)
   * ============================================================ */

  // Which nodes should show label text (kept small to avoid label clutter)
  const [showLabelsFor, setShowLabelsFor] = useState([]);

  // Toggle for expanding/collapsing the community filter list
  const [showFilters, setShowFilters] = useState(false);

  /* ============================================================
   * INTERACTION STATE (drives styling + details)
   * ============================================================ */

  // Current selected node (null = nothing selected)
  const [selectedNode, setSelectedNode] = useState(null);

  // Set of selected community ids (empty Set often means "all" depending on logic upstream)
  const [selectedCommunities, setSelectedCommunities] = useState(new Set());

  /* ============================================================
   * ZOOM + PHYSICS STATE (UI controls)
   * ============================================================ */

  // Zoom level shown in the zoom UI (Cosmograph is still the real source of truth)
  const [zoomLevel, setZoomLevel] = useState(1);

  // Whether force layout mode is enabled (stronger simulation / more movement)
  const [forceLayout, setForceLayout] = useState(false);

  /* ============================================================
   * STABILITY / LAYOUT
   * ============================================================ */

  // Random seed is captured once so layout remains stable across rerenders.
  // Without this, some engines can "jump" to a different initial layout.
  const randomSeedRef = useRef(Math.random());

  /* ============================================================
   * METRICS PIPELINE
   * - Produces filteredNodes/filteredEdges based on community selection + node selection
   * - Produces encoder functions (nodeColor, nodeSize, linkColor, linkWidth)
   * ============================================================ */
  const {
    filteredNodes,
    filteredEdges,
    nodeMetrics,
    communities,
    communityColors,
  } = useGraphMetrics(
    nodes,
    edges,
    selectedCommunities,
    selectedNode ? selectedNode.id : null // selection affects neighbor highlighting
  );

  /* ============================================================
   * PHYSICS PARAMETERS
   * - Automatically adapts repulsion/gravity/etc based on graph scale
   * - forceLayout acts as a "more aggressive" mode
   * ============================================================ */
  const simulation = useSimulationParams(
    filteredNodes.length,
    filteredEdges.length,
    selectedCommunities.size,
    communities.length,
    forceLayout
  );

  /* ============================================================
   * INTERACTION HANDLERS
   * ============================================================ */

  /**
   * handleNodeClick
   *
   * Called by Cosmograph when user clicks on the canvas.
   *
   * Behavior:
   * - Clears search highlight (so selection source is unambiguous)
   * - If a node is clicked: select it, show its label, open details, focus camera
   * - If background clicked: clear selection, clear labels, close details
   */
  const handleNodeClick = (node) => {
    // clear search highlight when clicking in the graph
    searchRef.current?.clearInput?.();

    if (node) {
      // Tell Cosmograph which node is selected (for internal state + highlights)
      cosmographRef.current?.selectNode?.(node);

      // Keep label visible for selected node only
      setShowLabelsFor([node]);

      // Save selection for panels/encoders
      setSelectedNode(node);

      // Camera focus for better UX (duration in ms)
      cosmographRef.current?.focusNode?.(node, 1200);
    } else {
      // Clear all internal selections in Cosmograph
      cosmographRef.current?.unselectNodes?.();

      // Hide labels and close details panel
      setShowLabelsFor([]);
      setSelectedNode(null);
    }
  };

  /**
   * handleSearchSelect
   *
   * Called when user selects a node from the search results.
   *
   * Behavior:
   * - Keep label visible for that node
   * - Update selection state
   * - Focus camera to the node
   */
  const handleSearchSelect = (node) => {
    setShowLabelsFor(node ? [node] : []);
    setSelectedNode(node || null);
    if (node) cosmographRef.current?.focusNode?.(node, 1200);
  };

  /* ============================================================
   * RENDERING PARAMETERS
   * - GPU/perf safety tuning based on graph size
   * - controls labels, arrows, pixel ratio, and spaceSize
   * ============================================================ */
  const {
    spaceSize,
    pixelRatio,
    showTopLabels,
    showTopLabelsLimit,
    renderLinks,
    showLinkArrows,
  } = getRenderingParams(filteredNodes.length, filteredEdges.length);

  return (
    // Root wrapper ensures Cosmograph fills the screen and overlays can be positioned
    <div style={{ width: "100%", height: "100%", position: "relative" }}>
      {/* Provider supplies graph to CosmographSearch (and other @cosmograph/react components) */}
      <CosmographProvider nodes={filteredNodes} links={filteredEdges}>
        {/* Draggable search panel (selecting result triggers focus/select) */}
        <SearchPanel ref={searchRef} onSelectResult={handleSearchSelect} />

        {/* Draggable community selection filter (drives filteredNodes/filteredEdges) */}
        <CommunityFilters
          communities={communities}
          selectedCommunities={selectedCommunities}
          setSelectedCommunities={setSelectedCommunities}
          showFilters={showFilters}
          setShowFilters={setShowFilters}
          communityColors={communityColors}
        />

        {/* Small overlay stack for info + zoom controls (kept above graph) */}
        <div
          style={{
            position: "absolute", // overlay on top of the canvas
            flexDirection: "column",
            zIndex: 1001,         // above the Cosmograph canvas
            maxWidth: "320px",    // keeps panels compact
          }}
        >
          {/* Graph-level stats panel (nodes, edges, density, etc.) */}
          <GraphInfo
            filteredNodes={filteredNodes}
            filteredEdges={filteredEdges}
            forceLayout={forceLayout}
            communityStats={communityStats}
            communities={communities}
          />

          {/* Zoom + force layout toggle panel */}
          <ZoomControls
            zoomLevel={zoomLevel}
            setZoomLevel={setZoomLevel}
            cosmographRef={cosmographRef}
            forceLayout={forceLayout}
            setForceLayout={setForceLayout}
          />
        </div>

        {/* Selected node details panel (only renders when selectedNode != null) */}
        <NodeDetailsPanel
          selectedNode={selectedNode}
          setSelectedNode={setSelectedNode}
          communityColors={communityColors}
          communityText={communityText}
        />

        {/* ============================================================
            Cosmograph Renderer
            - This is the actual graph canvas
            - Most values are computed upstream (metrics + simulation)
           ============================================================ */}
        <Cosmograph
          ref={cosmographRef}
          style={{ width: "100%", height: "100%" }}

          /* ---------- metric-based styling (encoders) ---------- */
          nodeColor={nodeMetrics.nodeColor}
          nodeSize={nodeMetrics.nodeSize}
          linkColor={nodeMetrics.linkColor}
          linkWidth={nodeMetrics.linkWidth}

          /* ---------- labels / appearance ---------- */
          nodeLabelColor="#E8F4F8"          // default label color
          hoveredNodeLabelColor="#FFD166"   // label highlight on hover
          focusedNodeRingColor="#FFD166"    // ring highlight when focused
          hoveredNodeRingColor="#FFD166"
          showTopLabels={showTopLabels}     // enable "top labels" mode for small graphs
          showLabelsFor={showLabelsFor}     // explicitly show labels for selected nodes
          showTopLabelsLimit={showTopLabelsLimit}
          nodeLabelAccessor={(n) => n.label || n.id} // label text source

          /* ---------- link visuals ---------- */
          linkArrows={showLinkArrows}
          linkArrowsSizeScale={0.8}

          /* ---------- physics (force simulation) ---------- */
          simulationRepulsion={simulation.repulsion}
          simulationGravity={simulation.gravity}
          simulationLinkDistance={simulation.linkDistance}
          simulationLinkSpring={simulation.linkSpring}
          simulationFriction={simulation.friction}
          simulationDecay={simulation.decay}
          simulationRepulsionFromMouse={0.8}  // interactive "push away" feel
          simulationCenterInertia={0.01}      // slows center drift for stability

          /* ---------- canvas ---------- */
          backgroundColor="#0A1628"
          spaceSize={spaceSize}             // global coordinate space size
          pixelRatio={pixelRatio}           // quality vs performance trade-off

          /* ---------- interaction ---------- */
          onClick={handleNodeClick}
          scaleNodesOnZoom={true}
          disableSimulation={false}
          randomSeed={randomSeedRef.current}
          antialias={false}                 // performance: reduce expensive smoothing
          renderLinks={renderLinks}

          /* ---------- initial viewport ---------- */
          fitViewOnInit={true}
          fitViewDelay={100}                // wait for layout to settle before fitting
          fitViewByNodesInRect={false}
          zoomScaleSensitivity={0.5}
          initialZoomLevel={1}
        />
      </CosmographProvider>
    </div>
  );
}
