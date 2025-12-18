import { useState, useRef, useEffect } from "react";
import BasePanel from "../BasePanel";
import "./style_zoom.css";

/**
 * ZoomControls
 *
 * Draggable UI panel providing zoom interaction for a graph.
 *
 * @param {Object} props
 * @param {React.RefObject} props.cosmographRef
 *   Ref pointing to the Cosmograph instance.
 *   Required methods (best-effort, version-dependent):
 *     - zoomBy(factor)
 *     - setZoomLevel(level)
 *     - getZoomLevel()
 *
 * @returns {JSX.Element}
 */
const ZoomControls = ({ cosmographRef }) => {
  /**
   * zoomLevel
   *
   * Local UI state representing the current zoom.
   * Display-only; the source of truth remains the Cosmograph instance.
   */
  const [zoomLevel, setZoomLevel] = useState(1.0);

  /**
   * updateRef
   *
   * Holds a timeout ID for deferred zoom updates.
   * Purpose:
   *  - prevent stacking timeouts during rapid clicking
   *  - ensure UI updates after zoom animation completes
   */
  const updateRef = useRef(null);

  /**
   * readZoom
   *
   * Safely reads the current zoom level from Cosmograph.
   * Supports multiple API shapes for compatibility.
   *
   * @returns {number} zoom level (fallback: 1.0)
   */
  const readZoom = () => {
    const cg = cosmographRef.current;
    if (!cg) return 1.0;

    return (
      cg.getZoomLevel?.() ??
      cg.zoomLevel ??
      cg.getZoom?.() ??
      1.0
    );
  };

  /**
   * Zoom polling effect
   *
   * Purpose:
   *  - Keeps the displayed zoom level in sync even if zoom changes
   *    happen via mouse wheel, gestures, or other UI.
   *
   * Design choice:
   *  - Interval polling (200ms) is cheap and avoids tight coupling
   *    to internal Cosmograph events.
   */
  useEffect(() => {
    const interval = setInterval(() => {
      const currentZoom = readZoom();

      // Only update state if change is meaningful
      if (Math.abs(currentZoom - zoomLevel) > 0.01) {
        setZoomLevel(currentZoom);
      }
    }, 200);

    return () => clearInterval(interval);
  }, [zoomLevel]);

  /**
   * handleZoom
   *
   * Applies a multiplicative zoom factor.
   *
   * @param {number} factor
   *   >1  → zoom in
   *   <1  → zoom out
   */
  const handleZoom = (factor) => {
    const cg = cosmographRef.current;
    if (!cg) return;

    // Cancel any pending UI updates
    if (updateRef.current) {
      clearTimeout(updateRef.current);
    }

    // Preferred API: smooth incremental zoom
    if (typeof cg.zoomBy === "function") {
      cg.zoomBy(factor);
    }
    // Fallback API: absolute zoom setting
    else if (typeof cg.setZoomLevel === "function") {
      const current = readZoom();
      cg.setZoomLevel(current * factor);
    }

    // Update UI after zoom animation settles
    updateRef.current = setTimeout(() => {
      setZoomLevel(readZoom());
    }, 100);
  };

  /** Zoom helpers for buttons */
  const handleZoomIn = () => handleZoom(1.2);
  const handleZoomOut = () => handleZoom(0.8);

  /**
   * handleReset
   *
   * Resets zoom to default (1.0).
   * Uses Cosmograph-native reset if available.
   */
  const handleReset = () => {
    const cg = cosmographRef.current;
    if (!cg) return;

    if (typeof cg.resetZoom === "function") {
      cg.resetZoom();
    } else if (typeof cg.setZoomLevel === "function") {
      cg.setZoomLevel(1.0);
    }

    setZoomLevel(1.0);
  };

  return (
    <BasePanel
      storageKey="zoom-controls-position"
      defaultPosition={{ x: 20, y: 20 }}
      className="zoom-controls-panel"
      dragHandle={<div className="zoom-drag-handle" />}
      /**
       * BasePanel responsibilities:
       *  - draggable container
       *  - remembers position in localStorage
       *  - isolates drag logic from content
       */
    >
      <div className="zoom-controls-grid">
        {/* Zoom In */}
        <button
          className="zoom-button zoom-in-button"
          onClick={handleZoomIn}
          aria-label="Zoom In"
        >
          {/* SVG icon for crisp scaling */}
          <svg className="zoom-icon" viewBox="0 0 24 24">
            <path d="M19 13h-6v6h-2v-6H5v-2h6V5h2v6h6v2z" />
          </svg>
        </button>

        {/* Zoom Level Indicator */}
        <div className="zoom-level-indicator">
          {zoomLevel.toFixed(1)}x
        </div>

        {/* Zoom Out */}
        <button
          className="zoom-button zoom-out-button"
          onClick={handleZoomOut}
          aria-label="Zoom Out"
        >
          <svg className="zoom-icon" viewBox="0 0 24 24">
            <path d="M19 13H5v-2h14v2z" />
          </svg>
        </button>

        {/* Reset Zoom */}
        <button
          className="zoom-button zoom-reset-button"
          onClick={handleReset}
          aria-label="Reset Zoom"
        >
          <svg className="zoom-icon" viewBox="0 0 24 24">
            {/* Path: clockwise arrow with a break */}
            <path d="M12 5V1L7 6l5 5V7c3.31 0 6 2.69 6 6s-2.69 6-6 6-6-2.69-6-6H4c0 4.42 3.58 8 8 8s8-3.58 8-8-3.58-8-8-8z" />
          </svg>
        </button>
      </div>
    </BasePanel>
  );
};

export default ZoomControls;
