import { useEffect } from "react";

/**
 * readStoredPosition
 *
 * Reads a persisted position without owning state.
 *
 * Purpose:
 *  - Used during resize events
 *  - Allows recalculation from the last known persisted value
 *
 * @param {string} storageKey
 * @param {{x:number,y:number}} fallback
 * @returns {{x:number,y:number}}
 */
const readStoredPosition = (storageKey, fallback) => {
  try {
    const stored = localStorage.getItem(storageKey);
    if (!stored) return fallback;

    const parsed = JSON.parse(stored);
    if (
      typeof parsed?.x === "number" &&
      typeof parsed?.y === "number"
    ) {
      return { x: parsed.x, y: parsed.y };
    }
  } catch (error) {
    console.warn("Failed to read stored position:", error);
  }
  return fallback;
};

/**
 * clampToViewport
 *
 * Ensures a panel remains fully visible within the window.
 *
 * Purpose:
 *  - Prevent panels from being pushed off-screen after resize
 *  - Enforce padding so panels never touch screen edges
 *
 * @param {{x:number,y:number}} pos
 * @param {{width:number,height:number,padding:number}} constraints
 * @returns {{x:number,y:number}}
 */
const clampToViewport = (pos, constraints) => {
  const { width = 0, height = 0, padding = 10 } = constraints;

  // Maximum allowed top-left position
  const maxX = window.innerWidth - width - padding;
  const maxY = window.innerHeight - height - padding;

  return {
    x: Math.max(padding, Math.min(pos.x, maxX)),
    y: Math.max(padding, Math.min(pos.y, maxY)),
  };
};

/**
 * useResponsivePosition
 *
 * Hook that reacts to window resizes and requests
 * the caller to update panel position accordingly.
 *
 * IMPORTANT:
 *  - This hook does NOT own position state
 *  - It only *suggests* a corrected position
 *
 * Why this separation matters:
 *  - Keeps storage, drag, and resize logic decoupled
 *  - Allows BasePanel to stay dumb and reusable
 *
 * Typical usage:
 *
 *   const [pos, setPos] = useStoredPosition(...)
 *   useResponsivePosition(setPos, key, defaultPos, {
 *     width: 320,
 *     height: 180,
 *     padding: 16
 *   })
 *
 * @param {(pos:{x:number,y:number}) => void} onPositionChange
 * @param {string} storageKey
 * @param {{x:number,y:number}} defaultPosition
 * @param {{width?:number,height?:number,padding?:number}} constraints
 */
export const useResponsivePosition = (
  onPositionChange,
  storageKey,
  defaultPosition,
  constraints = {}
) => {
  useEffect(() => {
    /**
     * handleResize
     *
     * Reads last known position, clamps it,
     * and notifies the owner to update state.
     */
    const handleResize = () => {
      const storedOrDefault = readStoredPosition(
        storageKey,
        defaultPosition
      );

      const adjusted = clampToViewport(
        storedOrDefault,
        constraints
      );

      onPositionChange(adjusted);
    };

    // Listen for viewport changes
    window.addEventListener("resize", handleResize);

    // Run once on mount so initial layout is safe
    handleResize();

    return () =>
      window.removeEventListener("resize", handleResize);
  }, [onPositionChange, storageKey, defaultPosition, constraints]);
};
