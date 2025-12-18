import { useState, useRef, useCallback } from "react";

/**
 * Clamp coordinates so an element stays fully visible in the viewport.
 *
 * @param {{x:number,y:number}} pos
 * @param {{width:number,height:number,padding:number}} dims
 * @returns {{x:number,y:number}}
 */
const clampToViewport = (pos, dims) => {
  const { width, height, padding } = dims;

  const maxX = window.innerWidth - width - padding;
  const maxY = window.innerHeight - height - padding;

  return {
    x: Math.max(padding, Math.min(pos.x, maxX)),
    y: Math.max(padding, Math.min(pos.y, maxY)),
  };
};

/**
 * Read an initial position from localStorage (if any).
 *
 * @param {string} storageKey
 * @param {{x:number,y:number}} defaultPosition
 */
const loadInitialPosition = (storageKey, defaultPosition) => {
  try {
    const saved = localStorage.getItem(storageKey);
    if (!saved) return defaultPosition;
    const parsed = JSON.parse(saved);
    if (
      typeof parsed?.x === "number" &&
      typeof parsed?.y === "number"
    ) {
      return { x: parsed.x, y: parsed.y };
    }
  } catch (e) {
    console.warn("Failed to load position from storage:", e);
  }
  return defaultPosition;
};

/**
 * Save position into localStorage (best-effort).
 *
 * @param {string} storageKey
 * @param {{x:number,y:number}} pos
 */
const persistPosition = (storageKey, pos) => {
  try {
    localStorage.setItem(storageKey, JSON.stringify(pos));
  } catch (e) {
    console.warn("Failed to save position to storage:", e);
  }
};

/**
 * useUniversalDrag
 *
 * Generic "drag a floating panel" hook with:
 *  - persisted position via localStorage,
 *  - viewport clamping (padding around edges),
 *  - minimal API for panels: { position, isDragging, handleDragStart, setPosition }.
 *
 * The pattern assumes:
 *  - The element you apply `handleDragStart` to is *inside* the panel.
 *  - The actual panel is `e.currentTarget.parentElement` (so we can read/set its position).
 *
 * Typical usage:
 *
 *   const { position, isDragging, handleDragStart, setPosition } =
 *     useUniversalDrag("node-panel-pos", { x: 40, y: 80 }, { padding: 12 });
 *
 *   return (
 *     <div
 *       style={{ left: position.x, top: position.y, position: "absolute" }}
 *     >
 *       <div onMouseDown={handleDragStart}>Drag me</div>
 *       ...
 *     </div>
 *   );
 *
 * @param {string} storageKey - localStorage key for persisting panel position.
 * @param {{x:number,y:number}} defaultPosition - fallback position.
 * @param {{padding?:number}} [options={}] - Optional settings (currently: padding).
 *
 * @returns {{
 *   position: {x:number,y:number},
 *   isDragging: boolean,
 *   handleDragStart: (e:MouseEvent) => void,
 *   setPosition: (pos:{x:number,y:number}) => void
 * }}
 */
export const useUniversalDrag = (
  storageKey,
  defaultPosition,
  options = {}
) => {
  const padding = options.padding ?? 10;

  const [position, setPositionState] = useState(() =>
    loadInitialPosition(storageKey, defaultPosition)
  );
  const [isDragging, setIsDragging] = useState(false);

  const dragOffset = useRef({ x: 0, y: 0 });
  const isDraggingRef = useRef(false);

  const savePosition = useCallback(
    (newPosition) => {
      setPositionState(newPosition);
      persistPosition(storageKey, newPosition);
    },
    [storageKey]
  );

  const handleDragStart = useCallback(
    (e) => {
      e.preventDefault();
      e.stopPropagation();

      setIsDragging(true);
      isDraggingRef.current = true;

      // The panel is the parent element of the drag handle
      const panelElement = e.currentTarget.parentElement;
      if (!panelElement) {
        console.warn("No panel element found for drag handle");
        return;
      }

      // Get current panel position from style or from state as fallback
      const currentX =
        parseInt(panelElement.style.left, 10) || position.x;
      const currentY =
        parseInt(panelElement.style.top, 10) || position.y;

      // Mouse-to-panel offset at drag start
      dragOffset.current = {
        x: e.clientX - currentX,
        y: e.clientY - currentY,
      };

      const handleMouseMove = (moveEvent) => {
        if (!isDraggingRef.current) return;

        const rawX = moveEvent.clientX - dragOffset.current.x;
        const rawY = moveEvent.clientY - dragOffset.current.y;

        // Read current panel bounds for clamping
        const rect = panelElement.getBoundingClientRect();
        const dims = {
          width: rect.width,
          height: rect.height,
          padding,
        };

        const constrained = clampToViewport(
          { x: rawX, y: rawY },
          dims
        );

        savePosition(constrained);
      };

      const handleMouseUp = () => {
        setIsDragging(false);
        isDraggingRef.current = false;

        document.removeEventListener("mousemove", handleMouseMove);
        document.removeEventListener("mouseup", handleMouseUp);
        document.removeEventListener("mouseleave", handleMouseUp);
      };

      document.addEventListener("mousemove", handleMouseMove);
      document.addEventListener("mouseup", handleMouseUp);
      document.addEventListener("mouseleave", handleMouseUp);
    },
    [position, padding, savePosition]
  );

  return {
    position,
    isDragging,
    handleDragStart,
    setPosition: savePosition,
  };
};
