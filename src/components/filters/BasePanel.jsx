import { useUniversalDrag } from './base/useUniversalDrag';
import UniversalDragHandle from './base/UniversalDragHandle';

/**
 * BasePanel
 *
 * Generic floating panel with:
 *   - absolute positioning
 *   - persisted position using `useUniversalDrag` (localStorage)
 *   - a drag handle area (any React node)
 *
 * This component does NOT impose content layout; it only provides:
 *   - the floating container with styling + zIndex
 *   - the drag handle at the top
 *   - the children as the content area
 *
 * Props:
 *   - storageKey       (string, required)
 *       Unique key used by `useUniversalDrag` to persist panel position.
 *
 *   - defaultPosition  ({ x: number, y: number }, required)
 *       Fallback position when nothing is stored yet.
 *
 *   - dragHandle       (ReactNode, required)
 *       Small visual element rendered inside the draggable header area.
 *       The actual drag behavior is implemented by `UniversalDragHandle`.
 *
 *   - style            (object, optional)
 *       Additional inline styles merged into the panel container.
 *
 *   - className        (string, optional)
 *       Optional CSS class name applied to the panel container.
 *
 *   - children         (ReactNode)
 *       Free-form content inside the panel (buttons, text, etc.).
 */
const BasePanel = ({
  storageKey,
  defaultPosition = { x: 5, y: 5 },
  children,
  dragHandle,
  style = {},
  className = '',
  ...props
}) => {
  const safeDefault = {};
  if (Number.isFinite(defaultPosition.x)) {
    safeDefault.x = defaultPosition.x;
  }
  if (Number.isFinite(defaultPosition.y)) {
    safeDefault.y = defaultPosition.y;
  }

  const { position, isDragging, handleDragStart } = useUniversalDrag(
    storageKey,
    safeDefault,
    { padding: 5 }
  );

  const x = Number.isFinite(position?.x) ? position.x : safeDefault.x;
  const y = Number.isFinite(position?.y) ? position.y : safeDefault.y;

  const panelStyle = {
    position: 'absolute',
    left: x,
    top: y,
    zIndex: 1000,
    cursor: 'default',
    userSelect: 'none',
    ...style,
  };

  return (
    <div style={panelStyle} className={className} {...props}>
      <UniversalDragHandle isDragging={isDragging} onDragStart={handleDragStart}>
        {dragHandle}
      </UniversalDragHandle>

      {children}
    </div>
  );
};

export default BasePanel;
