/**
 * 
 * UniversalDragHandle component provides a draggable handle UI element.
 * It changes the cursor style based on dragging state and handles drag start events.
 * 
 * @param {boolean} isDragging - Indicates if the item is currently being dragged.
 * @param {function} onDragStart - Callback function to handle the drag start event.
 * @param {React.ReactNode} children - The child elements to be rendered inside the drag handle.
 * @returns {JSX.Element} The UniversalDragHandle component.
 */
const UniversalDragHandle = ({ isDragging, onDragStart, children }) => {
  return (
    <div
      style={{
        cursor: isDragging ? 'grabbing' : 'grab',
        userSelect: 'none',
        WebkitUserSelect: 'none',
        MozUserSelect: 'none',
        msUserSelect: 'none'
      }}
      onMouseDown={onDragStart}
    >
      {children}
    </div>
  );
};

export default UniversalDragHandle;
