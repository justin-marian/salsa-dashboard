import React from "react";
import { CosmographSearch } from "@cosmograph/react";
import BasePanel from "../BasePanel";
import "./style_search.css";

/**
 * SearchPanel
 *
 * Floating, draggable search UI for selecting nodes in a Cosmograph graph.
 *
 * Key behaviors:
 * - Uses CosmographSearch to provide instant node search results.
 * - Calls onSelectResult(node) when the user selects a result.
 * - Exposes a forwarded ref to allow parent components to control the input
 *   (ex: ref.current.clearInput()).
 * - Persists its position via BasePanel storageKey.
 *
 * @param {Object} props
 * @param {(node:any) => void} props.onSelectResult Called when a result is chosen.
 * @param {React.Ref} ref Forwarded ref for CosmographSearch.
 * @returns {JSX.Element}
 */
const SearchPanel = React.forwardRef(({ onSelectResult }, ref) => {
  /**
   * dragHandle
   *
   * A minimal “grab bar” shown above the input.
   * Purpose:
   * - gives a clear draggable area without interfering with typing/selecting
   * - prevents accidental drags when clicking inside the search input
   */
  const dragHandle = <div className="search-drag-handle" />;

  return (
    <BasePanel
      storageKey="search-panel-position"
      defaultPosition={{ x: 20, y: 20 }}
      className="search-panel"
      dragHandle={dragHandle}
      /**
       * Purpose:
       * - search panel stays draggable and position-persistent
       * - className ties into the “floating glass card” theme in CSS
       */
    >
      <div
        className="search-input-container"
        /**
         * Purpose:
         * - layout wrapper for the search component
         * - can be used to position icons/overlays later if needed
         */
      >
        <CosmographSearch
          ref={ref}
          onSelectResult={onSelectResult}
          maxVisibleItems={10}
          placeholder="Search nodes..."
          /**
           * Purpose:
           * - ref: lets parent clear or focus input programmatically
           * - onSelectResult: propagate selection to graph controller
           * - maxVisibleItems: keep dropdown compact
           * - placeholder: hints user what this input does
           */
        />
      </div>
    </BasePanel>
  );
});

SearchPanel.displayName = "SearchPanel";

export { SearchPanel };
export default SearchPanel;
