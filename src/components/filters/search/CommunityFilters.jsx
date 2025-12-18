import React from "react";
import BasePanel from "../BasePanel";
import "./style_community.css";

/**
 * CommunityFilters
 *
 * Draggable UI panel that controls which graph communities are visible/active.
 * It provides:
 *  - a collapsible list (expand/collapse)
 *  - per-community toggles
 *  - Select All / Deselect All actions
 *
 * Notes:
 *  - The panel position is persisted via BasePanel (storageKey).
 *  - Selection state is represented as a Set<number> for fast membership checks.
 *
 * @param {Object} props
 * @param {{id:number,color:string,count:number}[]} props.communities List of communities to display.
 * @param {Set<number>} props.selectedCommunities Set of selected community IDs.
 * @param {(updater: (prev:Set<number>) => Set<number>) => void} props.setSelectedCommunities Setter for selection.
 * @param {boolean} props.showFilters Whether the list content is expanded.
 * @param {(show:boolean) => void} props.setShowFilters Toggle expanded/collapsed state.
 * @returns {JSX.Element}
 */

/**
 * ToggleButton
 *
 * Header/drag handle UI:
 *  - shows current selection count (selected/total)
 *  - toggles expanding/collapsing the panel content
 *
 * Purpose: keeps the “always visible” part of the panel small, and makes
 * dragging predictable (entire header is draggable).
 *
 * @param {Object} props
 * @param {boolean} props.showFilters Whether content is visible.
 * @param {number} props.selectedCount Selected community count.
 * @param {number} props.totalCount Total community count.
 * @param {() => void} props.onClick Click handler to toggle showFilters.
 * @returns {JSX.Element}
 */
const ToggleButton = ({ showFilters, selectedCount, totalCount, onClick }) => (
  <button
    onClick={onClick}
    className="community-toggle-button"
    /**
     * Purpose:
     * - Full-width clickable header
     * - Styled as the “top bar” of the panel
     * - Also acts as a drag handle (cursor: grab in CSS)
     */
  >
    {/* Left side: summary label for quick scanning */}
    <span>Communities ({selectedCount}/{totalCount})</span>

    {/* Right side: simple +/- indicates expanded state */}
    <span>{showFilters ? "-" : "+"}</span>
  </button>
);

/**
 * ActionButtons
 *
 * Provides bulk selection actions:
 *  - Select All: marks all communities as selected
 *  - Deselect All: clears the selection set
 *
 * Purpose: avoids forcing the user to click each item when filtering.
 *
 * @param {Object} props
 * @param {() => void} props.onSelectAll Select all communities.
 * @param {() => void} props.onDeselectAll Clear all selections.
 * @returns {JSX.Element}
 */
const ActionButtons = ({ onSelectAll, onDeselectAll }) => (
  <div
    className="community-button-group"
    /**
     * Purpose:
     * - horizontal layout container for the two action buttons
     */
  >
    <button
      onClick={onSelectAll}
      className="community-action-button community-select-all-button"
      /**
       * Purpose:
       * - visually “positive” action (green tint in CSS)
       * - selects every known community id
       */
    >
      Select All
    </button>

    <button
      onClick={onDeselectAll}
      className="community-action-button community-deselect-all-button"
      /**
       * Purpose:
       * - visually “destructive/clearing” action (red tint in CSS)
       * - empties the selection set
       */
    >
      Deselect All
    </button>
  </div>
);

/**
 * CommunityItem
 *
 * One row in the list for a single community.
 * It includes:
 *  - checkbox state indicator
 *  - a color swatch (community color)
 *  - label “Community X”
 *  - member count
 *
 * Implementation detail:
 *  - The row is clickable (div onClick) to toggle selection.
 *  - The checkbox itself is “controlled” but its onChange is a no-op,
 *    because the row click is the single source of truth.
 *
 * @param {Object} props
 * @param {{id:number,color:string,count:number}} props.community Community data.
 * @param {boolean} props.isSelected Whether this community is selected.
 * @param {(id:number) => void} props.onToggle Called to toggle selection for this id.
 * @returns {JSX.Element}
 */
const CommunityItem = ({ community, isSelected, onToggle }) => (
  <div
    className={`community-item ${isSelected ? "community-item-selected" : ""}`}
    onClick={() => onToggle(community.id)}
    /**
     * Purpose:
     * - clickable row container
     * - gets highlighted style when selected
     */
  >
    <input
      type="checkbox"
      checked={isSelected}
      onChange={() => {}}
      className="community-checkbox"
      /**
       * Purpose:
       * - provides an immediate “selected/unselected” affordance
       * - actual toggle happens by clicking the row to keep UX consistent
       */
    />

    <div
      className="community-color-swatch"
      style={{ background: community.color }}
      /**
       * Purpose:
       * - shows the community’s assigned color so user can correlate it with the graph
       */
    />

    <span className="community-label">
      {/* Purpose: human-readable community label */}
      Community {community.id}
    </span>

    <span className="community-count">
      {/* Purpose: shows how many nodes belong to this community */}
      {community.count}
    </span>
  </div>
);

/**
 * CommunitiesList
 *
 * Pure renderer that maps communities -> CommunityItem rows.
 *
 * Purpose:
 * - isolates list rendering from parent state logic
 * - keeps CommunityFilters readable
 *
 * @param {Object} props
 * @param {{id:number,color:string,count:number}[]} props.communities
 * @param {Set<number>} props.selectedCommunities
 * @param {(id:number) => void} props.onCommunityToggle
 * @returns {JSX.Element}
 */
const CommunitiesList = ({ communities, selectedCommunities, onCommunityToggle }) => (
  <>
    {communities.map((c) => (
      <CommunityItem
        key={c.id}
        community={c}
        isSelected={selectedCommunities.has(c.id)}
        onToggle={onCommunityToggle}
      />
    ))}
  </>
);

/**
 * FilterContent
 *
 * Body content that appears only when expanded.
 * Contains:
 *  - bulk action buttons
 *  - list of communities
 *
 * Purpose:
 * - keep header always visible
 * - avoid clutter when collapsed
 *
 * @param {Object} props
 * @param {{id:number,color:string,count:number}[]} props.communities
 * @param {Set<number>} props.selectedCommunities
 * @param {() => void} props.onSelectAll
 * @param {() => void} props.onDeselectAll
 * @param {(id:number) => void} props.onCommunityToggle
 * @returns {JSX.Element}
 */
const FilterContent = ({
  communities,
  selectedCommunities,
  onSelectAll,
  onDeselectAll,
  onCommunityToggle,
}) => (
  <div
    className="community-content-area"
    /**
     * Purpose:
     * - scrollable content region (max-height in CSS)
     * - ensures list does not grow beyond viewport
     */
  >
    <ActionButtons onSelectAll={onSelectAll} onDeselectAll={onDeselectAll} />
    <CommunitiesList
      communities={communities}
      selectedCommunities={selectedCommunities}
      onCommunityToggle={onCommunityToggle}
    />
  </div>
);

const CommunityFilters = ({
  communities,
  selectedCommunities,
  setSelectedCommunities,
  showFilters,
  setShowFilters,
}) => {
  /**
   * toggleCommunity
   *
   * Adds/removes a community id from the selection Set.
   *
   * Purpose:
   * - fast selection toggling
   * - preserves immutability by returning a new Set
   *
   * @param {number} commId
   */
  const toggleCommunity = (commId) => {
    setSelectedCommunities((prev) => {
      const next = new Set(prev);
      next.has(commId) ? next.delete(commId) : next.add(commId);
      return next;
    });
  };

  /**
   * selectAllCommunities
   *
   * Selects every community in the current communities list.
   * Purpose: one-click “show all” or “include all in filter”.
   */
  const selectAllCommunities = () => {
    setSelectedCommunities(new Set(communities.map((c) => c.id)));
  };

  /**
   * deselectAllCommunities
   *
   * Clears the selection set.
   * Purpose: quick “hide all” or “start filtering from scratch”.
   */
  const deselectAllCommunities = () => {
    setSelectedCommunities(new Set());
  };

  /**
   * dragHandle
   *
   * BasePanel uses this element as the draggable region.
   * We use the ToggleButton so users can drag by grabbing the header.
   */
  const dragHandle = (
    <ToggleButton
      showFilters={showFilters}
      selectedCount={selectedCommunities.size}
      totalCount={communities.length}
      onClick={() => setShowFilters(!showFilters)}
    />
  );

  return (
    <BasePanel
      storageKey="community-filters-position"
      defaultPosition={{ x: 320, y: 20 }}
      dragHandle={dragHandle}
      className="community-filters-panel"
      /**
       * Purpose of BasePanel props:
       * - storageKey: persist position across reloads
       * - defaultPosition: where it appears first time
       * - dragHandle: what part is draggable
       * - className: visual styling hook
       */
    >
      {/* Purpose: render list only when expanded to reduce DOM + clutter */}
      {showFilters && (
        <FilterContent
          communities={communities}
          selectedCommunities={selectedCommunities}
          onSelectAll={selectAllCommunities}
          onDeselectAll={deselectAllCommunities}
          onCommunityToggle={toggleCommunity}
        />
      )}
    </BasePanel>
  );
};

export default React.memo(CommunityFilters);
