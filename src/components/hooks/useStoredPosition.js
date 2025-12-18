import { useState, useEffect } from "react";

/**
 * loadStoredPosition
 *
 * Safely reads a `{x, y}` position from localStorage.
 *
 * Purpose:
 *  - Restore UI panel position across reloads
 *  - Be resilient to corrupted or missing storage entries
 *
 * @param {string} storageKey
 * @param {{x:number,y:number}} fallback
 * @returns {{x:number,y:number}}
 */
const loadStoredPosition = (storageKey, fallback) => {
  try {
    const stored = localStorage.getItem(storageKey);
    if (!stored) return fallback;

    const parsed = JSON.parse(stored);

    // Validate shape defensively (avoid NaN / objects / strings)
    if (
      typeof parsed?.x === "number" &&
      typeof parsed?.y === "number"
    ) {
      return { x: parsed.x, y: parsed.y };
    }
  } catch (error) {
    // Never break UI because of localStorage
    console.warn("Failed to load stored position:", error);
  }

  return fallback;
};

/**
 * saveStoredPosition
 *
 * Persists a `{x, y}` position into localStorage.
 *
 * Purpose:
 *  - Keep persistence logic centralized
 *  - Allow silent failure in restricted environments (private mode, quota)
 *
 * @param {string} storageKey
 * @param {{x:number,y:number}} position
 */
const saveStoredPosition = (storageKey, position) => {
  try {
    localStorage.setItem(storageKey, JSON.stringify(position));
  } catch (error) {
    console.warn("Failed to save position:", error);
  }
};

/**
 * useStoredPosition
 *
 * React hook that treats `{x, y}` position as state
 * and automatically syncs it to localStorage.
 *
 * Responsibilities:
 *  - Initialize from localStorage (or fallback)
 *  - Expose a setter that ALSO persists
 *
 * Why this hook exists:
 *  - Keeps BasePanel and drag logic clean
 *  - Prevents copy-pasting localStorage logic across panels
 *
 * Typical usage:
 *
 *   const [position, setPosition] = useStoredPosition(
 *     "graph-info-panel-position",
 *     { x: 20, y: 40 }
 *   );
 *
 * @param {string} storageKey
 * @param {{x:number,y:number}} defaultPosition
 *
 * @returns {[{x:number,y:number}, (pos:{x:number,y:number}) => void]}
 */
export const useStoredPosition = (storageKey, defaultPosition) => {
  /**
   * position
   *
   * Local React state that drives panel placement.
   * Initialized lazily in effect to allow SSR safety.
   */
  const [position, setPosition] = useState(defaultPosition);

  /**
   * On mount (or when key changes):
   *  - load persisted value
   *  - update state once
   */
  useEffect(() => {
    const initial = loadStoredPosition(
      storageKey,
      defaultPosition
    );
    setPosition(initial);
  }, [storageKey, defaultPosition]);

  /**
   * updatePosition
   *
   * Wrapper setter:
   *  - updates React state
   *  - persists immediately
   *
   * Purpose:
   *  - Single source of truth for writes
   */
  const updatePosition = (newPosition) => {
    setPosition(newPosition);
    saveStoredPosition(storageKey, newPosition);
  };

  return [position, updatePosition];
};
