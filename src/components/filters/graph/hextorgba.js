/**
 * Convert a color into an `rgba(r,g,b,a)` string.
 *
 * Supports:
 *  - Named CSS-like rgba/rgb strings: "rgba(10,20,30,0.5)", "rgb(10,20,30)"
 *  - Hex strings:
 *      "#rgb" / "rgb"
 *      "#rgba" / "rgba"
 *      "#rrggbb"
 *      "#rrggbbaa"
 *
 * If `a` is provided it overrides any alpha encoded in the color.
 * On invalid input, falls back to a pleasant blue-ish color.
 *
 * @param {string} color - CSS or hex color string.
 * @param {number} [a=1] - Optional alpha 0..1 to override encoded alpha.
 * @returns {string} CSS rgba() string.
 */
export const hexToRgba = (color, a = 1) => {
    const fallback = `rgba(91,143,199,${a})`;
    if (!color) return fallback;

    const c = String(color).trim();
    if (c.startsWith("rgb")) {
        if (c.includes("rgba")) return c;
        return c.replace("rgb(", "rgba(").replace(")", `,${a})`);
    }

    let h = c.startsWith("#") ? c.slice(1) : c;
    if (h.length === 3) h = h.split("").map((x) => x + x).join("");
    if (!(h.length === 6 || h.length === 8) || !/^[0-9a-f]+$/i.test(h)) return fallback;

    const r = parseInt(h.slice(0, 2), 16);
    const g = parseInt(h.slice(2, 4), 16);
    const b = parseInt(h.slice(4, 6), 16);
    return `rgba(${r},${g},${b},${a})`;
};
