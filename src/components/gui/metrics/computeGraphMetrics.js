/**
 * percentile
 *
 * Small-array percentile helper.
 *
 * Purpose:
 *    - Used for *robust normalization* (q10/q90)
 *    - Prevents outliers from dominating visual encodings
 *
 * @param {number[]} arr - Input values
 * @param {number} p - Percentile in [0, 1]
 * @returns {number}
 */
function percentile(arr, p) {
    if (!arr.length) return 0;

    // Copy + sort to avoid mutating original array
    const sorted = [...arr].sort((a, b) => a - b);

    // Linear index (floor keeps it stable for small arrays)
    const idx = Math.floor((sorted.length - 1) * p);
    return sorted[idx];
}

/**
 * computeGraphMetrics
 *
 * Computes all graph metrics needed for visualization and interaction.
 *
 * Responsibilities:
 *    1. Normalize raw backend fields (pagerank, size, community, etc.)
 *    2. Compute degrees from edge list
 *    3. Derive composite importance score
 *    4. Build community metadata (counts + colors)
 *    5. Optionally filter graph by selected communities
 *
 * IMPORTANT:
 *    - This function is PURE.
 *    - No mutation of inputs.
 *    - No React hooks.
 *
 * @param {Array} nodes - Graph nodes (Cosmograph-compatible)
 * @param {Array} edges - Graph edges
 * @param {Set<number>} selectedCommunities - Active community filter
 * @returns {Object} Metrics + filtered graph + community metadata
 */
function computeGraphMetrics(nodes, edges, selectedCommunities) {
    /* ============================================================
     * DEGREE ACCUMULATORS
     * Purpose:
     *    - Build degree maps in O(E)
     * ============================================================ */
    const degOut = {};
    const degIn = {};
    const degTotal = {};

    /**
     * metrics
     * Keyed by node.id
     * Holds *all* computed metrics for downstream usage
     */
    const metrics = {};

    /* Community bookkeeping */
    const commSet = new Set();                        // unique community ids
    const commColorMap = new Map();             // first-seen color per community

    /* Global arrays for normalization */
    const allPR = [];
    const allDegTot = [];
    const allSizes = [];

    /* ------------------------------------------------------------
     * Normalize edges
     * Purpose:
     *    - Ensure edge payload lives on .data
     *    - Keep link objects flat and consistent
     * ------------------------------------------------------------ */
    const links = edges.map((e) => ({
        source: e.source,
        target: e.target,
        ...(e.data || {}),
    }));

    /* ============================================================
     * DEGREE COMPUTATION
     * ============================================================ */
    for (const l of links) {
        degOut[l.source] = (degOut[l.source] || 0) + 1;
        degIn[l.target] = (degIn[l.target] || 0) + 1;
    }

    for (const n of nodes) {
        const dIn = degIn[n.id] || 0;
        const dOut = degOut[n.id] || 0;
        degTotal[n.id] = dIn + dOut;
    }

    /* ============================================================
     * COLLECT RAW VALUES FOR NORMALIZATION
     * ============================================================ */
    for (const n of nodes) {
        const raw = n.data || n.properties || {};

        // Defensive extraction: backend schemas are not guaranteed
        const pr = raw.pagerank ?? raw.properties?.pagerank ?? 0;
        const community = raw.community ?? raw.properties?.community ?? 0;
        const rawSize = raw.size ?? raw.properties?.size ?? 0;
        const nodeColor = raw.color ?? raw.properties?.color ?? null;

        // Preserve first-seen color for each community
        if (nodeColor && !commColorMap.has(community)) {
            commColorMap.set(community, nodeColor);
        }

        commSet.add(community);

        allPR.push(pr);
        allDegTot.push(degTotal[n.id] || 0);
        allSizes.push(Number.isFinite(rawSize) ? rawSize : 0);
    }

    /* ============================================================
     * NORMALIZATION STATS (ROBUST)
     * ============================================================ */
    const p90PR = percentile(allPR, 0.9);
    const p90DegTot = percentile(allDegTot, 0.9);

    // Size uses q10–q90 to ignore extreme outliers
    const sizeQ10 = percentile(allSizes, 0.1);
    const sizeQ90 = percentile(allSizes, 0.9);
    const sizeMin = allSizes.length ? Math.min(...allSizes) : 0;
    const sizeMax = allSizes.length ? Math.max(...allSizes) : 0;

    let sizeRange = sizeQ90 - sizeQ10;
    if (sizeRange <= 0) sizeRange = sizeMax - sizeMin;

    /**
     * sizeStats
     * Purpose:
     *    - Passed to nodeSizeEncoder
     *    - Avoid recomputing quantiles per node
     */
    const sizeStats = {
        q10: sizeQ10,
        q90: sizeQ90,
        min: sizeMin,
        max: sizeMax,
        range: sizeRange > 0 ? sizeRange : 1,
    };

    /* ============================================================
     * PER-NODE METRICS + IMPORTANCE
     * ============================================================ */
    let maxImportance = 0;
    let maxDegreeTotal = 0;

    for (const n of nodes) {
        const raw = n.data || n.properties || {};

        const dIn = degIn[n.id] || 0;
        const dOut = degOut[n.id] || 0;
        const dTot = dIn + dOut;

        const pr = raw.pagerank ?? raw.properties?.pagerank ?? 0;
        const community = raw.community ?? raw.properties?.community ?? 0;
        const community_name =
            raw.community_name ?? raw.properties?.community_name ?? null;

        const rawSize = raw.size ?? raw.properties?.size ?? 0;

        /**
         * Normalized components
         * Purpose:
         *    - Keep importance bounded
         *    - Prevent hubs from dominating purely by degree
         */
        const prNorm = p90PR > 0 ? Math.min(pr / p90PR, 1.5) : 0;
        const degNormTot = p90DegTot > 0 ? Math.min(dTot / p90DegTot, 1.5) : 0;

        /**
         * importance
         * Weighted mix of structure (degree) and global influence (pagerank)
         */
        const importance = degNormTot * 0.65 + prNorm * 0.35;

        metrics[n.id] = {
            size: Number.isFinite(rawSize) ? rawSize : 0,
            degree: dTot,
            degreeIn: dIn,
            degreeOut: dOut,
            pagerank: pr,
            community,
            community_name,
            importance,
            prNorm,
            degNormTot,

            // Optional centrality metrics (if backend provides them)
            salsa_hub: raw.salsa_hub ?? raw.properties?.salsa_hub ?? 0,
            salsa_auth: raw.salsa_auth ?? raw.properties?.salsa_auth ?? 0,
            hits_hub: raw.hits_hub ?? raw.properties?.hits_hub ?? 0,
            hits_auth: raw.hits_auth ?? raw.properties?.hits_auth ?? 0,

            // Preserve original fields for debug panels
            deg_in: raw.deg_in ?? raw.properties?.deg_in ?? dIn,
            deg_out: raw.deg_out ?? raw.properties?.deg_out ?? dOut,
            ...raw,
        };

        if (importance > maxImportance) maxImportance = importance;
        if (dTot > maxDegreeTotal) maxDegreeTotal = dTot;
    }

    // Safety guards (avoid division by zero downstream)
    if (maxImportance <= 0) maxImportance = 1;
    if (maxDegreeTotal <= 0) maxDegreeTotal = 1;

    /* ============================================================
     * COMMUNITY LIST
     * ============================================================ */
    const commList = Array.from(commSet)
        .sort((a, b) => a - b)
        .map((c) => {
            const count = nodes.filter((n) => {
                const nc =
                    n.data?.community ?? n.data?.properties?.community ?? 0;
                return nc === c;
            }).length;

            const color = commColorMap.get(c) || "#5B8FC7";
            return { id: c, color, count };
        });

    const communityColors = commList.map((c) => c.color);

    /* ============================================================
     * COMMUNITY FILTERING
     * ============================================================ */
    const allCommunitiesSelected =
        selectedCommunities.size === 0 ||
        selectedCommunities.size === commList.length;

    if (allCommunitiesSelected) {
        return {
            filteredNodes: nodes,
            filteredEdges: links,
            nodeMetrics: metrics,
            communities: commList,
            communityColors,
            maxImportance,
            maxDegreeTotal,
            sizeStats,
        };
    }

    // Filter nodes
    const visibleNodes = nodes.filter((n) => {
        const comm =
            n.data?.community ?? n.data?.properties?.community ?? 0;
        return selectedCommunities.has(comm);
    });

    const visibleNodeIds = new Set(visibleNodes.map((n) => n.id));

    // Filter edges to visible nodes only
    const visibleEdges = edges
        .filter(
            (e) =>
                visibleNodeIds.has(e.source) &&
                visibleNodeIds.has(e.target)
        )
        .map((e) => ({
            source: e.source,
            target: e.target,
            ...(e.data || {}),
        }));

    return {
        filteredNodes: visibleNodes,
        filteredEdges: visibleEdges,
        nodeMetrics: metrics,
        communities: commList,
        communityColors,
        maxImportance,
        maxDegreeTotal,
        sizeStats,
    };
}

export { computeGraphMetrics };
