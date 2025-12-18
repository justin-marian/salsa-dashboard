"""
Dataset configurations for experiments.

This module centralizes all dataset definitions used by the SALSA + community
detection pipeline. Each entry describes:

- how to load the graph,
- which community detection method to use,
- which SALSA variant / structural operator to apply,
- and where to store the final export.
"""

from typing import Any
from .register import (
    load_wikivote,
    load_ego_facebook,
    load_soc_sign_bitcoin_otc,
)
from .salsa_prep import S_identity, S_break_reciprocals


def get_dataset_configs() -> list[dict[str, Any]]:
    """
    Return all dataset configs (SNAP) used in the pipeline.

    Each config dictionary is consumed by a higher-level pipeline with the
    following stages:

        1. loader() -> (G, node_meta)
        2. Run SALSA + community detection on G
        3. Export results to `outfile` (regraph JSON)

    Keys (core ones)
    ----------------
    - name                    : short identifier
    - title                   : human-readable label
    - loader                  : callable that returns G (or (G, meta))
    - outfile                 : path to the JSON export
    - community_method        : community detection algorithm
    - community_kwargs        : parameters for that algorithm
    - graph_smoothing_function: structural operator (S_identity, S_break_reciprocals, ...)
    - damping_factor          : SALSA damping parameter
    - max_iterations          : SALSA iteration budget
    - tolerance               : SALSA convergence tolerance
    - extra                   : optional misc metadata (e.g. min_coll_size)
    """
    datasets: list[dict[str, Any]] = [
        # ------------------------------------------------------------------
        # ego-Facebook: undirected social graph (friendship network)
        # ------------------------------------------------------------------
        {
            "name": "ego-facebook",
            "title": "Web Graphs (ego-Facebook)",
            "loader": load_ego_facebook,
            "outfile": "../public/exports/ego_facebook_regraph.json",

            # Community detection: Leiden on undirected projection
            "community_method": "leiden",
            "community_kwargs": {
                "leiden_resolution": 1.0,
                "use_weights": True,
                "random_state": 42,
            },

            # Structural operator + SALSA hyperparameters
            "graph_smoothing_function": S_identity,
            "damping_factor": 0.15,
            "max_iterations": 500,
            "tolerance": 1e-8,

            # Only keep things here that don’t have a dedicated key
            "extra": {
                "min_coll_size": 15,
            },
        },

        # ------------------------------------------------------------------
        # soc-sign-bitcoin-otc: signed, directed trust network
        # ------------------------------------------------------------------
        {
            "name": "soc-sign-bitcoin-otc",
            "title": "Social Networks (soc-sign-bitcoin-otc)",
            "loader": load_soc_sign_bitcoin_otc,
            "outfile": "../public/exports/bitcoin_otc_regraph.json",

            "community_method": "leiden",
            "community_kwargs": {
                "leiden_resolution": 1.2,
                "use_weights": True,
                "random_state": 42,
            },

            "graph_smoothing_function": S_break_reciprocals,
            "damping_factor": 0.15,
            "max_iterations": 400,
            "tolerance": 1e-8,

            "extra": {
                "min_coll_size": 15,
            },
        },

        # ------------------------------------------------------------------
        # wiki-Vote: directed voting network (Wikipedia RfA)
        # ------------------------------------------------------------------
        {
            "name": "wiki-vote",
            "title": "Academic Networks (wiki-Vote)",
            "loader": load_wikivote,
            "outfile": "../public/exports/wiki_vote_regraph.json",

            "community_method": "leiden",
            "community_kwargs": {
                "leiden_resolution": 1.1,
                "use_weights": True,
                "random_state": 42,
            },

            "graph_smoothing_function": S_break_reciprocals,
            "damping_factor": 0.15,
            "max_iterations": 400,
            "tolerance": 1e-8,

            "extra": {
                "min_coll_size": 10,
            },
        },
    ]

    return datasets
