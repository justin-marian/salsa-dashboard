from typing import Any


def build_human_readable_community_text(
    community_info: dict[int, dict[str, Any]],
    community_names: dict[int, str]
) -> dict[int, str]:
    text_map: dict[int, str] = {}
    for cid, info in community_info.items():
        cname = community_names.get(cid, f"Community {cid}")
        size_c = info.get("size", 0)
        dens = info.get("internal_density", 0.0)
        neigh = info.get("strongest_external_neighbor", None)
        ext_edges = info.get("strongest_external_edges", 0)
        reps = info.get("representatives", [])

        # representative bullet preview
        if reps:
            rep_preview = "; ".join(reps[:3])
        else:
            rep_preview = "N/A"

        if neigh is None:
            relation_line = "This group is mostly internally connected."
        else:
            relation_line = (
                f"It connects most strongly to Community {neigh} "
                f"({ext_edges} cross edges)."
            )

        blurb = (
            f"{cname} with {size_c} nodes.\n"
            f"Internal cohesion ~{dens:.3f} (higher means they cite/follow each other a lot).\n"
            f"{relation_line}\n"
            f"Key representatives: {rep_preview}"
        )
        text_map[cid] = blurb
    return text_map
