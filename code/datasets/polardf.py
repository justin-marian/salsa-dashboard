import gzip
from typing import Generator
import polars as pl


def read_edge(
    path: str,
    comment_char: str | None = "#",
) -> pl.DataFrame:
    """
    Read an edge list into a Polars DataFrame with columns ``src`` and ``dst``.

    Pipeline
    --------
    1. Try :func:`read_df`:
        - Uses Polars' CSV reader with several common separators.
        - Returns a small, already-typed DataFrame if successful.

    2. If that fails:
        - Fall back to :func:`polar_fallback_csv`, which parses the file
            line-by-line (supports .gz) and yields batches of edges.
        - Concatenate batches and pass them through :func:`clean_df`.

    Parameters
    ----------
    path:
        Path to the edge list file (plain text or gzip-compressed).
    comment_char:
        Lines starting with this character are treated as comments and skipped.
        If None, no comment filtering is applied.

    Returns
    -------
    df:
        Polars DataFrame with columns:
            - ``src`` (Utf8)
            - ``dst`` (Utf8)
        Duplicates and empty rows are removed.
    """
    df = read_df(path, comment_char)
    if df is not None:
        return clean_df(df)

    data_batches = list(polar_fallback_csv(path, comment_char))
    if not data_batches:
        return pl.DataFrame({"src": [], "dst": []})

    if len(data_batches) == 1:
        return clean_df(data_batches[0])

    combined_dataframe = pl.concat(data_batches, how="vertical")
    return clean_df(combined_dataframe)


def read_df(path: str, comment_char: str | None = "#") -> pl.DataFrame | None:
    """
    Try to read an edge list using Polars' CSV reader with common separators.

    The function iterates over a small list of candidate separators and returns
    the first DataFrame that has at least two columns. The first two columns
    are then renamed to ``src`` and ``dst``.

    Parameters
    ----------
    path:
        Path to the input file.
    comment_char:
        Comment prefix passed to Polars; lines starting with this character
        are ignored during parsing.

    Returns
    -------
    df or None:
        A Polars DataFrame with columns ``src`` and ``dst`` if parsing is
        successful for any separator, otherwise None.
    """
    seps = [",", "\t", ";", " "]
    for sep in seps:
        df = pl.read_csv(
            path,
            has_header=False,
            separator=sep,
            comment_prefix=comment_char,
            ignore_errors=True,
            infer_schema_length=1000,
            low_memory=True,
            rechunk=True,
            quote_char=None,
        )
        # Need at least two columns to form (src, dst)
        if df.width < 2:
            continue

        c0, c1 = df.columns[:2]
        df = df.select(
            [
                pl.col(c0).cast(pl.Utf8).alias("src"),
                pl.col(c1).cast(pl.Utf8).alias("dst"),
            ]
        )
        return df

    return None


def clean_df(df: pl.DataFrame) -> pl.DataFrame:
    """
    Normalize and clean a raw edge DataFrame with ``src`` and ``dst`` columns.

    Operations applied (in order):
    1. Cast ``src`` and ``dst`` to Utf8.
    2. Strip leading/trailing whitespace.
    3. Filter out:
        - null values,
        - empty strings.
    4. Drop duplicate (src, dst) pairs.

    Parameters
    ----------
    df:
        Polars DataFrame with at least ``src`` and ``dst`` columns.

    Returns
    -------
    cleaned_df:
        A cleaned DataFrame with unique, non-empty string endpoints.
    """
    if df.is_empty():
        return df

    return (
        df.with_columns(
            [
                pl.col("src").cast(pl.Utf8).str.strip_chars(),
                pl.col("dst").cast(pl.Utf8).str.strip_chars(),
            ]
        )
        .filter(
            pl.col("src").is_not_null()
            & pl.col("dst").is_not_null()
            & (pl.col("src") != "")
            & (pl.col("dst") != "")
        )
        .unique(subset=["src", "dst"])
    )


def polar_fallback_csv(
    path: str,
    comment_char: str | None = "#",
    batch_size: int = 10000,  # Process in batches to reduce memory overhead
) -> Generator[pl.DataFrame, None, None]:
    """
    Fallback line-based parser that yields edge batches as Polars DataFrames.

    This is used when the main Polars CSV reader can't cleanly handle the file
    (e.g. irregular separators, malformed lines). It favors robustness and
    low memory usage:

    Behavior
    --------
    - Transparently supports gzip-compressed files (``.gz``).
    - Skips:
        * comment lines (starting with `comment_char`),
        * empty lines,
        * lines with fewer than two tokens.
    - Splits on whitespace and uses the first two tokens as ``src`` and ``dst``.
    - Accumulates edges into Python lists and yields a Polars DataFrame every ``batch_size`` edges.

    Parameters
    ----------
    path:
        Path to the edge list file (plain or gzipped).
    comment_char:
        Lines starting with this character are treated as comments.
        If None, no comment filtering is applied.
    batch_size:
        Maximum number of edges per yielded batch.

    Yields
    ------
    df_batch:
        Polars DataFrames with columns ``src`` and ``dst`` (Utf8), each
        representing one batch of parsed edges.
    """
    # Handle compressed files transparently
    open_function = gzip.open if path.endswith(".gz") else open
    source_batch, destination_batch = [], []

    with open_function(path, "rt", encoding="utf-8", errors="ignore") as file:
        for _, line in enumerate(file, 1):
            # Skip comment lines
            if comment_char and line.startswith(comment_char):
                continue

            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) < 2:
                continue

            source_batch.append(parts[0])
            destination_batch.append(parts[1])

            # Emit a batch when we reach the threshold
            if len(source_batch) >= batch_size:
                yield pl.DataFrame(
                    {"src": source_batch, "dst": destination_batch},
                    schema={"src": pl.Utf8, "dst": pl.Utf8},
                )
                source_batch, destination_batch = [], []

        # Emit any remaining edges as a final batch
        if source_batch:
            yield pl.DataFrame(
                {"src": source_batch, "dst": destination_batch},
                schema={"src": pl.Utf8, "dst": pl.Utf8},
            )
