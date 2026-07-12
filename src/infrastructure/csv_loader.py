import pandas as pd

# answer is the support agent's reply — it is never available for a new incoming
# ticket and must not be used as prediction input or stored anywhere.
_LEAKY_COLUMNS = {"answer"}


def load_csv(path: str) -> pd.DataFrame:
    """
    Load the Kaggle ticket CSV into a DataFrame.

    The answer column is dropped immediately to prevent any downstream leakage.
    Tag columns (tag_1 … tag_9) are kept as-is; callers that need a single
    tags list can call collect_tags() below.
    """
    df = pd.read_csv(path, dtype=str)
    # Replace the string 'nan' that sometimes appears after dtype=str coercion
    df = df.fillna("")
    # Drop leaky columns if present; errors="ignore" is safe if the CSV lacks them
    df = df.drop(columns=[c for c in _LEAKY_COLUMNS if c in df.columns])
    return df


def collect_tags(row: pd.Series) -> list[str]:
    """Return a deduplicated, non-empty list of tags from tag_1 … tag_9."""
    tags = []
    for i in range(1, 10):
        col = f"tag_{i}"
        if col in row.index:
            value = str(row[col]).strip()
            if value and value.lower() != "nan":
                tags.append(value)
    return tags
