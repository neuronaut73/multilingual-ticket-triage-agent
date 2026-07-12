"""
Sprint 6A — CSV output writer.

Writes a list of flat result dicts to a CSV file.
List and dict values are serialised as JSON strings so they round-trip safely.
text_snippet is normalised to a single line.
"""

import csv
import json
import os


def write_csv_rows(path: str, rows: list[dict]) -> None:
    """
    Write rows to a CSV file at path.

    Creates the parent directory if needed.
    Writes a header row followed by one data row per dict.
    List and dict values are serialised as compact JSON strings.
    text_snippet newlines are replaced with spaces for readability.

    Parameters
    ----------
    path:
        File path for the output CSV, e.g. "outputs/triage_results.csv".
    rows:
        List of flat dicts.  All dicts must share the same keys.
    """
    if not rows:
        return

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    fieldnames = list(rows[0].keys())

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(_serialize_row(row))


def _serialize_row(row: dict) -> dict:
    """
    Return a copy of row with all values safe for CSV cells.

    - list/dict values → compact JSON string
    - text_snippet → single line (newlines replaced with spaces)
    - everything else → unchanged
    """
    out = {}
    for key, value in row.items():
        if isinstance(value, (list, dict)):
            out[key] = json.dumps(value, ensure_ascii=False)
        elif key == "text_snippet" and isinstance(value, str):
            out[key] = value.replace("\n", " ").replace("\r", " ")
        else:
            out[key] = value
    return out
