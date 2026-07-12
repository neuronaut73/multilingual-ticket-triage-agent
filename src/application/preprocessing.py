import html
import re


def normalize_text(text: str | None, max_chars: int = 3000) -> str:
    """
    Lightweight text normalization for LLM-based ticket triage.

    Handles None/NaN safely, decodes HTML entities, strips HTML tags,
    and collapses whitespace. Does not lowercase or remove punctuation.
    """
    if text is None:
        return ""

    text = str(text)

    # Decode HTML entities, e.g. &amp; -> &
    text = html.unescape(text)

    # Remove simple HTML tags
    text = re.sub(r"<[^>]+>", " ", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # Trim very long tickets to keep inference stable
    if len(text) > max_chars:
        text = text[:max_chars].strip()

    return text


def build_raw_text(subject: str | None, body: str | None) -> str:
    """
    Combine subject and body into a single string for storage.

    Converts None to empty string safely.
    """
    subject = str(subject) if subject is not None else ""
    body    = str(body)    if body    is not None else ""
    return f"{subject}\n\n{body}"


def build_representation_text(
    subject: str | None,
    body: str | None,
    body_limit: int = 1500,
) -> str:
    """
    Produce the labeled text used for embeddings and retrieval.

    Explicit labels ('Subject:'/'Body:') make the structure clear to the
    embedding model and to anyone reading the stored value.
    The body is truncated to body_limit characters.
    Converts None to empty string safely.
    """
    subject = str(subject) if subject is not None else ""
    body    = str(body)    if body    is not None else ""
    return f"Subject: {subject}\n\nBody: {body[:body_limit]}"


def make_text_snippet(text: str | None, max_chars: int = 240) -> str:
    """
    Return the first max_chars characters of text.

    Used for display in logs, CSV outputs, and neighbor evidence.
    Converts None to empty string safely.
    """
    text = str(text) if text is not None else ""
    return text[:max_chars]
