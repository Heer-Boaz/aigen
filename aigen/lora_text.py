from __future__ import annotations


def caption_contains_token(text: str, token: str) -> bool:
    words = {part.strip(" ,.;:!?()[]{}\"'").lower() for part in text.split()}
    return token.lower() in words


def join_caption_parts(*values: str) -> str:
    parts = []
    seen = set()
    for value in values:
        for part in value.split(","):
            cleaned = " ".join(part.split())
            key = cleaned.lower()
            if cleaned and key not in seen:
                parts.append(cleaned)
                seen.add(key)
    return ", ".join(parts)
