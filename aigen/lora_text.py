from __future__ import annotations


def caption_contains_token(text: str, token: str) -> bool:
    words = {part.strip(" ,.;:!?()[]{}\"'").lower() for part in text.split()}
    return token.lower() in words
