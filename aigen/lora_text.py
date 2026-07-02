from __future__ import annotations


def caption_contains_token(text: str, token: str) -> bool:
    words = {part.strip(" ,.;:!?()[]{}\"'").lower() for part in text.split()}
    return token.lower() in words


def join_prompt_parts(*parts: str) -> str:
    return ", ".join(" ".join(part.split()).rstrip(" ,.;:") for part in parts)
