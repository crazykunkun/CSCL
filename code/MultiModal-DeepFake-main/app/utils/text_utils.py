from __future__ import annotations


def highlight_tokens(text: str, positions: list[int] | tuple[int, ...]) -> str:
    words = text.split()
    pos = set(int(p) for p in positions or [] if str(p).isdigit())
    html_words = []
    for idx, word in enumerate(words):
        if idx in pos:
            html_words.append(f"<mark>{word}</mark>")
        else:
            html_words.append(word)
    return " ".join(html_words)
