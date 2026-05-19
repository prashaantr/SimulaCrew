from __future__ import annotations

import os
from textwrap import fill


class Style:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    CYAN = "\033[36m"
    GREEN = "\033[32m"
    MAGENTA = "\033[35m"
    YELLOW = "\033[33m"
    RED = "\033[31m"


def supports_color() -> bool:
    return os.getenv("NO_COLOR") is None and os.getenv("TERM") != "dumb"


def color(text: str, style: str) -> str:
    if not supports_color():
        return text
    return f"{style}{text}{Style.RESET}"


def banner(title: str, subtitle: str = "") -> str:
    width = max(64, min(96, len(title) + 8))
    top = "=" * width
    title_line = title.center(width)
    parts = [
        color(top, Style.CYAN),
        color(title_line, Style.BOLD + Style.CYAN),
    ]
    if subtitle:
        parts.append(color(subtitle.center(width), Style.DIM))
    parts.append(color(top, Style.CYAN))
    return "\n".join(parts)


def section(title: str) -> str:
    return color(f"\n== {title} ==", Style.BOLD + Style.MAGENTA)


def key_value(key: str, value: str) -> str:
    return f"{color(key + ':', Style.BOLD)} {value}"


def wrap(text: str, width: int = 88) -> str:
    return fill(text, width=width, replace_whitespace=False)
