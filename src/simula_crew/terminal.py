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
    BLUE = "\033[34m"
    CORAL = "\033[38;2;255;56;92m"
    MUTED = "\033[38;2;106;106;106m"


def supports_color() -> bool:
    return os.getenv("NO_COLOR") is None and os.getenv("TERM") != "dumb"


def color(text: str, style: str) -> str:
    if not supports_color():
        return text
    return f"{style}{text}{Style.RESET}"


def banner(title: str, subtitle: str = "") -> str:
    parts = [color(f"● {title}", Style.BOLD + Style.CORAL)]
    if subtitle:
        parts.append(color(f"  {subtitle}", Style.MUTED))
    return "\n".join(parts)


def section(title: str) -> str:
    return color(f"\n● {title}", Style.BOLD + Style.CORAL)


def key_value(key: str, value: str) -> str:
    return f"{color(key + ':', Style.BOLD)} {value}"


def wrap(text: str, width: int = 88) -> str:
    return fill(text, width=width, replace_whitespace=False)


def panel(title: str, body: str, width: int = 88) -> str:
    width = max(48, width)
    lines = [color(f"● {title}", Style.BOLD + Style.CORAL)]
    for paragraph in body.splitlines() or [""]:
        wrapped = fill(paragraph, width=width - 4) if paragraph else ""
        for line in wrapped.splitlines() or [""]:
            lines.append("  " + line)
    return "\n".join(lines)


def card(title: str, rows: list[tuple[str, str]], width: int = 88) -> str:
    body = "\n".join(f"{key:<18} {value}" for key, value in rows)
    return panel(title, body, width=width)


def tree(items: list[tuple[str, str]]) -> str:
    lines: list[str] = []
    for index, (label, detail) in enumerate(items):
        branch = "`--" if index == len(items) - 1 else "|--"
        lines.append(f"{branch} {color(label, Style.BOLD)} {detail}")
    return "\n".join(lines)


def logo() -> str:
    art = """
███████╗██╗███╗   ███╗██╗   ██╗██╗      █████╗  ██████╗██████╗ ███████╗██╗    ██╗
██╔════╝██║████╗ ████║██║   ██║██║     ██╔══██╗██╔════╝██╔══██╗██╔════╝██║    ██║
███████╗██║██╔████╔██║██║   ██║██║     ███████║██║     ██████╔╝█████╗  ██║ █╗ ██║
╚════██║██║██║╚██╔╝██║██║   ██║██║     ██╔══██║██║     ██╔══██╗██╔══╝  ██║███╗██║
███████║██║██║ ╚═╝ ██║╚██████╔╝███████╗██║  ██║╚██████╗██║  ██║███████╗╚███╔███╔╝
╚══════╝╚═╝╚═╝     ╚═╝ ╚═════╝ ╚══════╝╚═╝  ╚═╝ ╚═════╝╚═╝  ╚═╝╚══════╝ ╚══╝╚══╝
"""
    return color(art.strip("\n"), Style.BOLD + Style.CORAL)
