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


def supports_color() -> bool:
    return os.getenv("NO_COLOR") is None and os.getenv("TERM") != "dumb"


def color(text: str, style: str) -> str:
    if not supports_color():
        return text
    return f"{style}{text}{Style.RESET}"


def banner(title: str, subtitle: str = "") -> str:
    width = max(64, min(96, len(title) + 8))
    top = "/" + "=" * (width - 2) + "\\"
    middle = "|" + title.center(width - 2) + "|"
    parts = [
        color(top, Style.CYAN),
        color(middle, Style.BOLD + Style.CYAN),
    ]
    if subtitle:
        parts.append(color("|" + subtitle.center(width - 2) + "|", Style.DIM))
    parts.append(color("\\" + "=" * (width - 2) + "/", Style.CYAN))
    return "\n".join(parts)


def section(title: str) -> str:
    return color(f"\n--[ {title} ]" + "-" * max(0, 58 - len(title)), Style.BOLD + Style.MAGENTA)


def key_value(key: str, value: str) -> str:
    return f"{color(key + ':', Style.BOLD)} {value}"


def wrap(text: str, width: int = 88) -> str:
    return fill(text, width=width, replace_whitespace=False)


def panel(title: str, body: str, width: int = 88) -> str:
    width = max(48, width)
    title_text = f" {title} "
    top = "+" + title_text + "-" * max(0, width - len(title_text) - 2) + "+"
    bottom = "+" + "-" * (width - 2) + "+"
    lines = [color(top, Style.BLUE)]
    for paragraph in body.splitlines() or [""]:
        wrapped = fill(paragraph, width=width - 4) if paragraph else ""
        for line in wrapped.splitlines() or [""]:
            lines.append(color("| ", Style.BLUE) + line.ljust(width - 4) + color(" |", Style.BLUE))
    lines.append(color(bottom, Style.BLUE))
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
    art = r"""
   _____ _                 _        ______
  / ___/(_)___ ___  __  __/ /___ _ / ____/_______ _      __
  \__ \/ / __ `__ \/ / / / / __ `// /   / ___/ _ \ | /| / /
 ___/ / / / / / / / /_/ / / /_/ // /___/ /  /  __/ |/ |/ /
/____/_/_/ /_/ /_/\__,_/_/\__,_/ \____/_/   \___/|__/|__/
"""
    return color(art.strip("\n"), Style.BOLD + Style.CYAN)
