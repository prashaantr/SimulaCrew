from __future__ import annotations

import json
from json import JSONDecodeError
from pathlib import Path

from simula_crew.schema import ConfigError, CrewConfig, CrewResult


def load_config(path: str | Path) -> CrewConfig:
    config_path = Path(path)
    try:
        with config_path.open("r", encoding="utf-8") as file:
            data = json.load(file)
    except FileNotFoundError as exc:
        raise ConfigError(f"Config file not found: {config_path}") from exc
    except JSONDecodeError as exc:
        raise ConfigError(f"Invalid JSON in {config_path}: {exc}") from exc

    try:
        return CrewConfig.from_dict(data)
    except ConfigError as exc:
        raise ConfigError(f"{config_path}: {exc}") from exc


def list_configs(config_dir: str | Path = "configs") -> list[Path]:
    root = Path(config_dir)
    if not root.exists():
        return []
    return sorted(path for path in root.glob("*.json") if path.is_file())


def save_result(result: CrewResult, output_dir: str | Path) -> tuple[Path, Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    stem = f"{result.config_name}-{_safe_timestamp(result.started_at)}"
    json_path = output_path / f"{stem}.json"
    conversation_path = output_path / f"{stem}.txt"

    with json_path.open("w", encoding="utf-8") as file:
        json.dump(result.to_dict(), file, indent=2)
        file.write("\n")

    conversation_path.write_text(format_conversation(result), encoding="utf-8")
    return json_path, conversation_path


def format_conversation(result: CrewResult) -> str:
    lines = [
        "SIMULACREW CONVERSATION",
        f"Topic: {result.topic_title}",
        f"Provider: {result.provider}",
        f"Model: {result.model}",
        f"Started: {result.started_at}",
        f"Prompt: {result.topic_prompt}",
    ]

    for round_result in result.rounds:
        lines.extend(
            [
                "",
                f"[{round_result.mode.upper()}] {round_result.title}",
            ]
        )
        for statement in round_result.statements:
            label = _statement_label(statement.event_type)
            lines.extend(
                [
                    f"{statement.agent_name} ({label}, turn {statement.turn_number}):",
                    _indent(statement.content.strip()),
                ]
            )

    return "\n".join(lines).rstrip() + "\n"


def format_markdown(result: CrewResult) -> str:
    """Backward-compatible alias for the compact conversation log."""
    return format_conversation(result)


def _statement_label(event_type: str) -> str:
    if event_type == "private":
        return "thinking"
    if event_type == "thought":
        return "stays quiet"
    if event_type == "interrupt":
        return "cuts in"
    if event_type == "synthesis":
        return "wraps up"
    return "says"


def _indent(text: str) -> str:
    return "\n".join(f"  {line}" if line else "" for line in text.splitlines())


def _safe_timestamp(value: str) -> str:
    return (
        value.replace(":", "")
        .replace("+", "")
        .replace(".", "")
        .replace("T", "-")
        .replace("Z", "")
    )
