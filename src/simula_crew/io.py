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
    markdown_path = output_path / f"{stem}.md"

    with json_path.open("w", encoding="utf-8") as file:
        json.dump(result.to_dict(), file, indent=2)
        file.write("\n")

    markdown_path.write_text(format_markdown(result), encoding="utf-8")
    return json_path, markdown_path


def format_markdown(result: CrewResult) -> str:
    lines = [
        f"# {result.topic_title}",
        "",
        result.description,
        "",
        f"- Config: `{result.config_name}`",
        f"- Provider: `{result.provider}`",
        f"- Model: `{result.model}`",
        f"- Started: `{result.started_at}`",
        "",
        "## Topic",
        "",
        result.topic_prompt,
        "",
    ]

    for round_result in result.rounds:
        lines.extend(
            [
                f"## {round_result.title}",
                "",
                f"- Mode: `{round_result.mode}`",
                f"- Transcript: `{round_result.transcript_visibility}`",
                f"- Max turns: `{round_result.max_turns or 'not applicable'}`",
                "",
            ]
        )
        for statement in round_result.statements:
            lines.extend(
                [
                    f"### {statement.agent_name}",
                    "",
                    f"- Turn: `{statement.turn_number}`",
                    f"- Event: `{statement.event_type}`",
                    "",
                    statement.content.strip(),
                    "",
                ]
            )

    return "\n".join(lines).rstrip() + "\n"


def _safe_timestamp(value: str) -> str:
    return (
        value.replace(":", "")
        .replace("+", "")
        .replace(".", "")
        .replace("T", "-")
        .replace("Z", "")
    )
