from __future__ import annotations

import json
from json import JSONDecodeError
from pathlib import Path
from typing import Any

import yaml

from simula_crew.schema import (
    AgentPersona,
    ConfigError,
    ExperimentConfig,
    ExperimentResult,
    ProcessConfig,
    TaskConfig,
)


def load_experiment(path: str | Path) -> ExperimentConfig:
    """Load an experiment from a YAML/JSON file that points to task, process, and agents.

    The experiment file has the shape:

      name: <slug>
      description: <text>
      task: <path or inline object>
      process: <path or inline object>
      agents: <path or inline list>
      metadata: <optional object>

    Each of task / process / agents may either be inline OR a path
    (relative to the experiment file) pointing to a JSON or YAML file.
    """
    experiment_path = Path(path)
    data = _load_structured(experiment_path)
    if not isinstance(data, dict):
        raise ConfigError(f"{experiment_path}: experiment file must be an object.")
    if _looks_like_legacy_config(data):
        raise ConfigError(
            f"{experiment_path}: legacy one-file configs are no longer supported. "
            "Use an experiment file that references separate task, process, and agents "
            "objects, or regenerate survey configs with `simulacrew ingest-survey --output-dir ...`."
        )

    try:
        name = _required_text(data, "name", "experiment")
        description = str(data.get("description", ""))
        task_data = _resolve_ref(data.get("task"), experiment_path, "task")
        process_data = _resolve_ref(data.get("process"), experiment_path, "process")
        agents_data = _resolve_agents_ref(data.get("agents"), experiment_path)
        metadata = data.get("metadata") or {}
        if not isinstance(metadata, dict):
            raise ConfigError("experiment.metadata must be an object.")

        return ExperimentConfig.from_parts(
            name=name,
            description=description,
            task=task_data,
            process=process_data,
            agents=agents_data,
            metadata=metadata,
        )
    except ConfigError as exc:
        raise ConfigError(f"{experiment_path}: {exc}") from exc


def list_experiments(experiments_dir: str | Path = "configs/experiments") -> list[Path]:
    root = Path(experiments_dir)
    if not root.exists():
        return []
    return sorted(
        path
        for path in root.iterdir()
        if path.is_file() and path.suffix.lower() in {".yaml", ".yml", ".json"}
    )


def save_experiment_result(
    result: ExperimentResult,
    output_dir: str | Path,
) -> tuple[Path, Path]:
    """Write the experiment result.

    Produces two files in output_dir:
      - <experiment>-<timestamp>.json: the full ExperimentResult.to_dict()
      - <experiment>-<timestamp>.txt: a compact human-readable conversation log
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    stem = f"{result.experiment_name}-{_safe_timestamp(result.started_at)}"
    json_path = output_path / f"{stem}.json"
    conversation_path = output_path / f"{stem}.txt"

    with json_path.open("w", encoding="utf-8") as file:
        json.dump(result.to_dict(), file, indent=2, default=_json_default)
        file.write("\n")

    conversation_path.write_text(format_conversation(result), encoding="utf-8")
    return json_path, conversation_path


def format_conversation(result: ExperimentResult) -> str:
    lines = [
        "SIMULACREW CONVERSATION",
        f"Task: {result.task_title}",
        f"Provider: {result.provider}",
        f"Model: {result.model}",
        f"Started: {result.started_at}",
        f"Prompt: {result.task_prompt}",
        f"Format check: valid={result.format_check.get('valid')} errors={result.format_check.get('errors')}",
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


def _resolve_ref(
    value: Any,
    base_path: Path,
    label: str,
) -> dict[str, Any]:
    if value is None:
        raise ConfigError(f"experiment.{label} is required.")
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        target = (base_path.parent / value).resolve()
        loaded = _load_structured(target)
        if not isinstance(loaded, dict):
            raise ConfigError(f"{target}: {label} file must be an object.")
        return loaded
    raise ConfigError(f"experiment.{label} must be a path or an inline object.")


def _resolve_agents_ref(value: Any, base_path: Path) -> list[dict[str, Any]]:
    if value is None:
        raise ConfigError("experiment.agents is required.")
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        target = (base_path.parent / value).resolve()
        loaded = _load_structured(target)
        if isinstance(loaded, list):
            return loaded
        if isinstance(loaded, dict) and "agents" in loaded:
            agents = loaded["agents"]
            if not isinstance(agents, list):
                raise ConfigError(f"{target}: agents must be a list.")
            return agents
        raise ConfigError(f"{target}: agents file must be a list or an object with 'agents'.")
    raise ConfigError("experiment.agents must be a path or an inline list.")


def _looks_like_legacy_config(data: dict[str, Any]) -> bool:
    return (
        "agents" in data
        and isinstance(data.get("agents"), list)
        and ("topic" in data or "harness" in data or "rounds" in data)
        and ("task" not in data or "process" not in data)
    )


def _load_structured(path: Path) -> Any:
    if not path.exists():
        raise ConfigError(f"File not found: {path}")
    suffix = path.suffix.lower()
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"Cannot read {path}: {exc}") from exc
    try:
        if suffix in {".yaml", ".yml"}:
            return yaml.safe_load(text)
        return json.loads(text)
    except (JSONDecodeError, yaml.YAMLError) as exc:
        raise ConfigError(f"Invalid {suffix or 'config'} in {path}: {exc}") from exc


def _required_text(data: dict[str, Any], key: str, path: str) -> str:
    if key not in data:
        raise ConfigError(f"{path}.{key} is required.")
    value = str(data[key]).strip()
    if not value:
        raise ConfigError(f"{path}.{key} cannot be empty.")
    return value


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


def _json_default(value: Any) -> Any:
    if hasattr(value, "to_dict"):
        return value.to_dict()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")
