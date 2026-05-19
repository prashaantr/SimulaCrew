from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from simula_crew.schema import ConfigError, CrewConfig


def apply_runtime_inputs(
    config: CrewConfig,
    *,
    prompt: str | None = None,
    prompt_file: str | Path | None = None,
    variables: dict[str, str] | None = None,
) -> CrewConfig:
    topic_prompt = _resolve_prompt(prompt=prompt, prompt_file=prompt_file)
    topic_variables = dict(config.topic.variables)

    if topic_prompt:
        topic_variables["challenge_prompt"] = topic_prompt
        topic = replace(config.topic, prompt=topic_prompt, variables=topic_variables)
    else:
        topic = config.topic

    if variables:
        topic_variables.update(variables)
        topic = replace(topic, variables=topic_variables)

    return replace(config, topic=topic)


def parse_variable_assignments(assignments: list[str] | None) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for assignment in assignments or []:
        if "=" not in assignment:
            raise ConfigError(f"Invalid --var '{assignment}'. Expected KEY=VALUE.")
        key, value = assignment.split("=", 1)
        key = key.strip()
        if not key:
            raise ConfigError(f"Invalid --var '{assignment}'. KEY cannot be empty.")
        parsed[key] = value
    return parsed


def _resolve_prompt(
    *,
    prompt: str | None,
    prompt_file: str | Path | None,
) -> str | None:
    if prompt and prompt_file:
        raise ConfigError("Use either --prompt or --prompt-file, not both.")
    if prompt_file:
        path = Path(prompt_file)
        try:
            prompt = path.read_text(encoding="utf-8")
        except FileNotFoundError as exc:
            raise ConfigError(f"Prompt file not found: {path}") from exc
    if prompt is None:
        return None
    prompt = prompt.strip()
    if not prompt:
        raise ConfigError("Prompt cannot be empty.")
    return prompt
