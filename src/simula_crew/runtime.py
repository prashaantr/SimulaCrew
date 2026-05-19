from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from simula_crew.schema import ConfigError, ExperimentConfig


def apply_runtime_inputs(
    experiment: ExperimentConfig,
    *,
    prompt: str | None = None,
    prompt_file: str | Path | None = None,
    variables: dict[str, str] | None = None,
) -> ExperimentConfig:
    task_prompt = _resolve_prompt(prompt=prompt, prompt_file=prompt_file)
    task_variables = dict(experiment.task.variables)

    if task_prompt:
        task_variables["challenge_prompt"] = task_prompt
        task = replace(experiment.task, prompt=task_prompt, variables=task_variables)
    else:
        task = experiment.task

    if variables:
        task_variables.update(variables)
        task = replace(task, variables=task_variables)

    return replace(experiment, task=task)


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
