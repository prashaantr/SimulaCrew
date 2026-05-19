from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal, cast


RoundMode = Literal["private", "debate", "discussion", "interruptions", "synthesis"]
TranscriptVisibility = Literal["named", "anonymous", "hidden"]
TurnStrategy = Literal["round_robin", "interruption_priority"]
OutputFormatType = Literal["markdown", "json", "number", "text"]


class ConfigError(ValueError):
    """Raised when a SimulaCrew config is structurally invalid."""


@dataclass(frozen=True)
class OutputFormat:
    """Describes the expected final task result shape."""

    type: OutputFormatType = "text"
    description: str = ""
    required_sections: list[str] = field(default_factory=list)
    json_schema: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None, *, path: str = "task.output_format") -> "OutputFormat":
        if data is None:
            return cls()
        _ensure_object(data, path)
        type_value = str(data.get("type", "text")).lower()
        if type_value not in {"markdown", "json", "number", "text"}:
            raise ConfigError(f"{path}.type must be one of: markdown, json, number, text")
        json_schema = data.get("json_schema") or {}
        if json_schema and not isinstance(json_schema, dict):
            raise ConfigError(f"{path}.json_schema must be an object.")
        return cls(
            type=cast(OutputFormatType, type_value),
            description=_optional_text(data, "description", "", path),
            required_sections=_optional_text_list(data, "required_sections", path),
            json_schema=dict(json_schema),
        )

    def validate(self, content: str) -> tuple[Any, list[str]]:
        """Return (parsed_value, errors). parsed_value is the structured task result."""
        content = (content or "").strip()
        errors: list[str] = []
        if self.type == "markdown":
            for section in self.required_sections:
                if section.lower() not in content.lower():
                    errors.append(f"Missing required section: {section}")
            return content, errors
        if self.type == "json":
            try:
                parsed = json.loads(_strip_code_fence(content))
            except json.JSONDecodeError as exc:
                errors.append(f"Invalid JSON: {exc}")
                return content, errors
            return parsed, errors
        if self.type == "number":
            match = re.search(r"-?\d+(?:\.\d+)?", content)
            if not match:
                errors.append("No numeric value found in output.")
                return content, errors
            try:
                value = float(match.group(0))
            except ValueError as exc:
                errors.append(f"Could not parse number: {exc}")
                return content, errors
            return value, errors
        return content, errors

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "description": self.description,
            "required_sections": list(self.required_sections),
            "json_schema": dict(self.json_schema),
        }


@dataclass(frozen=True)
class TaskConfig:
    """What the team is being asked to do and the shape of its final result."""

    title: str
    prompt: str
    target_user: str = "the user"
    success_criteria: str = "useful, specific, feasible, and honest about uncertainty"
    output_format: OutputFormat = field(default_factory=OutputFormat)
    variables: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TaskConfig":
        _ensure_object(data, "task")
        return cls(
            title=_required_text(data, "title", "task"),
            prompt=_required_text(data, "prompt", "task"),
            target_user=_optional_text(data, "target_user", "the user", "task"),
            success_criteria=_optional_text(
                data,
                "success_criteria",
                "useful, specific, feasible, and honest about uncertainty",
                "task",
            ),
            output_format=OutputFormat.from_dict(data.get("output_format")),
            variables=_optional_mapping(data, "variables", "task"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "prompt": self.prompt,
            "target_user": self.target_user,
            "success_criteria": self.success_criteria,
            "output_format": self.output_format.to_dict(),
            "variables": dict(self.variables),
        }


@dataclass(frozen=True)
class AgentPersona:
    id: str
    name: str
    base_prompt: str
    personality: dict[str, Any] = field(default_factory=dict)
    backstory: str = ""
    speaking_style: str = ""
    knowledge: list[str] = field(default_factory=list)
    skills: list[str] = field(default_factory=list)
    interests: list[str] = field(default_factory=list)
    history: list[str] = field(default_factory=list)
    goals: list[str] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AgentPersona":
        _ensure_object(data, "agents[]")
        agent_id = _required_identifier(data, "id", "agents[]")
        return cls(
            id=agent_id,
            name=_optional_text(data, "name", agent_id, "agents[]"),
            base_prompt=_required_text(data, "base_prompt", f"agents[{agent_id}]"),
            personality=_optional_mapping(data, "personality", f"agents[{agent_id}]"),
            backstory=_optional_text(data, "backstory", "", f"agents[{agent_id}]"),
            speaking_style=_optional_text(data, "speaking_style", "", f"agents[{agent_id}]"),
            knowledge=_optional_text_list(data, "knowledge", f"agents[{agent_id}]"),
            skills=_optional_text_list(data, "skills", f"agents[{agent_id}]"),
            interests=_optional_text_list(data, "interests", f"agents[{agent_id}]"),
            history=_optional_text_list(data, "history", f"agents[{agent_id}]"),
            goals=_optional_text_list(data, "goals", f"agents[{agent_id}]"),
            constraints=_optional_text_list(
                data,
                "constraints",
                f"agents[{agent_id}]",
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "base_prompt": self.base_prompt,
            "personality": dict(self.personality),
            "backstory": self.backstory,
            "speaking_style": self.speaking_style,
            "knowledge": list(self.knowledge),
            "skills": list(self.skills),
            "interests": list(self.interests),
            "history": list(self.history),
            "goals": list(self.goals),
            "constraints": list(self.constraints),
        }


@dataclass(frozen=True)
class RoundSpec:
    id: str
    title: str
    mode: RoundMode
    prompt: str
    participants: str | list[str] = "all"
    transcript_visibility: TranscriptVisibility = "named"
    max_turns: int | None = None
    turn_strategy: TurnStrategy = "round_robin"
    speaker_id: str = "recorder"
    speaker_name: str = "Recorder"
    speaker_prompt: str = "You are a neutral recorder. Preserve disagreement."

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RoundSpec":
        _ensure_object(data, "rounds[]")
        round_id = _required_identifier(data, "id", "rounds[]")
        path = f"rounds[{round_id}]"

        mode = str(data.get("mode", "debate"))
        if mode not in {"private", "debate", "discussion", "interruptions", "synthesis"}:
            raise ConfigError(
                f"{path}.mode must be one of: private, debate, discussion, interruptions, synthesis"
            )

        transcript_visibility = str(data.get("transcript_visibility", "named"))
        if transcript_visibility not in {"named", "anonymous", "hidden"}:
            raise ConfigError(
                f"{path}.transcript_visibility must be named, anonymous, or hidden"
            )

        turn_strategy = str(data.get("turn_strategy", "round_robin"))
        if turn_strategy not in {"round_robin", "interruption_priority"}:
            raise ConfigError(
                f"{path}.turn_strategy must be round_robin or interruption_priority"
            )

        return cls(
            id=round_id,
            title=_optional_text(data, "title", round_id, path),
            mode=cast(RoundMode, mode),
            prompt=_required_text(data, "prompt", path),
            participants=_round_participants(data.get("participants", "all"), path),
            transcript_visibility=cast(TranscriptVisibility, transcript_visibility),
            max_turns=_optional_positive_int(data, "max_turns", path),
            turn_strategy=cast(TurnStrategy, turn_strategy),
            speaker_id=_optional_identifier(data, "speaker_id", "recorder", path),
            speaker_name=_optional_text(data, "speaker_name", "Recorder", path),
            speaker_prompt=_optional_text(
                data,
                "speaker_prompt",
                "You are a neutral recorder. Preserve disagreement.",
                path,
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "mode": self.mode,
            "prompt": self.prompt,
            "participants": (
                list(self.participants)
                if isinstance(self.participants, list)
                else self.participants
            ),
            "transcript_visibility": self.transcript_visibility,
            "max_turns": self.max_turns,
            "turn_strategy": self.turn_strategy,
            "speaker_id": self.speaker_id,
            "speaker_name": self.speaker_name,
            "speaker_prompt": self.speaker_prompt,
        }


@dataclass(frozen=True)
class ProcessConfig:
    """How members are expected to deliberate and reach a final decision."""

    shared_instructions: str = ""
    character_prompt_template: str = ""
    interaction_rules: list[str] = field(default_factory=list)
    interruption_rules: list[str] = field(default_factory=list)
    rounds: list[RoundSpec] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ProcessConfig":
        _ensure_object(data, "process")
        round_items = data.get("rounds", [])
        if not isinstance(round_items, list):
            raise ConfigError("process.rounds must be a list.")
        rounds = [RoundSpec.from_dict(item) for item in round_items]
        if not rounds:
            raise ConfigError("process.rounds must contain at least one round.")
        _require_unique_ids("process.rounds", [round_spec.id for round_spec in rounds])
        return cls(
            shared_instructions=_optional_text(data, "shared_instructions", "", "process"),
            character_prompt_template=_optional_text(
                data, "character_prompt_template", "", "process"
            ),
            interaction_rules=_optional_text_list(data, "interaction_rules", "process"),
            interruption_rules=_optional_text_list(data, "interruption_rules", "process"),
            rounds=rounds,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "shared_instructions": self.shared_instructions,
            "character_prompt_template": self.character_prompt_template,
            "interaction_rules": list(self.interaction_rules),
            "interruption_rules": list(self.interruption_rules),
            "rounds": [round_spec.to_dict() for round_spec in self.rounds],
        }


@dataclass(frozen=True)
class ExperimentConfig:
    """A complete experiment: agents + task + process."""

    name: str
    description: str
    task: TaskConfig
    process: ProcessConfig
    agents: list[AgentPersona]
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_parts(
        cls,
        *,
        name: str,
        description: str,
        task: dict[str, Any] | TaskConfig,
        process: dict[str, Any] | ProcessConfig,
        agents: list[dict[str, Any]] | list[AgentPersona],
        metadata: dict[str, Any] | None = None,
    ) -> "ExperimentConfig":
        task_config = task if isinstance(task, TaskConfig) else TaskConfig.from_dict(task)
        process_config = (
            process if isinstance(process, ProcessConfig) else ProcessConfig.from_dict(process)
        )
        agent_list: list[AgentPersona] = []
        for item in agents:
            agent_list.append(item if isinstance(item, AgentPersona) else AgentPersona.from_dict(item))
        if not agent_list:
            raise ConfigError("experiment requires at least one agent.")
        _require_unique_ids("agents", [agent.id for agent in agent_list])

        agent_ids = {agent.id for agent in agent_list}
        for round_spec in process_config.rounds:
            if isinstance(round_spec.participants, list):
                unknown = sorted(set(round_spec.participants) - agent_ids)
                if unknown:
                    raise ConfigError(
                        f"Round '{round_spec.id}' references unknown agent(s): {unknown}"
                    )

        return cls(
            name=name,
            description=description,
            task=task_config,
            process=process_config,
            agents=agent_list,
            metadata=metadata or {},
        )

    @property
    def rounds(self) -> list[RoundSpec]:
        return self.process.rounds


@dataclass(frozen=True)
class Statement:
    agent_id: str
    agent_name: str
    round_id: str
    turn_number: int | str
    event_type: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RoundResult:
    id: str
    title: str
    mode: RoundMode
    transcript_visibility: TranscriptVisibility
    max_turns: int | None
    statements: list[Statement]

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "mode": self.mode,
            "transcript_visibility": self.transcript_visibility,
            "max_turns": self.max_turns,
            "statements": [statement.to_dict() for statement in self.statements],
        }


@dataclass(frozen=True)
class ExperimentResult:
    """Final output of one experiment run.

    - task_result: the final deliverable, parsed into the shape declared by
      task.output_format. For markdown, this is a string. For json, a dict.
      For number, a float.
    - format_check: whether task_result satisfies the declared output format.
    - transcript: all public interactions, in chronological order.
    - thinking: per-agent private thoughts (independent positions + stay-quiet
      notes) — keyed by agent id.
    """

    experiment_name: str
    description: str
    task_title: str
    task_prompt: str
    output_format: OutputFormat
    started_at: str
    provider: str
    model: str
    task_result: Any
    raw_task_result: str
    format_check: dict[str, Any]
    transcript: list[Statement]
    thinking: dict[str, list[str]]
    rounds: list[RoundResult]
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        *,
        experiment: ExperimentConfig,
        provider: str,
        model: str,
        rounds: list[RoundResult],
        raw_task_result: str,
        thinking: dict[str, list[str]],
        metadata: dict[str, Any] | None = None,
    ) -> "ExperimentResult":
        parsed, errors = experiment.task.output_format.validate(raw_task_result)
        transcript = [
            statement
            for round_result in rounds
            for statement in round_result.statements
            if statement.event_type not in {"private", "thought"}
        ]
        return cls(
            experiment_name=experiment.name,
            description=experiment.description,
            task_title=experiment.task.title,
            task_prompt=experiment.task.prompt,
            output_format=experiment.task.output_format,
            started_at=datetime.now(timezone.utc).isoformat(),
            provider=provider,
            model=model,
            task_result=parsed,
            raw_task_result=raw_task_result,
            format_check={
                "valid": not errors,
                "errors": errors,
                "format_type": experiment.task.output_format.type,
            },
            transcript=transcript,
            thinking=thinking,
            rounds=rounds,
            metadata={
                "agent_count": len(experiment.agents),
                "round_count": len(experiment.process.rounds),
                "experiment_metadata": experiment.metadata,
                **(metadata or {}),
            },
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_name": self.experiment_name,
            "description": self.description,
            "task_title": self.task_title,
            "task_prompt": self.task_prompt,
            "output_format": self.output_format.to_dict(),
            "started_at": self.started_at,
            "provider": self.provider,
            "model": self.model,
            "task_result": self.task_result,
            "raw_task_result": self.raw_task_result,
            "format_check": self.format_check,
            "transcript": [statement.to_dict() for statement in self.transcript],
            "thinking": {agent_id: list(notes) for agent_id, notes in self.thinking.items()},
            "rounds": [round_result.to_dict() for round_result in self.rounds],
            "metadata": self.metadata,
        }


def _ensure_object(value: Any, path: str) -> None:
    if not isinstance(value, dict):
        raise ConfigError(f"{path} must be an object.")


def _required_text(data: dict[str, Any], key: str, path: str) -> str:
    if key not in data:
        raise ConfigError(f"{path}.{key} is required.")
    value = str(data[key])
    if not value.strip():
        raise ConfigError(f"{path}.{key} cannot be empty.")
    return value


def _optional_text(data: dict[str, Any], key: str, default: str, path: str) -> str:
    if key not in data or data[key] is None:
        return default
    return str(data[key])


def _required_identifier(data: dict[str, Any], key: str, path: str) -> str:
    value = _required_text(data, key, path).strip()
    _validate_identifier(value, f"{path}.{key}")
    return value


def _optional_identifier(
    data: dict[str, Any],
    key: str,
    default: str,
    path: str,
) -> str:
    value = _optional_text(data, key, default, path).strip()
    _validate_identifier(value, f"{path}.{key}")
    return value


def _validate_identifier(value: str, path: str) -> None:
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-")
    if not value or any(character not in allowed for character in value):
        raise ConfigError(
            f"{path} must contain only letters, numbers, underscores, or hyphens."
        )


def _required_mapping(data: dict[str, Any], key: str, path: str) -> dict[str, Any]:
    if key not in data:
        raise ConfigError(f"{path}.{key} is required.")
    value = data[key]
    if not isinstance(value, dict):
        raise ConfigError(f"{path}.{key} must be an object.")
    return dict(value)


def _optional_mapping(data: dict[str, Any], key: str, path: str) -> dict[str, Any]:
    value = data.get(key, {})
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ConfigError(f"{path}.{key} must be an object.")
    return dict(value)


def _optional_text_list(data: dict[str, Any], key: str, path: str) -> list[str]:
    value = data.get(key, [])
    if value is None:
        return []
    if not isinstance(value, list):
        raise ConfigError(f"{path}.{key} must be a list.")
    return [str(item) for item in value]


def _round_participants(value: Any, path: str) -> str | list[str]:
    if value == "all":
        return "all"
    if isinstance(value, str):
        raise ConfigError(f"{path}.participants must be 'all' or a list of ids.")
    if not isinstance(value, list):
        raise ConfigError(f"{path}.participants must be 'all' or a list of ids.")
    participant_ids: list[str] = []
    for item in value:
        participant_id = str(item).strip()
        _validate_identifier(participant_id, f"{path}.participants[]")
        participant_ids.append(participant_id)
    return participant_ids


def _optional_positive_int(data: dict[str, Any], key: str, path: str) -> int | None:
    if key not in data or data[key] is None:
        return None
    try:
        value = int(data[key])
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{path}.{key} must be a positive integer.") from exc
    if value <= 0:
        raise ConfigError(f"{path}.{key} must be a positive integer.")
    return value


def _require_unique_ids(label: str, ids: list[str]) -> None:
    seen: set[str] = set()
    duplicates: list[str] = []
    for item_id in ids:
        if item_id in seen:
            duplicates.append(item_id)
        seen.add(item_id)
    if duplicates:
        raise ConfigError(f"{label} contains duplicate id(s): {sorted(duplicates)}")


def _strip_code_fence(content: str) -> str:
    content = content.strip()
    if content.startswith("```"):
        content = content.strip("`")
        if content.lower().startswith("json"):
            content = content[4:].strip()
    return content
