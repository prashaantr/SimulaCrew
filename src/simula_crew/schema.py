from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal, cast


RoundMode = Literal["private", "debate", "interruptions", "synthesis"]
TranscriptVisibility = Literal["named", "anonymous", "hidden"]
TurnStrategy = Literal["round_robin", "interruption_priority"]


class ConfigError(ValueError):
    """Raised when a SimulaCrew config is structurally invalid."""


@dataclass(frozen=True)
class TopicConfig:
    title: str
    prompt: str
    target_user: str = "the user"
    success_criteria: str = "useful, specific, feasible, and honest about uncertainty"
    variables: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TopicConfig":
        _ensure_object(data, "topic")
        return cls(
            title=_required_text(data, "title", "topic"),
            prompt=_required_text(data, "prompt", "topic"),
            target_user=_optional_text(data, "target_user", "the user", "topic"),
            success_criteria=_optional_text(
                data,
                "success_criteria",
                "useful, specific, feasible, and honest about uncertainty",
                "topic",
            ),
            variables=_optional_mapping(data, "variables", "topic"),
        )


@dataclass(frozen=True)
class HarnessConfig:
    shared_instructions: str = ""
    character_prompt_template: str = ""
    interaction_rules: list[str] = field(default_factory=list)
    interruption_rules: list[str] = field(default_factory=list)
    output_contract: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "HarnessConfig":
        if data is None:
            return cls()
        _ensure_object(data, "harness")
        return cls(
            shared_instructions=_optional_text(
                data,
                "shared_instructions",
                "",
                "harness",
            ),
            character_prompt_template=_optional_text(
                data,
                "character_prompt_template",
                "",
                "harness",
            ),
            interaction_rules=_optional_text_list(data, "interaction_rules", "harness"),
            interruption_rules=_optional_text_list(
                data,
                "interruption_rules",
                "harness",
            ),
            output_contract=_optional_text(data, "output_contract", "", "harness"),
        )


@dataclass(frozen=True)
class AgentPersona:
    id: str
    name: str
    base_prompt: str
    personality: dict[str, Any] = field(default_factory=dict)
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
            goals=_optional_text_list(data, "goals", f"agents[{agent_id}]"),
            constraints=_optional_text_list(
                data,
                "constraints",
                f"agents[{agent_id}]",
            ),
        )


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
        if mode not in {"private", "debate", "interruptions", "synthesis"}:
            raise ConfigError(
                f"{path}.mode must be one of: private, debate, interruptions, synthesis"
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


@dataclass(frozen=True)
class CrewConfig:
    name: str
    description: str
    topic: TopicConfig
    agents: list[AgentPersona]
    rounds: list[RoundSpec]
    harness: HarnessConfig = field(default_factory=HarnessConfig)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CrewConfig":
        _ensure_object(data, "config")
        agent_items = data.get("agents", [])
        if not isinstance(agent_items, list):
            raise ConfigError("agents must be a list.")
        agents = [AgentPersona.from_dict(item) for item in agent_items]
        if not agents:
            raise ConfigError("config requires at least one agent.")
        _require_unique_ids("agents", [agent.id for agent in agents])

        round_items = data.get("rounds", [])
        if not isinstance(round_items, list):
            raise ConfigError("rounds must be a list.")
        rounds = [RoundSpec.from_dict(item) for item in round_items]
        if not rounds:
            raise ConfigError("config requires at least one round.")
        _require_unique_ids("rounds", [round_spec.id for round_spec in rounds])

        agent_ids = {agent.id for agent in agents}
        for round_spec in rounds:
            if isinstance(round_spec.participants, list):
                unknown = sorted(set(round_spec.participants) - agent_ids)
                if unknown:
                    raise ConfigError(
                        f"Round '{round_spec.id}' references unknown agent(s): {unknown}"
                    )

        return cls(
            name=_required_identifier(data, "name", "config"),
            description=_optional_text(data, "description", "", "config"),
            topic=TopicConfig.from_dict(_required_mapping(data, "topic", "config")),
            agents=agents,
            rounds=rounds,
            harness=HarnessConfig.from_dict(data.get("harness")),
            metadata=_optional_mapping(data, "metadata", "config"),
        )


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
class CrewResult:
    config_name: str
    description: str
    topic_title: str
    topic_prompt: str
    started_at: str
    provider: str
    model: str
    rounds: list[RoundResult]
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        *,
        config: CrewConfig,
        provider: str,
        model: str,
        rounds: list[RoundResult],
        metadata: dict[str, Any] | None = None,
    ) -> "CrewResult":
        return cls(
            config_name=config.name,
            description=config.description,
            topic_title=config.topic.title,
            topic_prompt=config.topic.prompt,
            started_at=datetime.now(timezone.utc).isoformat(),
            provider=provider,
            model=model,
            rounds=rounds,
            metadata={
                "agent_count": len(config.agents),
                "round_count": len(config.rounds),
                "config_metadata": config.metadata,
                **(metadata or {}),
            },
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "config_name": self.config_name,
            "description": self.description,
            "topic_title": self.topic_title,
            "topic_prompt": self.topic_prompt,
            "started_at": self.started_at,
            "provider": self.provider,
            "model": self.model,
            "metadata": self.metadata,
            "rounds": [round_result.to_dict() for round_result in self.rounds],
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
