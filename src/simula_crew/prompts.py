from __future__ import annotations

from string import Formatter
from typing import Any

from simula_crew.schema import AgentPersona, ExperimentConfig, RoundSpec, Statement


class PromptRenderError(ValueError):
    """Raised when a prompt template references an unknown variable."""


def render_template(template: str, values: dict[str, Any]) -> str:
    try:
        return template.format_map(values)
    except KeyError as exc:
        missing = exc.args[0]
        available = ", ".join(sorted(values))
        raise PromptRenderError(
            f"Prompt references unknown variable '{missing}'. "
            f"Available variables: {available}"
        ) from exc


def referenced_fields(template: str) -> list[str]:
    fields: list[str] = []
    for _, field_name, _, _ in Formatter().parse(template):
        if field_name:
            fields.append(field_name.split(".", 1)[0])
    return fields


def format_rules(title: str, rules: list[str]) -> str:
    if not rules:
        return ""
    rendered = "\n".join(f"- {rule}" for rule in rules)
    return f"{title}:\n{rendered}"


def format_mapping(mapping: dict[str, Any]) -> str:
    if not mapping:
        return "None supplied."
    return "\n".join(f"- {key}: {value}" for key, value in sorted(mapping.items()))


def format_list(items: list[str]) -> str:
    if not items:
        return "None supplied."
    return "\n".join(f"- {item}" for item in items)


def format_agents(agents: list[AgentPersona]) -> str:
    return "\n".join(
        f"- {agent.name} ({agent.id}): {agent.base_prompt}"
        for agent in agents
    )


def format_transcript(
    statements: list[Statement],
    *,
    visibility: str,
) -> str:
    if visibility == "hidden":
        return "Prior transcript hidden for this round."
    if not statements:
        return "No prior transcript."

    aliases: dict[str, str] = {}
    blocks: list[str] = []
    for statement in statements:
        if visibility == "anonymous":
            speaker = aliases.setdefault(
                statement.agent_id,
                f"Agent {len(aliases) + 1}",
            )
        else:
            speaker = f"{statement.agent_name} ({statement.agent_id})"
        blocks.append(
            f"[{statement.round_id} turn {statement.turn_number} "
            f"{statement.event_type}] {speaker}:\n{statement.content}"
        )
    return "\n\n".join(blocks)


def agent_values(agent: AgentPersona) -> dict[str, Any]:
    values: dict[str, Any] = {
        "agent_id": agent.id,
        "agent_name": agent.name,
        "agent_base_prompt": agent.base_prompt,
        "agent_personality": format_mapping(agent.personality),
        "agent_backstory": agent.backstory or "None supplied.",
        "agent_speaking_style": agent.speaking_style or "None supplied.",
        "agent_knowledge": format_list(agent.knowledge),
        "agent_skills": format_list(agent.skills),
        "agent_interests": format_list(agent.interests),
        "agent_history": format_list(agent.history),
        "agent_goals": format_list(agent.goals),
        "agent_constraints": format_list(agent.constraints),
    }
    values.update({f"agent_{key}": value for key, value in agent.personality.items()})
    return values


def build_context(
    *,
    experiment: ExperimentConfig,
    round_spec: RoundSpec,
    transcript: list[Statement],
    agent: AgentPersona | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    task = experiment.task
    process = experiment.process
    values: dict[str, Any] = {
        "experiment_name": experiment.name,
        # Backwards-compatible aliases for templates that still use the old names.
        "config_name": experiment.name,
        "task_title": task.title,
        "task_prompt": task.prompt,
        "topic_title": task.title,
        "topic_prompt": task.prompt,
        "target_user": task.target_user,
        "success_criteria": task.success_criteria,
        "output_format_description": task.output_format.description,
        "output_format_type": task.output_format.type,
        "agents": format_agents(experiment.agents),
        "transcript": format_transcript(
            transcript,
            visibility=round_spec.transcript_visibility,
        ),
        "round_id": round_spec.id,
        "round_title": round_spec.title,
        "round_mode": round_spec.mode,
        "round_max_turns": round_spec.max_turns or "unspecified",
        "interaction_rules": format_rules(
            "Interaction rules",
            process.interaction_rules,
        ),
        "interruption_rules": format_rules(
            "Interruption rules",
            process.interruption_rules,
        ),
        "output_contract": _output_contract(task),
        "private_memory": "No private notes.",
    }
    values.update(task.variables)
    if agent is not None:
        values.update(agent_values(agent))
    if extra:
        values.update(extra)
    return values


def build_agent_system_prompt(
    experiment: ExperimentConfig,
    agent: AgentPersona,
) -> str:
    values = build_context(
        experiment=experiment,
        round_spec=experiment.process.rounds[0],
        transcript=[],
        agent=agent,
    )
    persona_contract = ""
    if experiment.process.character_prompt_template:
        persona_contract = render_template(
            experiment.process.character_prompt_template,
            values,
        )
    parts = [
        experiment.process.shared_instructions,
        persona_contract,
        format_rules("Interaction rules", experiment.process.interaction_rules),
        format_rules("Interruption rules", experiment.process.interruption_rules),
        agent.base_prompt,
    ]
    return "\n\n".join(part for part in parts if part.strip())


def _output_contract(task) -> str:
    fmt = task.output_format
    parts: list[str] = []
    if fmt.description:
        parts.append(fmt.description)
    if fmt.required_sections:
        parts.append(
            "Required sections: " + ", ".join(fmt.required_sections)
        )
    if fmt.type == "json":
        parts.append("Return valid JSON only. No prose, no Markdown fences.")
    elif fmt.type == "number":
        parts.append("Return a single numeric value.")
    return "\n\n".join(parts)
