from __future__ import annotations

import re
from typing import Any, Callable

from simula_crew.clients import ModelClient
from simula_crew.prompts import (
    build_agent_system_prompt,
    build_context,
    render_template,
)
from simula_crew.scoring import (
    DeterministicInterruptionClassifier,
    InterruptionClassifier,
    InterruptionDecision,
    choose_interrupting_agent,
    deterministic_interruption_score,
)
from simula_crew.schema import (
    AgentPersona,
    CrewConfig,
    CrewResult,
    RoundResult,
    RoundSpec,
    Statement,
)


def run_crew(
    *,
    config: CrewConfig,
    client: ModelClient,
    model: str,
    temperature: float = 0.2,
    max_agents: int | None = None,
    interruption_classifier: InterruptionClassifier | None = None,
    event_callback: Callable[[str, dict[str, Any]], None] | None = None,
    run_metadata: dict[str, Any] | None = None,
) -> CrewResult:
    agents = config.agents[:max_agents] if max_agents else list(config.agents)
    agent_by_id = {agent.id: agent for agent in agents}
    transcript: list[Statement] = []
    private_memory: dict[str, list[str]] = {}
    round_results: list[RoundResult] = []

    for round_spec in config.rounds:
        _emit(
            event_callback,
            "round_start",
            {
                "round_id": round_spec.id,
                "round_title": round_spec.title,
                "round_mode": round_spec.mode,
            },
        )
        participants = _select_agents(round_spec, agents, agent_by_id)
        if round_spec.mode == "private":
            statements = []
            for index, agent in enumerate(participants):
                statements.append(
                    _call_agent(
                        config=config,
                        round_spec=round_spec,
                        agent=agent,
                        transcript=list(transcript),
                        client=client,
                        model=model,
                        temperature=temperature,
                        turn_number=index + 1,
                        event_type="private",
                        extra={"turns_remaining": len(participants) - index - 1},
                        event_callback=event_callback,
                        private_memory=private_memory,
                    )
                )
        elif round_spec.mode in {"debate", "discussion", "interruptions"}:
            statements = _run_group_round(
                config=config,
                round_spec=round_spec,
                participants=participants,
                transcript=list(transcript),
                client=client,
                model=model,
                temperature=temperature,
                interruption_classifier=interruption_classifier,
                event_callback=event_callback,
                private_memory=private_memory,
            )
        else:
            statements = [
                _call_recorder(
                    config=config,
                    round_spec=round_spec,
                    transcript=list(transcript),
                    client=client,
                    model=model,
                    temperature=temperature,
                    event_callback=event_callback,
                    private_memory=private_memory,
                )
            ]

        transcript.extend(statements)
        round_results.append(
            RoundResult(
                id=round_spec.id,
                title=round_spec.title,
                mode=round_spec.mode,
                transcript_visibility=round_spec.transcript_visibility,
                max_turns=round_spec.max_turns,
                statements=statements,
            )
        )

    return CrewResult.create(
        config=config,
        provider=client.provider,
        model=model,
        rounds=round_results,
        metadata={
            "max_agents": max_agents,
            **(run_metadata or {}),
        },
    )


def _select_agents(
    round_spec: RoundSpec,
    agents: list[AgentPersona],
    agent_by_id: dict[str, AgentPersona],
) -> list[AgentPersona]:
    if round_spec.participants == "all":
        return agents
    return [agent_by_id[agent_id] for agent_id in round_spec.participants if agent_id in agent_by_id]


def _run_group_round(
    *,
    config: CrewConfig,
    round_spec: RoundSpec,
    participants: list[AgentPersona],
    transcript: list[Statement],
    client: ModelClient,
    model: str,
    temperature: float,
    interruption_classifier: InterruptionClassifier | None,
    event_callback: Callable[[str, dict[str, Any]], None] | None,
    private_memory: dict[str, list[str]],
) -> list[Statement]:
    if not participants:
        return []

    max_turns = round_spec.max_turns or len(participants)
    statements: list[Statement] = []
    for turn_index in range(max_turns):
        decision: InterruptionDecision | None = None
        speaker = _speaker_for_turn(
            participants=participants,
            round_spec=round_spec,
            turn_index=turn_index,
        )
        event_type = "debate"
        if (
            round_spec.mode == "discussion"
            and _should_stay_quiet(
                agent=speaker,
                transcript=[*transcript, *statements],
                turn_index=turn_index,
            )
        ):
            thought = _call_private_thought(
                config=config,
                round_spec=round_spec,
                agent=speaker,
                transcript=[*transcript, *statements],
                client=client,
                model=model,
                temperature=temperature,
                turn_number=turn_index + 1,
                private_memory=private_memory,
                event_callback=event_callback,
            )
            statements.append(thought)
            continue

        if round_spec.mode == "interruptions":
            speaker, decision = choose_interrupting_agent(
                agents=participants,
                config=config,
                round_spec=round_spec,
                transcript=[*transcript, *statements],
                turn_number=turn_index + 1,
                classifier=interruption_classifier
                or DeterministicInterruptionClassifier(),
            )
            event_type = "interrupt" if decision.should_interrupt else "debate"
        elif round_spec.mode == "discussion":
            candidate, decision = choose_interrupting_agent(
                agents=participants,
                config=config,
                round_spec=round_spec,
                transcript=[*transcript, *statements],
                turn_number=turn_index + 1,
                classifier=interruption_classifier
                or DeterministicInterruptionClassifier(),
            )
            if _should_cut_in(
                scheduled_speaker=speaker,
                candidate=candidate,
                decision=decision,
                transcript=[*transcript, *statements],
                turn_index=turn_index,
            ):
                speaker = candidate
                event_type = "interrupt"
        else:
            decision = None
        statement_decision = decision if event_type == "interrupt" or round_spec.mode == "interruptions" else None
        statement = _call_agent(
            config=config,
            round_spec=round_spec,
            agent=speaker,
            transcript=[*transcript, *statements],
            client=client,
            model=model,
            temperature=temperature,
            turn_number=turn_index + 1,
            event_type=event_type,
            extra={
                "current_speaker": speaker.name,
                "turns_remaining": max_turns - turn_index - 1,
                "interruption_score": (
                    statement_decision.score
                    if statement_decision
                    else deterministic_interruption_score(
                        speaker,
                        [*transcript, *statements],
                    )
                ),
                "interruption_rationale": statement_decision.rationale if statement_decision else "",
            },
            decision=statement_decision,
            event_callback=event_callback,
            private_memory=private_memory,
        )
        statements.append(statement)
    return statements


def _should_stay_quiet(
    *,
    agent: AgentPersona,
    transcript: list[Statement],
    turn_index: int,
) -> bool:
    if turn_index == 0:
        return False
    if transcript and transcript[-1].agent_id == agent.id:
        return False
    personality = agent.personality
    extraversion = _number(personality.get("extraversion"), 5.0)
    assertiveness = _number(personality.get("assertiveness"), 5.0)
    patience = _number(personality.get("patience"), 5.0)
    quiet_score = patience + (10.0 - extraversion) + (10.0 - assertiveness)
    threshold = 16.5
    return quiet_score >= threshold and turn_index % 3 == 2


def _should_cut_in(
    *,
    scheduled_speaker: AgentPersona,
    candidate: AgentPersona,
    decision: InterruptionDecision,
    transcript: list[Statement],
    turn_index: int,
) -> bool:
    if turn_index == 0:
        return False
    if candidate.id == scheduled_speaker.id:
        return False
    if not decision.should_interrupt or decision.score < 7.0:
        return False
    recent = transcript[-2:]
    if any(statement.event_type == "interrupt" for statement in recent):
        return False
    return True


def _speaker_for_turn(
    *,
    participants: list[AgentPersona],
    round_spec: RoundSpec,
    turn_index: int,
) -> AgentPersona:
    if round_spec.turn_strategy == "interruption_priority":
        ranked = sorted(
            participants,
            key=_interruption_score,
            reverse=True,
        )
        return ranked[turn_index % len(ranked)]
    return participants[turn_index % len(participants)]


def _interruption_score(agent: AgentPersona) -> float:
    personality = agent.personality
    return (
        _number(personality.get("assertiveness"), 3)
        + _number(personality.get("skepticism"), 3)
        + _number(personality.get("urgency"), 3)
        - (_number(personality.get("agreeableness"), 3) * 0.35)
    )


def _number(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _call_agent(
    *,
    config: CrewConfig,
    round_spec: RoundSpec,
    agent: AgentPersona,
    transcript: list[Statement],
    client: ModelClient,
    model: str,
    temperature: float,
    turn_number: int,
    event_type: str,
    extra: dict[str, Any] | None = None,
    decision: InterruptionDecision | None = None,
    event_callback: Callable[[str, dict[str, Any]], None] | None = None,
    private_memory: dict[str, list[str]] | None = None,
) -> Statement:
    memory = private_memory or {}
    context = build_context(
        config=config,
        round_spec=round_spec,
        transcript=transcript,
        agent=agent,
        extra={
            "turn_number": turn_number,
            "event_type": event_type,
            "private_memory": _format_private_memory(memory.get(agent.id, [])),
            **(extra or {}),
        },
    )
    system_prompt = build_agent_system_prompt(config, agent)
    user_prompt = render_template(round_spec.prompt, context)
    _emit(
        event_callback,
        "agent_start",
        {
            "agent_id": agent.id,
            "agent_name": agent.name,
            "round_id": round_spec.id,
            "round_title": round_spec.title,
            "round_mode": round_spec.mode,
            "turn_number": turn_number,
            "event_type": event_type,
            "interruption_score": (
                decision.score
                if decision
                else deterministic_interruption_score(agent, transcript)
            ),
            "interruption_rationale": decision.rationale if decision else None,
        },
    )
    content = _clean_conversation_text(
        client.complete(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model=model,
            temperature=temperature,
            metadata={
                "agent_id": agent.id,
                "agent_name": agent.name,
                "round_id": round_spec.id,
                "turn_number": turn_number,
                "event_type": event_type,
            },
        )
    )
    statement = Statement(
        agent_id=agent.id,
        agent_name=agent.name,
        round_id=round_spec.id,
        turn_number=turn_number,
        event_type=event_type,
        content=content,
        metadata={
            "interruption_score": (
                decision.score
                if decision
                else deterministic_interruption_score(agent, transcript)
            ),
            "classifier_interruption_score": decision.score if decision else None,
            "classifier_source": decision.source if decision else "deterministic-formula",
            "classifier_should_interrupt": decision.should_interrupt if decision else None,
            "classifier_rationale": decision.rationale if decision else None,
            "round_title": round_spec.title,
            "round_mode": round_spec.mode,
        },
    )
    _emit(
        event_callback,
        "statement",
        {
            "statement": statement,
            "round_id": round_spec.id,
            "round_title": round_spec.title,
        },
    )
    return statement


def _call_private_thought(
    *,
    config: CrewConfig,
    round_spec: RoundSpec,
    agent: AgentPersona,
    transcript: list[Statement],
    client: ModelClient,
    model: str,
    temperature: float,
    turn_number: int,
    private_memory: dict[str, list[str]],
    event_callback: Callable[[str, dict[str, Any]], None] | None = None,
) -> Statement:
    context = build_context(
        config=config,
        round_spec=round_spec,
        transcript=transcript,
        agent=agent,
        extra={
            "turn_number": turn_number,
            "event_type": "thought",
            "current_speaker": agent.name,
            "turns_remaining": (round_spec.max_turns or 0) - turn_number,
            "private_memory": _format_private_memory(private_memory.get(agent.id, [])),
        },
    )
    system_prompt = build_agent_system_prompt(config, agent)
    user_prompt = render_template(
        "You are {agent_name}. You decide not to speak right now. "
        "Write one short private thought that will inform your next spoken turn. "
        "Do not address the group.\n\nTranscript:\n{transcript}\n\nPrior private notes:\n{private_memory}",
        context,
    )
    _emit(
        event_callback,
        "agent_start",
        {
            "agent_id": agent.id,
            "agent_name": agent.name,
            "round_id": round_spec.id,
            "round_title": round_spec.title,
            "round_mode": round_spec.mode,
            "turn_number": turn_number,
            "event_type": "thought",
            "interruption_score": None,
            "interruption_rationale": None,
        },
    )
    content = _clean_conversation_text(
        client.complete(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model=model,
            temperature=temperature,
            metadata={
                "agent_id": agent.id,
                "agent_name": agent.name,
                "round_id": round_spec.id,
                "turn_number": turn_number,
                "event_type": "thought",
            },
        )
    )
    private_memory.setdefault(agent.id, []).append(content)
    statement = Statement(
        agent_id=agent.id,
        agent_name=agent.name,
        round_id=round_spec.id,
        turn_number=turn_number,
        event_type="thought",
        content="[stays quiet]",
        metadata={
            "round_title": round_spec.title,
            "round_mode": round_spec.mode,
            "private_thought": content,
        },
    )
    _emit(
        event_callback,
        "statement",
        {
            "statement": statement,
            "round_id": round_spec.id,
            "round_title": round_spec.title,
        },
    )
    return statement


def _call_recorder(
    *,
    config: CrewConfig,
    round_spec: RoundSpec,
    transcript: list[Statement],
    client: ModelClient,
    model: str,
    temperature: float,
    event_callback: Callable[[str, dict[str, Any]], None] | None = None,
    private_memory: dict[str, list[str]] | None = None,
) -> Statement:
    memory = private_memory or {}
    context = build_context(
        config=config,
        round_spec=round_spec,
        transcript=transcript,
        extra={
            "turn_number": "synthesis",
            "event_type": "synthesis",
            "current_speaker": round_spec.speaker_name,
            "turns_remaining": 0,
            "private_memory": _format_all_private_memory(memory),
        },
    )
    system_prompt = "\n\n".join(
        part
        for part in [
            config.harness.shared_instructions,
            round_spec.speaker_prompt,
            config.harness.output_contract,
        ]
        if part.strip()
    )
    user_prompt = render_template(round_spec.prompt, context)
    _emit(
        event_callback,
        "agent_start",
        {
            "agent_id": round_spec.speaker_id,
            "agent_name": round_spec.speaker_name,
            "round_id": round_spec.id,
            "round_title": round_spec.title,
            "round_mode": round_spec.mode,
            "turn_number": "synthesis",
            "event_type": "synthesis",
            "interruption_score": None,
            "interruption_rationale": None,
        },
    )
    content = _clean_conversation_text(
        client.complete(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model=model,
            temperature=temperature,
            metadata={
                "agent_id": round_spec.speaker_id,
                "agent_name": round_spec.speaker_name,
                "round_id": round_spec.id,
                "turn_number": "synthesis",
                "event_type": "synthesis",
            },
        )
    )
    statement = Statement(
        agent_id=round_spec.speaker_id,
        agent_name=round_spec.speaker_name,
        round_id=round_spec.id,
        turn_number="synthesis",
        event_type="synthesis",
        content=content,
        metadata={"round_title": round_spec.title, "round_mode": round_spec.mode},
    )
    _emit(
        event_callback,
        "statement",
        {
            "statement": statement,
            "round_id": round_spec.id,
            "round_title": round_spec.title,
        },
    )
    return statement


def _emit(
    event_callback: Callable[[str, dict[str, Any]], None] | None,
    event_type: str,
    payload: dict[str, Any],
) -> None:
    if event_callback:
        event_callback(event_type, payload)


def _format_private_memory(notes: list[str]) -> str:
    if not notes:
        return "No private notes."
    return "\n".join(f"- {note}" for note in notes[-3:])


def _format_all_private_memory(memory: dict[str, list[str]]) -> str:
    if not memory:
        return "No private notes."
    blocks: list[str] = []
    for agent_id, notes in memory.items():
        if notes:
            blocks.append(f"{agent_id}:\n" + "\n".join(f"- {note}" for note in notes[-3:]))
    return "\n\n".join(blocks) if blocks else "No private notes."


_LEADING_LABEL_RE = re.compile(
    r"^\s*(?:[-*]\s*)?(?:\*\*)?"
    r"(recommendation|strongest reason|biggest risk|must resolve|reason|risk|"
    r"next step|consensus|dissent|interrupt|build)"
    r"\s*:\s*(?:\*\*)?\s*",
    re.IGNORECASE,
)


def _clean_conversation_text(content: str) -> str:
    cleaned_lines: list[str] = []
    for raw_line in content.replace("**", "").replace("__", "").splitlines():
        line = raw_line.strip()
        line = re.sub(r"^#{1,6}\s*", "", line)
        line = re.sub(r"^[-*]\s+", "", line)
        line = re.sub(r"^\d+[.)]\s+", "", line)
        line = _LEADING_LABEL_RE.sub("", line).strip()
        if line:
            cleaned_lines.append(line)
    return "\n".join(cleaned_lines).strip()
