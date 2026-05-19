from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import json
import re
import threading
import time
from typing import Any, Callable

from simula_crew.clients import ModelClient
from simula_crew.prompts import (
    build_agent_system_prompt,
    build_context,
    format_transcript,
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
    goal_tracker = GoalTracker.create(config=config, agents=agents)
    event_callback = _synchronized_callback(event_callback)
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
            statements = _run_private_round(
                config=config,
                round_spec=round_spec,
                participants=participants,
                transcript=list(transcript),
                client=client,
                model=model,
                temperature=temperature,
                event_callback=event_callback,
                private_memory=private_memory,
                goal_tracker=goal_tracker,
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
                goal_tracker=goal_tracker,
            )
        else:
            statements = [
                _call_recorder(
                    config=config,
                    round_spec=round_spec,
                    transcript=_public_transcript(transcript),
                    client=client,
                    model=model,
                    temperature=temperature,
                    event_callback=event_callback,
                    private_memory=private_memory,
                    goal_tracker=goal_tracker,
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


def _run_private_round(
    *,
    config: CrewConfig,
    round_spec: RoundSpec,
    participants: list[AgentPersona],
    transcript: list[Statement],
    client: ModelClient,
    model: str,
    temperature: float,
    event_callback: Callable[[str, dict[str, Any]], None] | None,
    private_memory: dict[str, list[str]],
    goal_tracker: "GoalTracker",
) -> list[Statement]:
    if not participants:
        return []

    results: dict[int, Statement] = {}
    max_workers = min(len(participants), max(1, int(_number(config.topic.variables.get("private_thinking_workers"), len(participants)))))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _call_agent,
                config=config,
                round_spec=round_spec,
                agent=agent,
                transcript=transcript,
                client=client,
                model=model,
                temperature=temperature,
                turn_number=index + 1,
                event_type="private",
                extra={
                    "turns_remaining": len(participants) - index - 1,
                    **goal_tracker.visible_prompt_context(),
                },
                event_callback=event_callback,
                private_memory=private_memory,
            ): (index, agent)
            for index, agent in enumerate(participants)
        }
        for future in as_completed(futures):
            index, agent = futures[future]
            statement = future.result()
            private_memory.setdefault(agent.id, []).append(statement.content)
            goal_tracker.attach_metadata(statement)
            results[index] = statement

    return [results[index] for index in sorted(results)]


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
    goal_tracker: "GoalTracker",
) -> list[Statement]:
    if not participants:
        return []

    max_turns = round_spec.max_turns or len(participants)
    started_at = time.monotonic()
    statements: list[Statement] = []
    for turn_index in range(max_turns):
        if round_spec.mode == "discussion" and goal_tracker.time_limit_reached(started_at):
            _emit(
                event_callback,
                "discussion_time_limit_reached",
                {
                    "round_id": round_spec.id,
                    "turn_number": turn_index + 1,
                    "time_limit_seconds": goal_tracker.time_limit_seconds,
                    "summary": goal_tracker.summary(),
                },
            )
            break

        decision: InterruptionDecision | None = None
        speaker = _speaker_for_turn(participants=participants, round_spec=round_spec, turn_index=turn_index)
        event_type = "debate"

        if round_spec.mode == "interruptions":
            speaker, decision = choose_interrupting_agent(
                agents=participants,
                config=config,
                round_spec=round_spec,
                transcript=_public_transcript([*transcript, *statements]),
                turn_number=turn_index + 1,
                classifier=interruption_classifier
                or DeterministicInterruptionClassifier(),
            )
            event_type = "interrupt" if decision.should_interrupt else "debate"
        elif round_spec.mode == "discussion":
            decisions = _score_agents(
                agents=participants,
                config=config,
                round_spec=round_spec,
                transcript=_public_transcript([*transcript, *statements]),
                turn_number=turn_index + 1,
                classifier=interruption_classifier
                or DeterministicInterruptionClassifier(),
            )
            speaker, decision = _choose_discussion_speaker(
                agents=participants,
                decisions=decisions,
                transcript=_public_transcript([*transcript, *statements]),
                turn_index=turn_index,
            )
            if _should_stay_quiet(
                agent=speaker,
                decision=decision,
                transcript=_public_transcript([*transcript, *statements]),
                turn_index=turn_index,
            ):
                thought = _call_private_thought(
                    config=config,
                    round_spec=round_spec,
                    agent=speaker,
                    transcript=_public_transcript([*transcript, *statements]),
                    client=client,
                    model=model,
                    temperature=temperature,
                    turn_number=turn_index + 1,
                    private_memory=private_memory,
                    event_callback=event_callback,
                    goal_tracker=goal_tracker,
                )
                statements.append(thought)
                continue
            if _should_cut_in(
                config=config,
                decision=decision,
                transcript=_public_transcript([*transcript, *statements]),
                turn_index=turn_index,
            ):
                event_type = "interrupt"
        else:
            decision = None
        statement_decision = decision if event_type == "interrupt" or round_spec.mode == "interruptions" else None
        statement = _call_agent(
            config=config,
            round_spec=round_spec,
            agent=speaker,
            transcript=_public_transcript([*transcript, *statements]),
            client=client,
            model=model,
            temperature=temperature,
            turn_number=turn_index + 1,
            event_type=event_type,
            extra={
                "current_speaker": speaker.name,
                "turns_remaining": max_turns - turn_index - 1,
                **goal_tracker.visible_prompt_context(),
                "interruption_score": (
                    statement_decision.score
                    if statement_decision
                    else deterministic_interruption_score(
                        speaker,
                        _public_transcript([*transcript, *statements]),
                    )
                ),
                "interruption_rationale": statement_decision.rationale if statement_decision else "",
            },
            decision=statement_decision,
            event_callback=event_callback,
            private_memory=private_memory,
        )
        goal_tracker.update(
            statement=statement,
            agents=participants,
            config=config,
            client=client,
            model=model,
            transcript=_public_transcript([*transcript, *statements, statement]),
            private_memory=private_memory,
        )
        _emit(
            event_callback,
            "alignment_update",
            {
                "round_id": round_spec.id,
                "turn_number": turn_index + 1,
                "summary": goal_tracker.summary(),
                "by_agent": goal_tracker.alignments,
                "current_idea": goal_tracker.current_idea,
                "agent_idea_views": goal_tracker.agent_idea_views,
                "aligned": goal_tracker.aligned,
                "average": goal_tracker.average,
                "minimum": goal_tracker.minimum,
            },
        )
        statements.append(statement)
        if round_spec.mode == "discussion" and goal_tracker.aligned:
            _emit(
                event_callback,
                "goal_aligned",
                {
                    "round_id": round_spec.id,
                    "turn_number": turn_index + 1,
                    "summary": goal_tracker.summary(),
                    "threshold": goal_tracker.threshold,
                },
            )
            break
    return statements


def _score_agents(
    *,
    agents: list[AgentPersona],
    config: CrewConfig,
    round_spec: RoundSpec,
    transcript: list[Statement],
    turn_number: int,
    classifier: InterruptionClassifier,
) -> list[InterruptionDecision]:
    return [
        classifier.score(
            agent=agent,
            config=config,
            round_spec=round_spec,
            transcript=transcript,
            turn_number=turn_number,
        )
        for agent in agents
    ]


def _public_transcript(statements: list[Statement]) -> list[Statement]:
    return [
        statement
        for statement in statements
        if statement.event_type not in {"private", "thought"}
    ]


def _choose_discussion_speaker(
    *,
    agents: list[AgentPersona],
    decisions: list[InterruptionDecision],
    transcript: list[Statement],
    turn_index: int,
) -> tuple[AgentPersona, InterruptionDecision]:
    agent_by_id = {agent.id: agent for agent in agents}
    recent = transcript[-6:]
    scores: list[tuple[float, InterruptionDecision]] = []
    for decision in decisions:
        recent_count = sum(1 for statement in recent if statement.agent_id == decision.agent_id)
        last_penalty = 3.0 if transcript and transcript[-1].agent_id == decision.agent_id else 0.0
        silence_bonus = 4.0 if transcript and recent_count == 0 else 0.0
        jitter = (_stable_fraction(decision.agent_id, turn_index, len(transcript)) - 0.5) * 1.2
        score = decision.score + silence_bonus + jitter - last_penalty - recent_count * 0.6
        scores.append((score, decision))

    _, chosen = max(scores, key=lambda item: item[0])
    return agent_by_id[chosen.agent_id], chosen


def _should_stay_quiet(
    *,
    agent: AgentPersona,
    decision: InterruptionDecision,
    transcript: list[Statement],
    turn_index: int,
) -> bool:
    if turn_index == 0:
        return False
    if decision.should_interrupt or decision.score >= 7.0:
        return False
    if transcript and transcript[-1].agent_id == agent.id:
        return False
    personality = agent.personality
    extraversion = _number(personality.get("extraversion"), 5.0)
    assertiveness = _number(personality.get("assertiveness"), 5.0)
    patience = _number(personality.get("patience"), 5.0)
    quiet_pressure = (patience + (10.0 - extraversion) + (10.0 - assertiveness)) / 30.0
    probability = min(0.45, max(0.05, quiet_pressure * 0.7))
    return _stable_fraction(agent.id, turn_index, len(transcript), "quiet") < probability


def _should_cut_in(
    *,
    config: CrewConfig,
    decision: InterruptionDecision,
    transcript: list[Statement],
    turn_index: int,
) -> bool:
    if turn_index == 0:
        return False
    min_public_turns = max(
        1,
        int(_number(config.topic.variables.get("min_public_turns_before_interruptions"), 3)),
    )
    if len(transcript) < min_public_turns:
        return False
    if not decision.should_interrupt or decision.score < 7.0:
        return False
    recent = transcript[-2:]
    if any(statement.event_type == "interrupt" for statement in recent):
        return False
    return True


def _stable_fraction(*parts: object) -> float:
    text = "|".join(str(part) for part in parts)
    total = 0
    for index, char in enumerate(text):
        total = (total * 131 + ord(char) + index) % 10000
    return total / 10000.0


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
    goal_tracker: "GoalTracker | None" = None,
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
            **((goal_tracker or GoalTracker.empty()).visible_prompt_context()),
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
    if goal_tracker:
        goal_tracker.attach_metadata(statement)
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
    goal_tracker: "GoalTracker | None" = None,
) -> Statement:
    memory = private_memory or {}
    goal = goal_tracker or GoalTracker.empty()
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
            **goal.visible_prompt_context(),
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
    goal.attach_metadata(statement)
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


def _synchronized_callback(
    event_callback: Callable[[str, dict[str, Any]], None] | None,
) -> Callable[[str, dict[str, Any]], None] | None:
    if event_callback is None:
        return None
    lock = threading.Lock()

    def handle(event_type: str, payload: dict[str, Any]) -> None:
        with lock:
            event_callback(event_type, payload)

    return handle


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


@dataclass
class GoalTracker:
    goal: str
    threshold: float
    min_turns: int
    time_limit_seconds: int
    current_idea: str
    evaluator: str
    fallback_self_step: float
    fallback_listener_step: float
    fallback_interrupt_factor: float
    alignments: dict[str, float]
    agent_idea_views: dict[str, str]
    public_turns: int = 0

    @classmethod
    def create(cls, *, config: CrewConfig, agents: list[AgentPersona]) -> "GoalTracker":
        variables = config.topic.variables
        goal = str(
            variables.get(
                "shared_goal",
                config.topic.success_criteria,
            )
        )
        threshold = _clamp_float(_number(variables.get("goal_alignment_threshold"), 0.84), 0.55, 0.98)
        min_turns = max(1, int(_number(variables.get("goal_alignment_min_turns"), 6)))
        time_limit_seconds = _discussion_time_limit_seconds(variables)
        default_start = _clamp_float(
            _number(variables.get("default_goal_alignment_start"), 0.45),
            0.0,
            1.0,
        )
        current_idea = str(variables.get("starting_idea", "No idea has been named yet."))
        return cls(
            goal=goal,
            threshold=threshold,
            min_turns=min_turns,
            time_limit_seconds=time_limit_seconds,
            current_idea=current_idea,
            evaluator=str(variables.get("goal_alignment_evaluator", "llm")).lower(),
            fallback_self_step=_clamp_float(
                _number(variables.get("goal_alignment_step_self"), 0.06),
                0.0,
                0.35,
            ),
            fallback_listener_step=_clamp_float(
                _number(variables.get("goal_alignment_step_listener"), 0.025),
                0.0,
                0.25,
            ),
            fallback_interrupt_factor=_clamp_float(
                _number(variables.get("goal_alignment_interrupt_factor"), 0.6),
                0.0,
                1.0,
            ),
            alignments={
                agent.id: _clamp_float(
                    _number(
                        agent.personality.get("goal_alignment_start"),
                        default_start,
                    ),
                    0.0,
                    1.0,
                )
                for agent in agents
            },
            agent_idea_views={agent.id: current_idea for agent in agents},
        )

    @classmethod
    def empty(cls) -> "GoalTracker":
        return cls(
            goal="No shared goal supplied.",
            threshold=1.0,
            min_turns=1,
            time_limit_seconds=20 * 60,
            current_idea="No idea has been named yet.",
            evaluator="off",
            fallback_self_step=0.0,
            fallback_listener_step=0.0,
            fallback_interrupt_factor=1.0,
            alignments={},
            agent_idea_views={},
        )

    @property
    def average(self) -> float:
        if not self.alignments:
            return 0.0
        return sum(self.alignments.values()) / len(self.alignments)

    @property
    def minimum(self) -> float:
        if not self.alignments:
            return 0.0
        return min(self.alignments.values())

    @property
    def aligned(self) -> bool:
        return bool(self.alignments) and self.public_turns >= self.min_turns and self.minimum >= self.threshold

    def time_limit_reached(self, started_at: float) -> bool:
        return self.time_limit_seconds > 0 and time.monotonic() - started_at >= self.time_limit_seconds

    def prompt_context(self, agent_id: str | None = None) -> dict[str, str]:
        current = self.alignments.get(agent_id, self.average) if agent_id else self.average
        return {
            "alignment_goal": self.goal,
            "goal_alignment": f"{current:.0%}",
            "goal_alignment_summary": self.summary(),
            "goal_alignment_threshold": f"{self.threshold:.0%}",
            "current_idea": self.current_idea,
            "agent_idea_view": self.agent_idea_views.get(agent_id, self.current_idea),
            "agent_idea_views_summary": self.idea_views_summary(),
            "discussion_time_limit": _format_seconds(self.time_limit_seconds),
        }

    def visible_prompt_context(self) -> dict[str, str]:
        """Context safe to show to speaking agents and the recorder."""
        return {
            "alignment_goal": self.goal,
            "discussion_time_limit": _format_seconds(self.time_limit_seconds),
        }

    def summary(self) -> str:
        if not self.alignments:
            return "No alignment state."
        parts = [f"{agent_id}={score:.0%}" for agent_id, score in sorted(self.alignments.items())]
        return f"avg={self.average:.0%}, min={self.minimum:.0%}, " + ", ".join(parts)

    def idea_views_summary(self) -> str:
        if not self.agent_idea_views:
            return "No idea views yet."
        return "\n".join(
            f"- {agent_id}: {view}"
            for agent_id, view in sorted(self.agent_idea_views.items())
        )

    def attach_metadata(self, statement: Statement) -> None:
        statement.metadata["goal"] = self.goal
        statement.metadata["current_idea"] = self.current_idea
        statement.metadata["agent_idea_views"] = dict(self.agent_idea_views)
        statement.metadata["goal_alignment_average"] = round(self.average, 4)
        statement.metadata["goal_alignment_minimum"] = round(self.minimum, 4)
        statement.metadata["goal_alignment_threshold"] = self.threshold
        statement.metadata["goal_alignment_by_agent"] = {
            agent_id: round(score, 4)
            for agent_id, score in self.alignments.items()
        }
        statement.metadata["goal_aligned"] = self.aligned

    def update(
        self,
        *,
        statement: Statement,
        agents: list[AgentPersona],
        config: CrewConfig,
        client: ModelClient,
        model: str,
        transcript: list[Statement],
        private_memory: dict[str, list[str]],
    ) -> None:
        if statement.event_type in {"thought", "private", "synthesis"}:
            self.attach_metadata(statement)
            return

        before = dict(self.alignments)
        self.public_turns += 1
        source = "config-fallback"
        rationale = "Fallback alignment update from configured step sizes."
        if self.evaluator == "llm" and client.provider != "dry-run":
            agent_states = self._evaluate_agent_states_with_llm(
                statement=statement,
                agents=agents,
                config=config,
                client=client,
                model=model,
                transcript=transcript,
                private_memory=private_memory,
            )
            if agent_states:
                source = "llm-agent-states"
                rationale = "Parallel per-agent idea and buy-in state update."
                self._apply_agent_states(agent_states, before)
            else:
                self._apply_fallback_alignment(statement)
        else:
            self._apply_fallback_alignment(statement)
        self.attach_metadata(statement)
        statement.metadata["goal_alignment_before"] = {
            agent_id: round(score, 4)
            for agent_id, score in before.items()
        }
        statement.metadata["goal_alignment_delta"] = {
            agent_id: round(self.alignments[agent_id] - before[agent_id], 4)
            for agent_id in self.alignments
        }
        statement.metadata["goal_alignment_source"] = source
        statement.metadata["goal_alignment_rationale"] = rationale
        statement.metadata["agent_state_rationales"] = {
            agent_id: state.get("rationale", "")
            for agent_id, state in getattr(self, "_last_agent_states", {}).items()
        }

    def _apply_fallback_alignment(self, statement: Statement) -> None:
        factor = self.fallback_interrupt_factor if statement.event_type == "interrupt" else 1.0
        self.agent_idea_views[statement.agent_id] = _fallback_current_idea(statement)
        for agent_id in self.alignments:
            step = self.fallback_self_step if agent_id == statement.agent_id else self.fallback_listener_step
            self.alignments[agent_id] = _clamp_float(
                self.alignments[agent_id] + step * factor,
                0.0,
                1.0,
            )
        self.current_idea = _derive_current_idea(
            alignments=self.alignments,
            views=self.agent_idea_views,
        )

    def _apply_agent_states(
        self,
        agent_states: dict[str, dict[str, Any]],
        before: dict[str, float],
    ) -> None:
        self._last_agent_states = agent_states
        for agent_id, prior_score in before.items():
            state = agent_states.get(agent_id, {})
            view = state.get("idea")
            if isinstance(view, str) and view.strip():
                self.agent_idea_views[agent_id] = view.strip()
            self.alignments[agent_id] = _clamp_float(
                _number(state.get("buy_in"), prior_score),
                0.0,
                1.0,
            )
        self.current_idea = _derive_current_idea(
            alignments=self.alignments,
            views=self.agent_idea_views,
        )

    def _evaluate_agent_states_with_llm(
        self,
        *,
        statement: Statement,
        agents: list[AgentPersona],
        config: CrewConfig,
        client: ModelClient,
        model: str,
        transcript: list[Statement],
        private_memory: dict[str, list[str]],
    ) -> dict[str, dict[str, Any]] | None:
        workers = min(
            len(agents),
            max(
                1,
                int(_number(config.topic.variables.get("idea_state_workers"), len(agents))),
            ),
        )
        state_model = _idea_state_model(config=config, client=client, model=model)
        states: dict[str, dict[str, Any]] = {}
        try:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {
                    executor.submit(
                        self._evaluate_one_agent_state,
                        agent=agent,
                        config=config,
                        client=client,
                        model=state_model,
                        transcript=transcript,
                        statement=statement,
                        private_notes=private_memory.get(agent.id, []),
                    ): agent
                    for agent in agents
                }
                for future in as_completed(futures):
                    agent = futures[future]
                    states[agent.id] = future.result()
        except Exception as exc:
            statement.metadata["goal_alignment_evaluator_error"] = str(exc)
            return None
        return states if states else None

    def _evaluate_one_agent_state(
        self,
        *,
        agent: AgentPersona,
        config: CrewConfig,
        client: ModelClient,
        model: str,
        transcript: list[Statement],
        statement: Statement,
        private_notes: list[str],
    ) -> dict[str, Any]:
        system_prompt = "\n\n".join(
            part
            for part in [
                build_agent_system_prompt(config, agent),
                (
                    "You are not speaking to the group. You are updating this "
                    "agent's private state after a public turn. Return compact "
                    "JSON only."
                ),
            ]
            if part.strip()
        )
        user_prompt = f"""
Shared goal:
{self.goal}

Agent:
- id: {agent.id}
- name: {agent.name}

Previous private idea view for this agent:
{self.agent_idea_views.get(agent.id, self.current_idea)}

Previous buy-in for this agent:
{self.alignments.get(agent.id, self.average):.2f}

Latest public statement:
{statement.agent_name}: {statement.content}

Private notes for this agent:
{_format_private_memory(private_notes)}

Transcript:
{format_transcript(transcript, visibility="named")}

Return JSON with this shape:
{{
  "idea": "one short phrase describing what this agent currently thinks the proposal is",
  "buy_in": 0.0,
  "rationale": "one sentence explaining why this agent's buy-in moved or stayed still"
}}

Buy-in is 0.0 to 1.0. It means how willing this agent is to move forward with the current idea as the PRD target.
"""
        raw = client.complete(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model=model,
            temperature=0.0,
            metadata={
                "agent_id": agent.id,
                "agent_name": agent.name,
                "round_id": statement.round_id,
                "turn_number": statement.turn_number,
                "event_type": "idea_state_evaluation",
            },
        )
        payload = _extract_json_object(raw)
        return {
            "idea": str(payload.get("idea", self.agent_idea_views.get(agent.id, self.current_idea))),
            "buy_in": _clamp_float(
                _number(payload.get("buy_in"), self.alignments.get(agent.id, self.average)),
                0.0,
                1.0,
            ),
            "rationale": str(payload.get("rationale", "")),
        }


def _discussion_time_limit_seconds(variables: dict[str, Any]) -> int:
    if "discussion_time_limit_seconds" in variables:
        return max(1, int(_number(variables.get("discussion_time_limit_seconds"), 20 * 60)))
    if "discussion_time_limit_minutes" in variables:
        minutes = _number(variables.get("discussion_time_limit_minutes"), 20.0)
        return max(1, int(minutes * 60))
    return 20 * 60


def _idea_state_model(*, config: CrewConfig, client: ModelClient, model: str) -> str:
    configured = config.topic.variables.get("idea_state_model")
    if configured:
        return str(configured)
    if client.provider == "claude":
        return "haiku"
    return model


def _derive_current_idea(
    *,
    alignments: dict[str, float],
    views: dict[str, str],
) -> str:
    candidates = [
        (alignments.get(agent_id, 0.0), view.strip())
        for agent_id, view in views.items()
        if view and view.strip()
    ]
    if not candidates:
        return "No concrete idea yet."
    _, view = max(candidates, key=lambda item: item[0])
    return view


def _format_seconds(seconds: int) -> str:
    minutes, remainder = divmod(max(0, seconds), 60)
    if minutes and not remainder:
        return f"{minutes} minutes"
    if minutes:
        return f"{minutes} minutes {remainder} seconds"
    return f"{remainder} seconds"


def _fallback_current_idea(statement: Statement) -> str:
    content = " ".join(statement.content.split())
    if not content:
        return "No concrete idea yet."
    first_sentence = re.split(r"(?<=[.!?])\s+", content, maxsplit=1)[0]
    if len(first_sentence) <= 120:
        return first_sentence
    return first_sentence[:117].rstrip() + "..."


def _clamp_float(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def _extract_json_object(raw: str) -> dict[str, Any]:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        raw = raw.removeprefix("json").strip()
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("No JSON object found in model response.")
    return json.loads(raw[start : end + 1])


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
