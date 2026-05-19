from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Protocol

from simula_crew.clients import ModelClient
from simula_crew.prompts import format_transcript
from simula_crew.schema import AgentPersona, CrewConfig, RoundSpec, Statement


@dataclass(frozen=True)
class InterruptionDecision:
    agent_id: str
    score: float
    should_interrupt: bool
    rationale: str
    source: str


class InterruptionClassifier(Protocol):
    def score(
        self,
        *,
        agent: AgentPersona,
        config: CrewConfig,
        round_spec: RoundSpec,
        transcript: list[Statement],
        turn_number: int,
    ) -> InterruptionDecision:
        """Return an interruption decision for an agent at a specific turn."""


class DeterministicInterruptionClassifier:
    """Personality-derived interruption propensity for repeatable tests and dry runs."""

    source = "deterministic"

    def score(
        self,
        *,
        agent: AgentPersona,
        config: CrewConfig,
        round_spec: RoundSpec,
        transcript: list[Statement],
        turn_number: int,
    ) -> InterruptionDecision:
        score = deterministic_interruption_score(agent, transcript)
        should_interrupt = score >= 6.5 and bool(transcript)
        rationale = (
            "Current interruption propensity from personality, contestation pressure, "
            "and recent speaking history."
        )
        return InterruptionDecision(
            agent_id=agent.id,
            score=score,
            should_interrupt=should_interrupt,
            rationale=rationale,
            source=self.source,
        )


class LLMInterruptionClassifier:
    """LLM classifier for context-sensitive interruption decisions."""

    source = "llm"

    def __init__(
        self,
        *,
        client: ModelClient,
        model: str,
        fallback: InterruptionClassifier | None = None,
    ) -> None:
        self.client = client
        self.model = model
        self.fallback = fallback or DeterministicInterruptionClassifier()

    def score(
        self,
        *,
        agent: AgentPersona,
        config: CrewConfig,
        round_spec: RoundSpec,
        transcript: list[Statement],
        turn_number: int,
    ) -> InterruptionDecision:
        if not transcript:
            return InterruptionDecision(
                agent_id=agent.id,
                score=0.0,
                should_interrupt=False,
                rationale="No prior transcript to interrupt.",
                source=self.source,
            )

        system_prompt = (
            "You classify an agent's current propensity to interrupt a live "
            "working-group deliberation. Interruption means cutting in because "
            "the current direction is wrong, vague, premature, or missing an "
            "important merge. Return compact JSON only. Do not write prose."
        )
        user_prompt = f"""
Agent:
- id: {agent.id}
- name: {agent.name}
- base prompt: {agent.base_prompt}
- personality: {json.dumps(agent.personality, sort_keys=True)}
- goals: {json.dumps(agent.goals)}
- constraints: {json.dumps(agent.constraints)}

Topic:
{config.topic.prompt}

Round:
- title: {round_spec.title}
- mode: {round_spec.mode}
- turn number: {turn_number}

Interruption rules:
{chr(10).join(f"- {rule}" for rule in config.harness.interruption_rules)}

Transcript:
{format_transcript(transcript, visibility="named")}

Classify how strongly this agent would interrupt right now. Consider:
- the agent's interruption-related personality traits
- whether the transcript contains a claim they would contest
- whether they spoke recently and should yield
- whether interrupting would move the group forward

Return JSON with:
{{
  "score": number from 0 to 10, where 0 means will not interrupt and 10 means almost certainly interrupts,
  "should_interrupt": boolean,
  "rationale": "one sentence"
}}
"""
        try:
            raw = self.client.complete(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                model=self.model,
                temperature=0.0,
                metadata={
                    "agent_id": agent.id,
                    "agent_name": agent.name,
                    "event_type": "interruption_classification",
                    "turn_number": turn_number,
                },
            )
            payload = _extract_json(raw)
            score = _clamp(float(payload.get("score", 0.0)), 0.0, 10.0)
            should_interrupt = bool(payload.get("should_interrupt", score >= 6.5))
            rationale = str(payload.get("rationale", "LLM classified interruption fit."))
            return InterruptionDecision(
                agent_id=agent.id,
                score=score,
                should_interrupt=should_interrupt,
                rationale=rationale,
                source=self.source,
            )
        except Exception as exc:
            fallback = self.fallback.score(
                agent=agent,
                config=config,
                round_spec=round_spec,
                transcript=transcript,
                turn_number=turn_number,
            )
            return InterruptionDecision(
                agent_id=agent.id,
                score=fallback.score,
                should_interrupt=fallback.should_interrupt,
                rationale=f"LLM classifier failed; fallback used. {exc}",
                source="llm-fallback",
            )


def deterministic_interruption_score(
    agent: AgentPersona,
    transcript: list[Statement] | None = None,
) -> float:
    personality = agent.personality
    transcript = transcript or []
    recent_turns = transcript[-6:]
    recent_speaks = sum(1 for statement in recent_turns if statement.agent_id == agent.id)
    last_speaker_penalty = 2.6 if transcript and transcript[-1].agent_id == agent.id else 0.0
    silence_bonus = 0.9 if recent_speaks == 0 and transcript else 0.0
    repeated_speaker_penalty = recent_speaks * 1.15
    contestation_pressure = 0.7 if transcript else 0.0

    assertiveness = _number(personality.get("assertiveness"), 5.0)
    skepticism = _number(personality.get("skepticism"), 5.0)
    urgency = _number(personality.get("urgency"), 5.0)
    extraversion = _number(personality.get("extraversion"), 5.0)
    agreeableness = _number(personality.get("agreeableness"), 5.0)
    conscientiousness = _number(personality.get("conscientiousness"), 5.0)
    interruptiveness = _number(personality.get("interruptiveness"), assertiveness)
    disagreement_sensitivity = _number(
        personality.get("disagreement_sensitivity"),
        skepticism,
    )
    patience = _number(
        personality.get("patience"),
        (agreeableness + conscientiousness) / 2,
    )

    score = (
        5.0
        + (assertiveness - 5.0) * 0.45
        + (skepticism - 5.0) * 0.35
        + (urgency - 5.0) * 0.35
        + (extraversion - 5.0) * 0.15
        + (interruptiveness - 5.0) * 0.75
        + (disagreement_sensitivity - 5.0) * 0.5
        - (agreeableness - 5.0) * 0.35
        - (conscientiousness - 5.0) * 0.15
        - (patience - 5.0) * 0.55
        + contestation_pressure
        + silence_bonus
        - repeated_speaker_penalty
        - last_speaker_penalty
    )
    return round(_clamp(score, 0.0, 10.0), 2)


def choose_interrupting_agent(
    *,
    agents: list[AgentPersona],
    config: CrewConfig,
    round_spec: RoundSpec,
    transcript: list[Statement],
    turn_number: int,
    classifier: InterruptionClassifier,
) -> tuple[AgentPersona, InterruptionDecision]:
    decisions = [
        classifier.score(
            agent=agent,
            config=config,
            round_spec=round_spec,
            transcript=transcript,
            turn_number=turn_number,
        )
        for agent in agents
    ]
    decision_by_id = {decision.agent_id: decision for decision in decisions}
    interrupting = [decision for decision in decisions if decision.should_interrupt]
    ranked = sorted(interrupting or decisions, key=lambda decision: decision.score, reverse=True)
    chosen_decision = ranked[0]
    if transcript and chosen_decision.agent_id == transcript[-1].agent_id:
        chosen_decision = _near_tie_alternative(ranked, chosen_decision)
    chosen_agent = next(agent for agent in agents if agent.id == chosen_decision.agent_id)
    return chosen_agent, decision_by_id[chosen_agent.id]


def _near_tie_alternative(
    ranked: list[InterruptionDecision],
    leader: InterruptionDecision,
) -> InterruptionDecision:
    for candidate in ranked[1:]:
        if candidate.score >= leader.score - 1.25:
            return candidate
    return leader


def _number(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def _extract_json(raw: str) -> dict[str, Any]:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        raw = raw.removeprefix("json").strip()
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("No JSON object found in classifier response.")
    return json.loads(raw[start : end + 1])
