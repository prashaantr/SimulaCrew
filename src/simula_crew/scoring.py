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
    """Personality-derived interruption scorer for repeatable tests and dry runs."""

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
            "Computed from assertiveness, skepticism, urgency, extraversion, "
            "agreeableness, conscientiousness, and recent speaking frequency."
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
            "You classify whether an agent should interrupt a live group "
            "deliberation. Return compact JSON only. Do not write prose."
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

Classify whether this agent should interrupt right now.
Return JSON with:
{{
  "score": number from 0 to 10,
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
    recent_turns = transcript[-4:]
    recent_speaks = sum(1 for statement in recent_turns if statement.agent_id == agent.id)
    silence_bonus = 0.6 if recent_speaks == 0 and transcript else 0.0
    dominance_penalty = recent_speaks * 0.8

    score = (
        _number(personality.get("assertiveness"), 3.0) * 0.7
        + _number(personality.get("skepticism"), 3.0) * 0.8
        + _number(personality.get("urgency"), 3.0) * 0.55
        + _number(personality.get("extraversion"), 3.0) * 0.35
        + _number(personality.get("neuroticism"), 3.0) * 0.15
        - _number(personality.get("agreeableness"), 3.0) * 0.45
        - _number(personality.get("conscientiousness"), 3.0) * 0.1
        + silence_bonus
        - dominance_penalty
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
    chosen_decision = max(interrupting or decisions, key=lambda decision: decision.score)
    chosen_agent = next(agent for agent in agents if agent.id == chosen_decision.agent_id)
    return chosen_agent, decision_by_id[chosen_agent.id]


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
