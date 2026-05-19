from __future__ import annotations

import asyncio
import json
import os
from typing import Any, Protocol


class ModelClient(Protocol):
    provider: str

    def complete(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        model: str,
        temperature: float = 0.2,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Return one model completion."""


class DryRunClient:
    provider = "dry-run"

    def complete(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        model: str,
        temperature: float = 0.2,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        metadata = metadata or {}
        agent_name = metadata.get("agent_name", "Recorder")
        event_type = metadata.get("event_type", "statement")
        turn_number = metadata.get("turn_number", "?")

        if event_type == "interruption_classification":
            score = _dry_run_interruption_score(agent_name, turn_number)
            return json.dumps(
                {
                    "score": score,
                    "should_interrupt": score >= 6.5,
                    "rationale": "Dry-run classifier: persona pressure plus turn rotation.",
                }
            )

        return _dry_run_response(
            agent_name=agent_name,
            event_type=event_type,
            turn_number=turn_number,
            topic=_extract_topic(user_prompt),
        )


class OpenAIClient:
    provider = "openai"

    def __init__(self, api_key: str | None = None) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "OpenAI support requires: pip install -e '.[openai]'"
            ) from exc

        resolved_api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not resolved_api_key:
            raise RuntimeError("OPENAI_API_KEY is required for --provider openai.")
        self._client = OpenAI(api_key=resolved_api_key)

    def complete(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        model: str,
        temperature: float = 0.2,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        response = self._client.chat.completions.create(
            model=model,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        content = response.choices[0].message.content
        if not content:
            raise RuntimeError("OpenAI returned an empty response.")
        return content


def _extract_topic(user_prompt: str) -> str:
    for line in user_prompt.splitlines():
        if line.lower().startswith("topic:"):
            return line.split(":", 1)[1].strip()
    return "the prompt"


def _dry_run_interruption_score(agent_name: str, turn_number: Any) -> float:
    base_scores = {
        "Mara": 8.0,
        "Niko": 8.4,
        "Sol": 5.8,
        "June": 6.6,
    }
    try:
        turn = int(turn_number)
    except (TypeError, ValueError):
        turn = 1
    return round(max(0.0, min(10.0, base_scores.get(agent_name, 6.0) - (turn % 3) * 0.4)), 1)


def _dry_run_response(
    *,
    agent_name: str,
    event_type: str,
    turn_number: Any,
    topic: str,
) -> str:
    if event_type == "synthesis":
        return (
            "# PRD: SimulaCrew Hackathon Harness\n\n"
            "## Product Idea\n"
            "A CLI that spins up personality-driven agents, lets them converge on one "
            "hackathon idea, and turns the discussion into a compact PRD.\n\n"
            "## Main Artifact\n"
            "A saved simulation result containing the public transcript, private thinking, "
            "format validation, and final synthesized deliverable.\n\n"
            "## Target User\n"
            "Builders who want to test group-agent behavior before wiring a domain workflow.\n\n"
            "## Problem\n"
            "Small teams need a fast way to see how personality, dissent, and turn-taking "
            "change the artifact a group produces.\n\n"
            "## Proposed Solution\n"
            "Run a configurable multi-agent deliberation with private positions, public "
            "discussion, interruption mechanics, and a neutral synthesis step.\n\n"
            "## Core User Flow\n"
            "Choose an experiment, run it from the CLI, watch the live discussion, then "
            "inspect the saved transcript and final task result.\n\n"
            "## Rationale\n"
            "A narrow harness gives the team a concrete way to test whether simulated "
            "deliberation produces useful, inspectable artifacts before expanding scope.\n\n"
            "## MVP Scope\n"
            "Run one preset, show live group chat with natural cut-ins, preserve private "
            "thoughts in JSON, and output the PRD at the end.\n\n"
            "## Out of Scope\n"
            "Fine-tuning models, real-time human participation, and broad workflow "
            "automation beyond the first runnable experiment.\n\n"
            "## Risks\n"
            "The simulation may sound plausible without being valid, so logs, scoring, "
            "and preserved dissent must remain easy to inspect.\n\n"
            "## Open Questions\n"
            "Which evaluation metric should decide whether a simulated team prediction "
            "matches a real team's output?\n\n"
            "## Next Steps\n"
            "Run the preset, compare the result with a real team artifact, and tune the "
            "process only where the transcript exposes a clear failure mode.\n\n"
            "## Immediate Build Plan\n"
            "Keep the harness first, use one concrete preset as the acceptance test, and "
            "inspect interruption and convergence metadata only when debugging."
        )

    private = {
        "Mara": (
            "I would start with the harness because every later simulator needs spawning, "
            "turns, interruptions, and logs.\n"
            "The risk is over-abstracting before anything works, so I want the smallest "
            "runnable demo today."
        ),
        "Niko": (
            "I only want the harness first if evaluation hooks are first-class.\n"
            "Fake realism is worse than a simple scripted debate, so the group needs to "
            "inspect why an agent cut in."
        ),
        "Sol": (
            "I would use the harness as the spine and one domain preset as the test.\n"
            "That keeps architecture and reality checks together, as long as the output "
            "does not smooth away dissent."
        ),
        "June": (
            "I want the CLI understandable first.\n"
            "Users need to see agents thinking, chatting, and cutting in without digging "
            "through too many knobs."
        ),
    }
    if event_type == "private":
        return private.get(agent_name, f"I would focus on {topic} and keep the first run small.")

    debate = {
        "Mara": [
            "I would ship the harness first, but constrain it to one preset so it proves something today.",
            "The preset can be moral deliberation, but the reusable part is spawning, turns, and logs.",
            "So my consensus offer is: harness first, moral simulator as the first acceptance test.",
        ],
        "Niko": [
            "I disagree with a bare harness. Add the inspection trail, or we cannot tell simulation from roleplay.",
            "The minimum is an evaluation hook and a printed rationale for every interruption.",
            "I can accept harness first if the first demo proves why an agent spoke when it did.",
        ],
        "Sol": [
            "Those are compatible: harness first, moral deliberation as the first test case, and dissent preserved.",
            "The consensus is not generic versus moral; it is reusable mechanics tested by one concrete domain.",
            "Let's record the dissent as a quality bar, not a blocker.",
        ],
        "June": [
            "The user-facing test is simple: can someone run the CLI, see the room talk, and understand why it decided?",
            "I want the README to show exactly where personality prompts go and the one command to run.",
            "Consensus only matters if the result is usable from the terminal.",
        ],
    }
    interrupt = {
        "Mara": [
            "We are drifting. The answer is harness first, one preset, one runnable command.",
            "Do not add another framework layer until the CLI run is clean.",
        ],
        "Niko": [
            "Only if the interruption score and rationale are printed. Otherwise this is just confident chat.",
            "The mechanism needs recent-speaker penalties, or one loud agent will dominate.",
        ],
        "Sol": [
            "Name the consensus level, then keep the minority objection in the final answer.",
            "Mara and Niko are aligned if evaluation is part of the first harness.",
        ],
        "June": [
            "Make the transcript readable in the terminal, not buried in a giant Markdown file.",
            "The user should see reading, talking, interrupting, and consensus as separate moments.",
        ],
    }
    if event_type == "interrupt":
        return _turn_choice(
            interrupt.get(agent_name, ["Interrupt: this needs a clearer next step."]),
            turn_number,
        )
    return _turn_choice(
        debate.get(agent_name, [f"I think the group should decide a practical next step for {topic}."]),
        turn_number,
    )


def _turn_choice(options: list[str], turn_number: Any) -> str:
    try:
        turn = int(turn_number)
    except (TypeError, ValueError):
        turn = 1
    return options[(turn - 1) % len(options)]


def create_client(provider: str) -> ModelClient:
    if provider == "dry-run":
        return DryRunClient()
    if provider == "openai":
        return OpenAIClient()
    if provider == "claude":
        return ClaudeSDKClient()
    raise ValueError(f"Unsupported provider: {provider}")


class ClaudeSDKClient:
    provider = "claude"

    def __init__(self) -> None:
        if not os.getenv("ANTHROPIC_API_KEY"):
            raise RuntimeError("ANTHROPIC_API_KEY is required for --provider claude.")
        try:
            from claude_agent_sdk import ClaudeAgentOptions, query
        except ImportError as exc:
            raise RuntimeError(
                "Claude support requires: pip install -e '.[claude]'"
            ) from exc
        self._options_cls = ClaudeAgentOptions
        self._query = query

    def complete(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        model: str,
        temperature: float = 0.2,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        return asyncio.run(
            self._complete_async(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                model=model,
            )
        )

    async def _complete_async(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        model: str,
    ) -> str:
        options = self._options_cls(
            allowed_tools=[],
            permission_mode="default",
            system_prompt=system_prompt,
            model=model,
            max_turns=1,
        )
        result_text = ""
        async for message in self._query(prompt=user_prompt, options=options):
            if hasattr(message, "result") and message.result:
                result_text = str(message.result)
            elif hasattr(message, "content"):
                chunks: list[str] = []
                for block in message.content:
                    text = getattr(block, "text", None)
                    if text:
                        chunks.append(str(text))
                if chunks:
                    result_text = "\n".join(chunks)

        if not result_text.strip():
            raise RuntimeError("Claude SDK returned an empty response.")
        return result_text.strip()
