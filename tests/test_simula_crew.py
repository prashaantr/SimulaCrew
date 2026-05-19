import json
import tempfile
import unittest
from pathlib import Path

from simula_crew.clients import DryRunClient
from simula_crew.cli import _default_model
from simula_crew.engine import run_crew
from simula_crew.io import load_config, format_conversation, save_result
from simula_crew.prompts import PromptRenderError, format_transcript, render_template
from simula_crew.runtime import apply_runtime_inputs, parse_variable_assignments
from simula_crew.scoring import (
    DeterministicInterruptionClassifier,
    deterministic_interruption_score,
)
from simula_crew.schema import ConfigError, CrewConfig, Statement


REPO_ROOT = Path(__file__).resolve().parents[1]


class PromptRecordingClient:
    provider = "dry-run"

    def __init__(self) -> None:
        self.calls = []

    def complete(self, *, system_prompt, user_prompt, model, temperature=0.2, metadata=None):
        metadata = metadata or {}
        self.calls.append({"metadata": metadata, "user_prompt": user_prompt})
        event_type = metadata.get("event_type")
        agent_id = metadata.get("agent_id", "recorder")
        if event_type == "private":
            return f"private-secret-{agent_id}"
        if event_type == "synthesis":
            return "# PRD\n\nThe group chose one buildable idea."
        if event_type == "thought":
            return f"private-note-{agent_id}"
        return f"public-message-{agent_id}"


class ConfigTests(unittest.TestCase):
    def test_simulacra_config_loads(self) -> None:
        config = load_config(REPO_ROOT / "configs" / "simulacra.json")
        self.assertIsInstance(config, CrewConfig)
        self.assertEqual(config.name, "simulacra")
        self.assertGreaterEqual(len(config.agents), 4)
        self.assertTrue(config.agents[0].skills)
        self.assertTrue(config.agents[0].interests)
        self.assertTrue(config.agents[0].history)

    def test_duplicate_agent_ids_are_rejected(self) -> None:
        bad = {
            "name": "bad",
            "topic": {"title": "T", "prompt": "P"},
            "agents": [
                {"id": "a", "base_prompt": "one"},
                {"id": "a", "base_prompt": "two"},
            ],
            "rounds": [{"id": "r", "mode": "private", "prompt": "P"}],
        }
        with self.assertRaisesRegex(ConfigError, "duplicate"):
            CrewConfig.from_dict(bad)


class PromptTests(unittest.TestCase):
    def test_unknown_prompt_variable_fails_loudly(self) -> None:
        with self.assertRaisesRegex(PromptRenderError, "unknown variable"):
            render_template("{missing}", {"known": "value"})

    def test_anonymous_transcript_hides_agent_identity(self) -> None:
        transcript = [
            Statement("a", "Alex", "r", 1, "debate", "hello"),
        ]
        rendered = format_transcript(transcript, visibility="anonymous")
        self.assertIn("Agent 1", rendered)
        self.assertNotIn("Alex", rendered)


class RuntimeTests(unittest.TestCase):
    def test_runtime_prompt_overrides_topic(self) -> None:
        config = load_config(REPO_ROOT / "configs" / "simulacra.json")
        updated = apply_runtime_inputs(config, prompt="New challenge")
        self.assertEqual(updated.topic.prompt, "New challenge")
        self.assertEqual(updated.topic.variables["challenge_prompt"], "New challenge")

    def test_variable_parser(self) -> None:
        parsed = parse_variable_assignments(["target_user=Judges"])
        self.assertEqual(parsed["target_user"], "Judges")

    def test_bad_variable_parser(self) -> None:
        with self.assertRaisesRegex(ConfigError, "KEY=VALUE"):
            parse_variable_assignments(["bad"])

    def test_claude_defaults_to_haiku(self) -> None:
        self.assertEqual(_default_model("claude"), "haiku")


class EngineTests(unittest.TestCase):
    def test_dry_run_executes_interruption_round(self) -> None:
        config = load_config(REPO_ROOT / "configs" / "simulacra.json")
        events = []
        result = run_crew(
            config=config,
            client=DryRunClient(),
            model="dry-run-model",
            max_agents=4,
            event_callback=lambda event_type, payload: events.append((event_type, payload)),
        )
        self.assertEqual(result.config_name, "simulacra")
        group_round = next(
            round_result
            for round_result in result.rounds
            if round_result.id == "group_chat"
        )
        self.assertGreaterEqual(len(group_round.statements), 8)
        self.assertLessEqual(len(group_round.statements), group_round.max_turns or 999)
        self.assertTrue(
            any(
                statement.metadata.get("classifier_interruption_score") is not None
                for statement in group_round.statements
            )
        )
        self.assertTrue(
            any(statement.event_type == "interrupt" for statement in group_round.statements)
        )
        self.assertGreater(
            len({statement.agent_id for statement in group_round.statements}),
            1,
        )
        self.assertTrue(
            all("System prompt focus" not in statement.content for statement in group_round.statements)
        )
        self.assertTrue(
            all(
                "goal_alignment_by_agent" in statement.metadata
                for statement in group_round.statements
            )
        )
        synthesis_round = next(
            round_result
            for round_result in result.rounds
            if round_result.id == "final_synthesis"
        )
        self.assertIn("PRD", synthesis_round.statements[0].content)
        event_types = [event_type for event_type, _ in events]
        self.assertIn("round_start", event_types)
        self.assertIn("agent_start", event_types)
        self.assertIn("statement", event_types)
        self.assertIn("alignment_update", event_types)
        self.assertGreater(event_types.index("agent_start"), event_types.index("round_start"))

        with tempfile.TemporaryDirectory() as output_dir:
            json_path, conversation_path = save_result(result, output_dir)
            self.assertTrue(json_path.exists())
            self.assertTrue(conversation_path.exists())
            self.assertEqual(conversation_path.suffix, ".txt")
            payload = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["provider"], "dry-run")
            conversation = format_conversation(result)
            self.assertIn("SIMULACREW CONVERSATION", conversation)
            self.assertIn("[DISCUSSION] Group Chat", conversation)

    def test_deterministic_interruption_score_ranks_assertive_skeptic(self) -> None:
        config = load_config(REPO_ROOT / "configs" / "simulacra.json")
        mara, niko = config.agents[0], config.agents[1]
        self.assertGreater(
            deterministic_interruption_score(niko),
            deterministic_interruption_score(config.agents[2]),
        )
        self.assertGreater(
            DeterministicInterruptionClassifier()
            .score(
                agent=mara,
                config=config,
                round_spec=config.rounds[1],
                transcript=[Statement("x", "X", "r", 1, "debate", "claim")],
                turn_number=2,
            )
            .score,
            0,
        )

    def test_private_positions_inform_only_the_same_agent(self) -> None:
        config = load_config(REPO_ROOT / "configs" / "simulacra.json")
        client = PromptRecordingClient()
        run_crew(
            config=config,
            client=client,
            model="dry-run-model",
            max_agents=2,
        )
        group_calls = [
            call
            for call in client.calls
            if call["metadata"].get("round_id") == "group_chat"
            and call["metadata"].get("event_type") in {"debate", "interrupt"}
        ]
        self.assertTrue(group_calls)
        mara_call = next(
            call
            for call in group_calls
            if call["metadata"].get("agent_id") == "mara"
        )
        niko_call = next(
            call
            for call in group_calls
            if call["metadata"].get("agent_id") == "niko"
        )
        self.assertIn("private-secret-mara", mara_call["user_prompt"])
        self.assertNotIn("private-secret-niko", mara_call["user_prompt"])
        self.assertIn("private-secret-niko", niko_call["user_prompt"])
        self.assertNotIn("private-secret-mara", niko_call["user_prompt"])


if __name__ == "__main__":
    unittest.main()
