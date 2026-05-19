import json
import tempfile
import unittest
from pathlib import Path

from simula_crew.clients import DryRunClient
from simula_crew.engine import run_crew
from simula_crew.io import load_config, save_result
from simula_crew.prompts import PromptRenderError, format_transcript, render_template
from simula_crew.runtime import apply_runtime_inputs, parse_variable_assignments
from simula_crew.scoring import (
    DeterministicInterruptionClassifier,
    deterministic_interruption_score,
)
from simula_crew.schema import ConfigError, CrewConfig, Statement


REPO_ROOT = Path(__file__).resolve().parents[1]


class ConfigTests(unittest.TestCase):
    def test_simulacra_config_loads(self) -> None:
        config = load_config(REPO_ROOT / "configs" / "simulacra.json")
        self.assertIsInstance(config, CrewConfig)
        self.assertEqual(config.name, "simulacra")
        self.assertGreaterEqual(len(config.agents), 4)

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


class EngineTests(unittest.TestCase):
    def test_dry_run_executes_interruption_round(self) -> None:
        config = load_config(REPO_ROOT / "configs" / "simulacra.json")
        result = run_crew(
            config=config,
            client=DryRunClient(),
            model="dry-run-model",
            max_agents=2,
        )
        self.assertEqual(result.config_name, "simulacra")
        interruption_round = next(
            round_result
            for round_result in result.rounds
            if round_result.id == "interruption_window"
        )
        self.assertEqual(len(interruption_round.statements), 6)
        self.assertTrue(
            any(
                statement.metadata.get("classifier_interruption_score") is not None
                for statement in interruption_round.statements
            )
        )

        with tempfile.TemporaryDirectory() as output_dir:
            json_path, markdown_path = save_result(result, output_dir)
            self.assertTrue(json_path.exists())
            self.assertTrue(markdown_path.exists())
            payload = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["provider"], "dry-run")

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
                round_spec=config.rounds[2],
                transcript=[Statement("x", "X", "r", 1, "debate", "claim")],
                turn_number=2,
            )
            .score,
            0,
        )


if __name__ == "__main__":
    unittest.main()
