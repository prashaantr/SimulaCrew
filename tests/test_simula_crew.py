import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from simula_crew.clients import DryRunClient
from simula_crew.cli import _default_model, main
from simula_crew.engine import run_crew
from simula_crew.google_drive import google_sheet_export_url
from simula_crew.ingest import (
    DocumentText,
    agents_from_survey_rows,
    build_survey_crew_config,
    load_survey_csv,
)
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


class AgentStateClient:
    provider = "claude"

    def __init__(self) -> None:
        self.calls = []

    def complete(self, *, system_prompt, user_prompt, model, temperature=0.2, metadata=None):
        metadata = metadata or {}
        self.calls.append({"metadata": metadata, "model": model, "user_prompt": user_prompt})
        event_type = metadata.get("event_type")
        agent_id = metadata.get("agent_id", "recorder")
        if event_type == "idea_state_evaluation":
            buy_in = 0.88 if agent_id == "mara" else 0.42
            return json.dumps(
                {
                    "idea": f"{agent_id} concept",
                    "buy_in": buy_in,
                    "rationale": f"{agent_id} updated their private idea state.",
                }
            )
        if event_type == "private":
            return f"private-secret-{agent_id}"
        if event_type == "synthesis":
            return "# PRD\n\nThe group chose one buildable idea."
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


class SurveyIngestionTests(unittest.TestCase):
    def test_survey_row_builds_agent_with_demographics_and_document_evidence(self) -> None:
        rows = [
            {
                "Please tell us the name you use in the Hackpad (we want to match you to your real team!)": "Ada Example",
                "Which part of the United States do you currently live in?": "Northeast",
                "What is the sex that you were assigned at birth?": "Female",
                "How old are you?": "30-49",
                "What is your race or origin?": "Asian",
                "Which best describes your primary role?": "Faculty",
                "What is your occupation?": "Economist",
                "In this occupation, what kind of work do you do and what are the most important activities or duties?": "Research, teaching, and advising.",
                "In a few sentences, describe your educational and occupational background. Do you have a disciplinary affiliation or approach (e.g., economics, computer science)?": "Labor economics and computer science.",
                "In a few sentences, what draws you to the question of how AI is reshaping jobs and the economy?": "I care about broad access to AI benefits.",
                "How would you describe the skill set that you contribute to the team?": "Causal inference, field experiments",
                "How would you describe the data sources you are most excited to work with?": "Job postings and worker accounts",
                "I enjoy being unique and different from others in many ways.": "Agree",
                "I often do “my own thing.”": "Strongly Agree",
                "I feel good when I cooperate with others.": "Agree",
                "I prefer to work without instructions from others": "Agree",
                "I am outgoing, sociable.": "Disagree",
                "I tend to find fault with others.": "Agree",
                "I do a thorough job.": "Strongly Agree",
                "I get nervous easily.": "Disagree",
                "I have an active imagination.": "Agree",
                "I really enjoy a task that involves coming up with new solutions to problems": "Strongly Agree",
                "What do you value the most in your life?": "Family and useful work",
                "Imagine yourself a few years from now. Maybe you want your life to be the same in some ways as it is now. Maybe you want it to be different in some ways. What do you hope for?": "More time for important research.",
            }
        ]
        documents = {
            "ada-example": [
                DocumentText(
                    source="resume.txt",
                    content="Ada studies labor markets, AI adoption, and field experiments.",
                )
            ]
        }

        agents = agents_from_survey_rows(rows, documents_by_person=documents)

        self.assertEqual(len(agents), 1)
        agent = agents[0]
        self.assertEqual(agent.id, "ada-example")
        self.assertEqual(agent.name, "Ada Example")
        self.assertIn("Female", agent.backstory)
        self.assertIn("Labor economics", agent.backstory)
        self.assertIn("Causal inference", agent.skills)
        self.assertIn("Job postings and worker accounts", agent.interests)
        self.assertIn("resume.txt", agent.history[0])
        self.assertGreater(agent.personality["openness"], 5)
        self.assertGreater(agent.personality["conscientiousness"], 5)
        self.assertLess(agent.personality["extraversion"], 6)
        self.assertIn("broad access", agent.goals[0])

    def test_survey_csv_loader_skips_empty_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "responses.csv"
            path.write_text(
                "Name,Occupation\n"
                "Ada Example,Economist\n"
                ",\n",
                encoding="utf-8",
            )

            rows = load_survey_csv(path)

        self.assertEqual(rows, [{"Name": "Ada Example", "Occupation": "Economist"}])

    def test_build_survey_crew_config_loads_as_existing_config(self) -> None:
        agents = agents_from_survey_rows(
            [
                {
                    "Name": "Ada Example",
                    "Occupation": "Economist",
                    "How would you describe the skill set that you contribute to the team?": "Research design",
                },
                {
                    "Name": "Grace Example",
                    "Occupation": "Product lead",
                    "How would you describe the skill set that you contribute to the team?": "Product strategy",
                },
            ]
        )

        payload = build_survey_crew_config(
            agents,
            name="survey_team",
            topic_prompt="Design an AI labor-market research prototype.",
        )
        config = CrewConfig.from_dict(payload)

        self.assertEqual(config.name, "survey_team")
        self.assertEqual(len(config.agents), 2)
        self.assertEqual(config.rounds[1].mode, "discussion")
        self.assertIn("survey-derived", config.harness.character_prompt_template)

    def test_ingest_survey_cli_writes_loadable_config(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = Path(temp_dir) / "responses.csv"
            output_path = Path(temp_dir) / "team.json"
            csv_path.write_text(
                "Name,What is your occupation?,How would you describe the skill set that you contribute to the team?\n"
                "Ada Example,Economist,Research design\n",
                encoding="utf-8",
            )

            with redirect_stdout(io.StringIO()):
                exit_code = main(
                    [
                        "ingest-survey",
                        str(csv_path),
                        "--output",
                        str(output_path),
                        "--topic",
                        "Design an AI labor-market research prototype.",
                        "--quiet",
                    ]
                )

            self.assertEqual(exit_code, 0)
            self.assertTrue(output_path.exists())
            config = load_config(output_path)
            self.assertEqual(config.agents[0].id, "ada-example")
            self.assertEqual(config.topic.prompt, "Design an AI labor-market research prototype.")

    def test_google_sheet_export_url_preserves_gid(self) -> None:
        url = google_sheet_export_url(
            "https://docs.google.com/spreadsheets/d/abc123/edit?gid=811813731#gid=811813731"
        )

        self.assertEqual(
            url,
            "https://docs.google.com/spreadsheets/d/abc123/export?format=csv&gid=811813731",
        )


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

    def test_hidden_idea_state_is_not_shown_to_agent_prompts(self) -> None:
        config = load_config(REPO_ROOT / "configs" / "simulacra.json")
        client = PromptRecordingClient()
        run_crew(
            config=config,
            client=client,
            model="dry-run-model",
            max_agents=2,
        )
        agent_visible_calls = [
            call
            for call in client.calls
            if call["metadata"].get("event_type")
            in {"private", "debate", "interrupt", "thought", "synthesis"}
        ]
        self.assertTrue(agent_visible_calls)
        forbidden_fragments = [
            "Current idea the room may be converging on",
            "Your private understanding of that idea",
            "Your private current buy-in",
            "Private group buy-in snapshot",
            "Final shared idea",
            "Final buy-in state",
            "Each agent's idea view",
            "No idea has been named yet.",
            "avg=",
        ]
        for call in agent_visible_calls:
            prompt = call["user_prompt"]
            for fragment in forbidden_fragments:
                self.assertNotIn(fragment, prompt)

    def test_claude_idea_state_updates_are_per_agent_and_haiku(self) -> None:
        config = load_config(REPO_ROOT / "configs" / "simulacra.json")
        client = AgentStateClient()
        result = run_crew(
            config=config,
            client=client,
            model="opus",
            max_agents=2,
        )
        state_calls = [
            call
            for call in client.calls
            if call["metadata"].get("event_type") == "idea_state_evaluation"
        ]
        self.assertTrue(state_calls)
        self.assertTrue(all(call["model"] == "haiku" for call in state_calls))
        self.assertGreaterEqual(
            len({call["metadata"].get("agent_id") for call in state_calls}),
            2,
        )
        group_round = next(
            round_result
            for round_result in result.rounds
            if round_result.id == "group_chat"
        )
        first_public = next(
            statement
            for statement in group_round.statements
            if statement.event_type != "thought"
        )
        self.assertEqual(
            first_public.metadata["agent_idea_views"]["mara"],
            "mara concept",
        )
        self.assertEqual(
            first_public.metadata["agent_idea_views"]["niko"],
            "niko concept",
        )


if __name__ == "__main__":
    unittest.main()
