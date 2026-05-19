import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from simula_crew.clients import DryRunClient
from simula_crew.cli import _default_model, _load_document_dir, main
from simula_crew.engine import run_experiment
from simula_crew.google_drive import google_sheet_export_url
from simula_crew.ingest import (
    BUILDATHON_DEFAULT_TASK_PROMPT,
    DocumentText,
    agents_from_survey_rows,
    build_survey_experiment_bundle,
    load_survey_csv,
)
from simula_crew.io import format_conversation, load_experiment, save_experiment_result
from simula_crew.prompts import PromptRenderError, format_transcript, render_template
from simula_crew.runtime import apply_runtime_inputs, parse_variable_assignments
from simula_crew.scoring import (
    DeterministicInterruptionClassifier,
    deterministic_interruption_score,
)
from simula_crew.schema import (
    AgentPersona,
    ConfigError,
    ExperimentConfig,
    OutputFormat,
    Statement,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_PATH = REPO_ROOT / "configs" / "experiments" / "simulacra.yaml"


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
    def test_simulacra_experiment_loads(self) -> None:
        experiment = load_experiment(EXPERIMENT_PATH)
        self.assertIsInstance(experiment, ExperimentConfig)
        self.assertEqual(experiment.name, "simulacra")
        self.assertGreaterEqual(len(experiment.agents), 4)
        self.assertTrue(experiment.agents[0].skills)
        self.assertTrue(experiment.agents[0].interests)
        self.assertTrue(experiment.agents[0].history)
        self.assertEqual(experiment.task.output_format.type, "markdown")
        self.assertIn("product idea", experiment.task.output_format.required_sections)
        self.assertGreaterEqual(len(experiment.process.rounds), 3)

    def test_duplicate_agent_ids_are_rejected(self) -> None:
        with self.assertRaisesRegex(ConfigError, "duplicate"):
            ExperimentConfig.from_parts(
                name="bad",
                description="",
                task={"title": "T", "prompt": "P"},
                process={
                    "rounds": [
                        {"id": "r", "mode": "private", "prompt": "P"},
                    ],
                },
                agents=[
                    {"id": "a", "base_prompt": "one"},
                    {"id": "a", "base_prompt": "two"},
                ],
            )

    def test_legacy_one_file_configs_fail_with_migration_message(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "legacy.json"
            path.write_text(
                json.dumps(
                    {
                        "name": "legacy",
                        "topic": {"prompt": "Old topic shape"},
                        "harness": {},
                        "rounds": [],
                        "agents": [{"id": "a", "base_prompt": "old"}],
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ConfigError, "legacy one-file configs"):
                load_experiment(path)


class OutputFormatTests(unittest.TestCase):
    def test_markdown_required_sections(self) -> None:
        fmt = OutputFormat.from_dict({
            "type": "markdown",
            "description": "PRD",
            "required_sections": ["product idea", "risks"],
        })
        parsed, errors = fmt.validate("Product idea: X\nRisks: Y")
        self.assertEqual(errors, [])
        self.assertEqual(parsed, "Product idea: X\nRisks: Y")
        _, missing = fmt.validate("only mentions product idea")
        self.assertIn("Missing required section: risks", missing)

    def test_json_format_parses(self) -> None:
        fmt = OutputFormat.from_dict({"type": "json"})
        parsed, errors = fmt.validate('{"answer": 42}')
        self.assertEqual(errors, [])
        self.assertEqual(parsed, {"answer": 42})

    def test_number_format_extracts(self) -> None:
        fmt = OutputFormat.from_dict({"type": "number"})
        parsed, errors = fmt.validate("the answer is 42")
        self.assertEqual(errors, [])
        self.assertEqual(parsed, 42.0)
        _, errors = fmt.validate("no digits here")
        self.assertTrue(errors)


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
    def test_runtime_prompt_overrides_task(self) -> None:
        experiment = load_experiment(EXPERIMENT_PATH)
        updated = apply_runtime_inputs(experiment, prompt="New challenge")
        self.assertEqual(updated.task.prompt, "New challenge")
        self.assertEqual(updated.task.variables["challenge_prompt"], "New challenge")

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
        self.assertEqual(
            agent.personality["I enjoy being unique and different from others in many ways."],
            "Agree",
        )
        self.assertEqual(agent.personality["I am outgoing, sociable."], "Disagree")
        self.assertNotIn("openness", agent.personality)
        self.assertNotIn("conscientiousness", agent.personality)
        self.assertNotIn("extraversion", agent.personality)
        self.assertNotIn("agreeableness", agent.personality)
        self.assertNotIn("assertiveness", agent.personality)
        self.assertNotIn("skepticism", agent.personality)
        self.assertNotIn("neuroticism", agent.personality)
        self.assertNotIn("patience", agent.personality)
        self.assertNotIn("urgency", agent.personality)
        self.assertNotIn("interruptiveness", agent.personality)
        self.assertNotIn("disagreement_sensitivity", agent.personality)
        self.assertNotIn("goal_alignment_start", agent.personality)
        self.assertEqual(agent.speaking_style, "")
        self.assertEqual(agent.constraints, [])
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

    def test_build_survey_experiment_bundle_is_loadable(self) -> None:
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

        bundle = build_survey_experiment_bundle(
            agents,
            name="survey-team",
            task_prompt="Design an AI labor-market research prototype.",
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            (base / "task.json").write_text(json.dumps(bundle["task"]), encoding="utf-8")
            (base / "process.json").write_text(json.dumps(bundle["process"]), encoding="utf-8")
            (base / "agents.json").write_text(json.dumps(bundle["agents"]), encoding="utf-8")
            import yaml as _yaml
            (base / "experiment.yaml").write_text(_yaml.safe_dump(bundle["experiment"]), encoding="utf-8")
            experiment = load_experiment(base / "experiment.yaml")

        self.assertEqual(experiment.name, "survey-team")
        self.assertEqual(len(experiment.agents), 2)
        self.assertEqual(experiment.process.rounds[1].mode, "discussion")
        self.assertIn("survey-derived", experiment.process.character_prompt_template)
        self.assertIn("Survey questionnaire responses", experiment.process.character_prompt_template)

    def test_survey_experiment_default_task_is_general_buildathon_context(self) -> None:
        agents = agents_from_survey_rows([{"Name": "Ada Example"}])
        bundle = build_survey_experiment_bundle(agents)

        self.assertEqual(bundle["task"]["prompt"], BUILDATHON_DEFAULT_TASK_PROMPT)
        self.assertIn("Gates/Wharton Build-a-thon", bundle["task"]["prompt"])
        self.assertIn("Do not assume any known real team project", bundle["task"]["prompt"])
        self.assertNotIn("role", bundle["task"]["prompt"].lower())

    def test_survey_experiment_bundle_dry_runs(self) -> None:
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
        bundle = build_survey_experiment_bundle(
            agents,
            name="survey-team",
            task_prompt="Design an AI labor-market research prototype.",
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            (base / "task.json").write_text(json.dumps(bundle["task"]), encoding="utf-8")
            (base / "process.json").write_text(json.dumps(bundle["process"]), encoding="utf-8")
            (base / "agents.json").write_text(json.dumps(bundle["agents"]), encoding="utf-8")
            import yaml as _yaml

            (base / "experiment.yaml").write_text(_yaml.safe_dump(bundle["experiment"]), encoding="utf-8")
            experiment = load_experiment(base / "experiment.yaml")

        result = run_experiment(
            experiment=experiment,
            client=DryRunClient(),
            model="dry-run-model",
            max_agents=2,
        )

        self.assertEqual(result.experiment_name, "survey-team")
        self.assertEqual(result.format_check["valid"], True)

    def test_ingest_survey_cli_writes_loadable_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = Path(temp_dir) / "responses.csv"
            output_dir = Path(temp_dir) / "bundle"
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
                        "--output-dir",
                        str(output_dir),
                        "--task-prompt",
                        "Design an AI labor-market research prototype.",
                        "--quiet",
                    ]
                )

            self.assertEqual(exit_code, 0)
            self.assertTrue((output_dir / "experiment.yaml").exists())
            experiment = load_experiment(output_dir / "experiment.yaml")
            self.assertEqual(experiment.agents[0].id, "ada-example")
            self.assertEqual(experiment.task.prompt, "Design an AI labor-market research prototype.")

    def test_document_dir_loads_top_level_and_nested_text_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "documents"
            (root / "ada-example").mkdir(parents=True)
            (root / "grace-example").mkdir()
            (root / "ada-example.txt").write_text("Ada top-level note.", encoding="utf-8")
            (root / "ada-example" / "resume.md").write_text("Ada nested resume.", encoding="utf-8")
            (root / "grace-example" / "profile.csv").write_text("field,value\nskill,Product\n", encoding="utf-8")
            (root / "grace-example" / "ignore.pdf").write_text("not extracted yet", encoding="utf-8")

            documents = _load_document_dir(str(root))

        self.assertEqual([doc.source for doc in documents["ada-example"]], ["ada-example/resume.md", "ada-example.txt"])
        self.assertEqual(documents["ada-example"][0].content, "Ada nested resume.")
        self.assertEqual([doc.source for doc in documents["grace-example"]], ["grace-example/profile.csv"])

    def test_ingest_survey_cli_uses_normalized_document_dir(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            csv_path = base / "responses.csv"
            output_dir = base / "bundle"
            document_dir = base / "documents"
            (document_dir / "ada-example").mkdir(parents=True)
            csv_path.write_text("Name,What is your occupation?\nAda Example,Economist\n", encoding="utf-8")
            (document_dir / "ada-example" / "resume.txt").write_text(
                "Ada works on labor market AI adoption.",
                encoding="utf-8",
            )

            with redirect_stdout(io.StringIO()):
                exit_code = main(
                    [
                        "ingest-survey",
                        str(csv_path),
                        "--output-dir",
                        str(output_dir),
                        "--document-dir",
                        str(document_dir),
                        "--quiet",
                    ]
                )

            self.assertEqual(exit_code, 0)
            agents = json.loads((output_dir / "agents.json").read_text(encoding="utf-8"))["agents"]
            self.assertEqual(agents[0]["history"][0], "ada-example/resume.txt: Ada works on labor market AI adoption.")

    def test_google_sheet_export_url_preserves_gid(self) -> None:
        url = google_sheet_export_url(
            "https://docs.google.com/spreadsheets/d/abc123/edit?gid=811813731#gid=811813731"
        )

        self.assertEqual(
            url,
            "https://docs.google.com/spreadsheets/d/abc123/export?format=csv&gid=811813731",
        )


class ChatCliTests(unittest.TestCase):
    def test_chat_message_runs_one_prompt_and_writes_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                exit_code = main(
                    [
                        "chat",
                        str(EXPERIMENT_PATH),
                        "--message",
                        "What should we build for a future-of-work hackathon?",
                        "--max-agents",
                        "2",
                        "--output-dir",
                        temp_dir,
                        "--quiet",
                    ]
                )

            self.assertEqual(exit_code, 0)
            paths = [Path(line) for line in stdout.getvalue().splitlines() if line.strip()]
            self.assertEqual(len(paths), 2)
            self.assertTrue(paths[0].exists())
            self.assertTrue(paths[1].exists())
            payload = json.loads(paths[0].read_text(encoding="utf-8"))
            self.assertEqual(payload["task_prompt"], "What should we build for a future-of-work hackathon?")
            self.assertTrue(payload["metadata"]["chat_interface"])

    def test_chat_live_output_starts_with_ascii_and_spaces_state_blocks(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                exit_code = main(
                    [
                        "chat",
                        str(EXPERIMENT_PATH),
                        "--message",
                        "Pick one quick demo idea.",
                        "--max-agents",
                        "2",
                        "--output-dir",
                        temp_dir,
                    ]
                )

            output = stdout.getvalue()
            self.assertEqual(exit_code, 0)
            self.assertIn("███████╗██╗", output)
            self.assertIn("SimulaCrew Chat", output)
            self.assertIn("room state", output)
            self.assertLess(output.count("room state"), output.count("message"))


class EngineTests(unittest.TestCase):
    def test_dry_run_executes_full_experiment(self) -> None:
        experiment = load_experiment(EXPERIMENT_PATH)
        events = []
        result = run_experiment(
            experiment=experiment,
            client=DryRunClient(),
            model="dry-run-model",
            max_agents=4,
            event_callback=lambda event_type, payload: events.append((event_type, payload)),
        )
        self.assertEqual(result.experiment_name, "simulacra")
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
        # New schema: transcript & thinking & task_result & format_check.
        self.assertTrue(
            all(statement.event_type not in {"private", "thought"} for statement in result.transcript)
        )
        self.assertTrue(result.thinking)
        for agent_id, notes in result.thinking.items():
            self.assertIsInstance(notes, list)
            self.assertTrue(all(isinstance(note, str) for note in notes))
        self.assertIsInstance(result.task_result, str)
        self.assertEqual(result.format_check["format_type"], "markdown")
        self.assertEqual(result.format_check["valid"], True)

        event_types = [event_type for event_type, _ in events]
        self.assertIn("round_start", event_types)
        self.assertIn("agent_start", event_types)
        self.assertIn("statement", event_types)
        self.assertIn("alignment_update", event_types)

        with tempfile.TemporaryDirectory() as output_dir:
            json_path, conversation_path = save_experiment_result(result, output_dir)
            self.assertTrue(json_path.exists())
            self.assertTrue(conversation_path.exists())
            payload = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["provider"], "dry-run")
            self.assertIn("transcript", payload)
            self.assertIn("thinking", payload)
            self.assertIn("task_result", payload)
            self.assertIn("format_check", payload)
            conversation = format_conversation(result)
            self.assertIn("SIMULACREW CONVERSATION", conversation)
            self.assertIn("[DISCUSSION] Group Chat", conversation)

    def test_deterministic_interruption_score_ranks_assertive_skeptic(self) -> None:
        experiment = load_experiment(EXPERIMENT_PATH)
        mara, niko = experiment.agents[0], experiment.agents[1]
        self.assertGreater(
            deterministic_interruption_score(niko),
            deterministic_interruption_score(experiment.agents[2]),
        )
        self.assertGreater(
            DeterministicInterruptionClassifier()
            .score(
                agent=mara,
                experiment=experiment,
                round_spec=experiment.process.rounds[1],
                transcript=[Statement("x", "X", "r", 1, "debate", "claim")],
                turn_number=2,
            )
            .score,
            0,
        )

    def test_private_positions_inform_only_the_same_agent(self) -> None:
        experiment = load_experiment(EXPERIMENT_PATH)
        client = PromptRecordingClient()
        run_experiment(
            experiment=experiment,
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

    def test_thinking_field_collects_private_thoughts(self) -> None:
        experiment = load_experiment(EXPERIMENT_PATH)
        client = PromptRecordingClient()
        result = run_experiment(
            experiment=experiment,
            client=client,
            model="dry-run-model",
            max_agents=2,
        )
        # Each participant should have at least one private thought (independent
        # positions round).
        self.assertIn("mara", result.thinking)
        self.assertIn("niko", result.thinking)
        for notes in result.thinking.values():
            self.assertTrue(any(note.startswith("private-secret-") or note.startswith("private-note-") for note in notes))

    def test_hidden_idea_state_is_not_shown_to_agent_prompts(self) -> None:
        experiment = load_experiment(EXPERIMENT_PATH)
        client = PromptRecordingClient()
        run_experiment(
            experiment=experiment,
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
        experiment = load_experiment(EXPERIMENT_PATH)
        client = AgentStateClient()
        result = run_experiment(
            experiment=experiment,
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
