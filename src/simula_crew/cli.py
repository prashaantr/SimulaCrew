from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from textwrap import dedent, wrap as wrap_text

from simula_crew.clients import create_client
from simula_crew.engine import run_experiment
from simula_crew.google_drive import load_google_sheet_rows
from simula_crew.ingest import (
    DocumentText,
    agents_from_survey_rows,
    build_survey_experiment_bundle,
    load_survey_csv,
)
from simula_crew.io import list_experiments, load_experiment, save_experiment_result
from simula_crew.runtime import apply_runtime_inputs, parse_variable_assignments
from simula_crew.scoring import LLMInterruptionClassifier
from simula_crew.schema import ConfigError, ExperimentResult
from simula_crew.terminal import (
    Style,
    banner,
    card,
    color,
    key_value,
    logo,
    panel,
    section,
    tree,
    wrap,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="simulacrew",
        description="Run personality-driven agent crews that debate, interrupt, and synthesize.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=dedent(
            """
            examples:
              simulacrew list
              simulacrew inspect configs/experiments/simulacra.yaml
              simulacrew run configs/experiments/simulacra.yaml
              simulacrew run configs/experiments/simulacra.yaml --prompt "What should the crew decide?"
              simulacrew run configs/experiments/simulacra.yaml --provider claude --interruption-classifier llm --prompt-file challenge.txt
"""
        ),
    )
    subparsers = parser.add_subparsers(dest="command")

    list_parser = subparsers.add_parser("list", help="List available experiments.")
    list_parser.add_argument("--experiments-dir", default="configs/experiments")

    inspect_parser = subparsers.add_parser("inspect", help="Show experiment details.")
    inspect_parser.add_argument("experiment")

    ingest_parser = subparsers.add_parser(
        "ingest-survey",
        help="Generate a SimulaCrew experiment bundle from a survey CSV export.",
    )
    ingest_parser.add_argument("survey_csv")
    ingest_parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where the experiment bundle (experiment + task + process + agents) is written.",
    )
    ingest_parser.add_argument(
        "--task-prompt",
        default="Deliberate as a team and produce the best final artifact for the task.",
    )
    ingest_parser.add_argument("--name", default="survey-team")
    ingest_parser.add_argument(
        "--document-text-dir",
        default=None,
        help="Optional directory of .txt files named by person id, for example ada-example.txt.",
    )
    ingest_parser.add_argument(
        "--quiet",
        action="store_true",
        help="Only print the output path.",
    )

    google_ingest_parser = subparsers.add_parser(
        "ingest-google-survey",
        help="Generate a SimulaCrew experiment bundle from a Google Sheets survey using Google auth.",
    )
    google_ingest_parser.add_argument("sheet_url")
    google_ingest_parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where the experiment bundle is written.",
    )
    google_ingest_parser.add_argument(
        "--credentials-file",
        default=None,
        help="Optional service-account JSON file. If omitted, application default credentials are used.",
    )
    google_ingest_parser.add_argument(
        "--task-prompt",
        default="Deliberate as a team and produce the best final artifact for the task.",
    )
    google_ingest_parser.add_argument("--name", default="survey-team")
    google_ingest_parser.add_argument(
        "--document-text-dir",
        default=None,
        help="Optional directory of .txt files named by person id, for example ada-example.txt.",
    )
    google_ingest_parser.add_argument(
        "--quiet",
        action="store_true",
        help="Only print the output path.",
    )

    run_parser = subparsers.add_parser("run", help="Run an experiment.")
    run_parser.add_argument("experiment")
    run_parser.add_argument("--prompt", default=None)
    run_parser.add_argument("--prompt-file", default=None)
    run_parser.add_argument("--var", action="append", default=None, metavar="KEY=VALUE")
    run_parser.add_argument(
        "--provider",
        choices=["dry-run", "openai", "claude"],
        default="dry-run",
    )
    run_parser.add_argument("--model", default=None)
    run_parser.add_argument(
        "--interruption-classifier",
        choices=["deterministic", "llm"],
        default="deterministic",
        help="Use deterministic personality scoring or an LLM classifier for interruption decisions.",
    )
    run_parser.add_argument(
        "--classifier-model",
        default=None,
        help="Model for LLM interruption classification. Defaults to --model.",
    )
    run_parser.add_argument("--temperature", type=float, default=0.2)
    run_parser.add_argument("--max-agents", type=int, default=None)
    run_parser.add_argument("--output-dir", default="runs")
    run_parser.add_argument(
        "--quiet",
        action="store_true",
        help="Only print output file paths.",
    )
    run_parser.add_argument(
        "--no-live",
        action="store_true",
        help="Disable live per-agent output and print a summary at the end.",
    )
    run_parser.add_argument(
        "--show-interruption-notes",
        action="store_true",
        help="Show internal interruption scores and classifier rationale in the live transcript.",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "list":
        return _list_experiments(args.experiments_dir)
    if args.command == "inspect":
        return _inspect(args.experiment)
    if args.command == "ingest-survey":
        return _ingest_survey(args, parser)
    if args.command == "ingest-google-survey":
        return _ingest_google_survey(args, parser)
    if args.command == "run":
        return _run(args, parser)

    parser.print_help()
    return 0


def _ingest_survey(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    try:
        documents = _load_document_text_dir(args.document_text_dir)
        agents = agents_from_survey_rows(
            load_survey_csv(args.survey_csv),
            documents_by_person=documents,
        )
        if not agents:
            parser.error("survey CSV did not contain any usable response rows")
        written = _write_survey_bundle(
            agents=agents,
            name=args.name,
            task_prompt=args.task_prompt,
            output_dir=Path(args.output_dir),
        )
    except OSError as exc:
        parser.exit(2, f"error: {exc}\n")

    if args.quiet:
        print(written["experiment"])
        return 0

    print(section("Survey Ingestion"))
    print(key_value("Agents", str(len(agents))))
    for label, path in written.items():
        print(key_value(label.capitalize(), str(path)))
    return 0


def _ingest_google_survey(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    try:
        documents = _load_document_text_dir(args.document_text_dir)
        agents = agents_from_survey_rows(
            load_google_sheet_rows(
                args.sheet_url,
                credentials_file=args.credentials_file,
            ),
            documents_by_person=documents,
        )
        if not agents:
            parser.error("Google Sheet did not contain any usable response rows")
        written = _write_survey_bundle(
            agents=agents,
            name=args.name,
            task_prompt=args.task_prompt,
            output_dir=Path(args.output_dir),
        )
    except (OSError, RuntimeError, ValueError) as exc:
        parser.exit(2, f"error: {exc}\n")

    if args.quiet:
        print(written["experiment"])
        return 0

    print(section("Google Survey Ingestion"))
    print(key_value("Agents", str(len(agents))))
    for label, path in written.items():
        print(key_value(label.capitalize(), str(path)))
    return 0


def _write_survey_bundle(
    *,
    agents,
    name: str,
    task_prompt: str,
    output_dir: Path,
) -> dict[str, Path]:
    import yaml as _yaml

    bundle = build_survey_experiment_bundle(
        agents,
        name=name,
        task_prompt=task_prompt,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for label in ("task", "process", "agents"):
        target = output_dir / f"{label}.json"
        target.write_text(json.dumps(bundle[label], indent=2) + "\n", encoding="utf-8")
        paths[label] = target
    experiment_path = output_dir / "experiment.yaml"
    experiment_path.write_text(_yaml.safe_dump(bundle["experiment"], sort_keys=False), encoding="utf-8")
    paths["experiment"] = experiment_path
    return paths


def _load_document_text_dir(path: str | None) -> dict[str, list[DocumentText]]:
    if not path:
        return {}
    root = Path(path)
    documents: dict[str, list[DocumentText]] = {}
    for document_path in sorted(root.glob("*.txt")):
        documents.setdefault(document_path.stem, []).append(
            DocumentText(
                source=document_path.name,
                content=document_path.read_text(encoding="utf-8"),
            )
        )
    return documents


def _list_experiments(experiments_dir: str) -> int:
    print(logo())
    print(banner("Agent Debate CLI", "personality-driven crews with interruption mechanics"))
    experiments = list_experiments(experiments_dir)
    if not experiments:
        print("No experiments found.")
        return 0
    print(section("Experiments"))
    for experiment_path in experiments:
        experiment = load_experiment(experiment_path)
        print(panel(experiment.name, f"{experiment.description}\npath: {experiment_path}", width=88))
    return 0


def _inspect(experiment_path: str) -> int:
    experiment = load_experiment(experiment_path)
    print(logo())
    print(banner(experiment.name, experiment.description))
    print(
        card(
            "Experiment",
            [
                ("file", experiment_path),
                ("agents", str(len(experiment.agents))),
                ("rounds", str(len(experiment.process.rounds))),
                ("task", experiment.task.title),
                ("output_format", experiment.task.output_format.type),
            ],
            width=88,
        )
    )
    print(section("Task"))
    print(wrap(experiment.task.prompt))
    if experiment.task.output_format.description:
        print(section("Expected output"))
        print(wrap(experiment.task.output_format.description))
    print(section("Agents"))
    for agent in experiment.agents:
        print(panel(agent.name, agent.base_prompt, width=88))
        if agent.personality:
            traits = ", ".join(f"{key}={value}" for key, value in sorted(agent.personality.items()))
            print("  " + color(traits, Style.DIM))
    print(section("Process"))
    print(
        tree(
            [
                (
                    round_spec.title,
                    f"[{round_spec.mode}, max_turns={round_spec.max_turns or 'n/a'}]",
                )
                for round_spec in experiment.process.rounds
            ]
        )
    )
    return 0


def _run(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    if args.max_agents is not None and args.max_agents <= 0:
        parser.error("--max-agents must be greater than zero")
    if args.temperature < 0 or args.temperature > 2:
        parser.error("--temperature must be between 0 and 2")

    try:
        experiment = apply_runtime_inputs(
            load_experiment(args.experiment),
            prompt=args.prompt,
            prompt_file=args.prompt_file,
            variables=parse_variable_assignments(args.var),
        )
        client = create_client(args.provider)
        model = args.model or _default_model(args.provider)
        interruption_classifier = None
        if args.interruption_classifier == "llm":
            interruption_classifier = LLMInterruptionClassifier(
                client=client,
                model=args.classifier_model or model,
            )
        event_callback = None
        if not args.quiet and not args.no_live:
            _print_run_header(experiment.name, experiment.task.title, args.provider, model)
            event_callback = _live_event_printer(
                show_interruption_notes=args.show_interruption_notes,
            )
        result = run_experiment(
            experiment=experiment,
            client=client,
            model=model,
            temperature=args.temperature,
            max_agents=args.max_agents,
            interruption_classifier=interruption_classifier,
            event_callback=event_callback,
            run_metadata={
                "experiment_path": str(Path(args.experiment)),
                "output_dir": str(Path(args.output_dir)),
                "interruption_classifier": args.interruption_classifier,
            },
        )
        json_path, conversation_path = save_experiment_result(result, args.output_dir)
    except (ConfigError, RuntimeError, ValueError) as exc:
        parser.exit(2, f"error: {exc}\n")

    if args.quiet:
        print(json_path)
        print(conversation_path)
        return 0

    if args.no_live:
        _print_run_summary(result)
    print(section("Artifacts"))
    print(key_value("JSON", str(json_path)))
    print(key_value("Conversation", str(conversation_path)))
    return 0


def _default_model(provider: str) -> str:
    if provider == "claude":
        return "haiku"
    if provider == "openai":
        return "gpt-4o-mini"
    return "dry-run-model"


def _print_run_header(
    config_name: str,
    topic_title: str,
    provider: str,
    model: str,
) -> None:
    print(logo(), flush=True)
    print(banner("SimulaCrew Live Run", topic_title), flush=True)
    print(
        card(
            "Session",
            [
                ("config", config_name),
                ("provider", provider),
                ("model", model),
                ("mode", "live per-agent output"),
            ],
            width=88,
        ),
        flush=True,
    )


def _live_event_printer(*, show_interruption_notes: bool = False):
    speaker_styles: dict[str, str] = {}
    indicator: TypingIndicator | None = None

    def handle(event_type: str, payload: dict) -> None:
        nonlocal indicator
        if event_type == "round_start":
            if indicator:
                indicator.stop()
                indicator = None
            title = payload["round_title"]
            mode = str(payload["round_mode"]).upper()
            print(_conversation_phase(mode, title), flush=True)
            return

        if event_type == "agent_start":
            style = _style_for_speaker(
                str(payload["agent_id"]),
                speaker_styles,
            )
            agent = payload["agent_name"]
            turn = payload["turn_number"]
            action = payload["event_type"]
            score = (
                payload.get("interruption_score")
                if action == "interrupt" and show_interruption_notes
                else None
            )
            label, verb = _action_label(action)
            score_text = _score_text(score)
            message = f"{agent} {verb} (turn {turn}){score_text}"
            if action == "private":
                print(color(f"░ {message}", style + Style.DIM), flush=True)
            elif sys.stdout.isatty():
                if indicator:
                    indicator.stop()
                indicator = TypingIndicator(message)
                indicator.start()
            rationale = payload.get("interruption_rationale")
            if rationale and show_interruption_notes:
                if indicator:
                    indicator.stop()
                    indicator = None
                print(color(f"note: {rationale}", Style.DIM), flush=True)
            return

        if event_type == "statement":
            if indicator:
                indicator.stop()
                indicator = None
            statement = payload["statement"]
            style = _style_for_speaker(statement.agent_id, speaker_styles)
            print(
                _conversation_statement(
                    statement,
                    style,
                    show_interruption_notes=show_interruption_notes,
                ),
                flush=True,
            )
            return

        if event_type == "alignment_update":
            print(_alignment_status(payload, speaker_styles), flush=True)
            return

        if event_type == "goal_aligned":
            if indicator:
                indicator.stop()
                indicator = None
            print(color("Goal convergence reached. Moving to PRD.", Style.DIM), flush=True)
            return

        if event_type == "discussion_time_limit_reached":
            if indicator:
                indicator.stop()
                indicator = None
            print(color("Discussion time limit reached. Moving to PRD.", Style.DIM), flush=True)
            return

    return handle


def _conversation_phase(mode: str, title: str) -> str:
    phase = "GROUP CHAT" if mode == "DISCUSSION" else mode
    return "\n".join(
        [
            "",
            color(f"█ {phase} / {title}", Style.BOLD + Style.MAGENTA),
        ]
    )


def _conversation_statement(
    statement,
    speaker_style: str,
    *,
    show_interruption_notes: bool = False,
) -> str:
    label, _ = _action_label(statement.event_type)
    time_text = datetime.now().strftime("%H:%M:%S")
    score = (
        statement.metadata.get("interruption_score")
        if statement.event_type == "interrupt" and show_interruption_notes
        else None
    )
    score_text = _score_text(score)
    marker = "!" if statement.event_type == "interrupt" else ">"
    header = (
        color(f"{marker} {statement.agent_name}", speaker_style + Style.BOLD)
        + color(f"  {label.lower()}  {time_text}{score_text}", Style.DIM)
    )
    lines = [header]
    if statement.event_type == "thought":
        return "\n".join(lines) + "\n"
    for paragraph in statement.content.strip().splitlines() or [""]:
        wrapped = wrap_text(paragraph, width=82) if paragraph else [""]
        for line in wrapped:
            lines.append("  " + line)
    return "\n".join(lines) + "\n"


def _action_label(action: str) -> tuple[str, str]:
    if action == "private":
        return "thinking", "thinking"
    if action == "thought":
        return "stays quiet", "thinking"
    if action == "interrupt":
        return "cuts in", "cuts in"
    if action == "synthesis":
        return "wraps up", "wrapping up"
    return "says", "typing"


def _score_text(score) -> str:
    if score is None:
        return ""
    try:
        return f" | interrupt {float(score):.1f}/10"
    except (TypeError, ValueError):
        return f" | interrupt {score}/10"


def _alignment_status(payload: dict, speaker_styles: dict[str, str]) -> str:
    current_idea = str(payload.get("current_idea") or "No concrete idea yet.")
    by_agent = payload.get("by_agent") or {}
    views = payload.get("agent_idea_views") or {}
    lines = [
        "",
        _state_rule("room state"),
        color("  │ buy-in", Style.DIM + Style.BOLD),
    ]
    if isinstance(by_agent, dict) and by_agent:
        for agent_id, score in by_agent.items():
            if not _is_number(score):
                continue
            score_value = float(score)
            style = _style_for_speaker(str(agent_id), speaker_styles)
            lines.append(
                color("  │   ", Style.DIM)
                + color(f"{_display_agent_id(str(agent_id)):<8}", style + Style.BOLD)
                + f" {score_value:>4.0%}  "
                + color(_buy_in_bar(score_value), style)
            )
    else:
        lines.append(color("  │   " + str(payload.get("summary") or "no scores yet"), Style.DIM))

    lines.extend([color("  │", Style.DIM), color("  │ shared idea", Style.DIM + Style.BOLD)])
    lines.extend(_wrapped_state_text(current_idea, indent="  │   ", width=78))

    if isinstance(views, dict) and views:
        lines.extend([color("  │", Style.DIM), color("  │ agent idea views", Style.DIM + Style.BOLD)])
        for agent_id, view in views.items():
            style = _style_for_speaker(str(agent_id), speaker_styles)
            lines.append(
                color("  │   ", Style.DIM)
                + color(_display_agent_id(str(agent_id)), style + Style.BOLD)
            )
            lines.extend(_wrapped_state_text(str(view), indent="  │     ", width=74, dim=True))

    lines.append(_state_bottom())
    return "\n".join(lines) + "\n"


def _state_rule(title: str, width: int = 88) -> str:
    prefix = f"  ┌─ {title} "
    return color(prefix + "─" * max(8, width - len(prefix)), Style.DIM)


def _state_bottom(width: int = 88) -> str:
    return color("  └" + "─" * (width - 3), Style.DIM)


def _buy_in_bar(score: float, width: int = 12) -> str:
    filled = round(_clamp(score, 0.0, 1.0) * width)
    return "█" * filled + "░" * (width - filled)


def _wrapped_state_text(
    text: str,
    *,
    indent: str,
    width: int,
    dim: bool = False,
) -> list[str]:
    style = Style.DIM if dim else ""
    lines: list[str] = []
    for paragraph in text.strip().splitlines() or [""]:
        wrapped = wrap_text(paragraph, width=width) if paragraph else [""]
        for line in wrapped:
            lines.append(color(indent, Style.DIM) + color(line, style))
    return lines


def _display_agent_id(agent_id: str) -> str:
    return agent_id.replace("_", " ").title()


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return min(maximum, max(minimum, value))


def _is_number(value) -> bool:
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False


def _style_for_speaker(agent_id: str, speaker_styles: dict[str, str]) -> str:
    if agent_id not in speaker_styles:
        palette = [
            Style.CYAN,
            Style.GREEN,
            Style.YELLOW,
            Style.MAGENTA,
            Style.BLUE,
            Style.RED,
        ]
        speaker_styles[agent_id] = palette[len(speaker_styles) % len(palette)]
    return speaker_styles[agent_id]


class TypingIndicator:
    def __init__(self, message: str) -> None:
        self.message = message
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=0.3)
        sys.stdout.write("\r\033[K")
        sys.stdout.flush()

    def _run(self) -> None:
        frames = ["░", "▒", "▓"]
        index = 0
        while not self._stop.is_set():
            sys.stdout.write("\r" + color(f"{frames[index % len(frames)]} {self.message}", Style.DIM))
            sys.stdout.flush()
            index += 1
            time.sleep(0.25)

def _print_run_summary(result: ExperimentResult) -> None:
    print(logo())
    print(banner("SimulaCrew Run Complete", result.task_title))
    print(section("Run"))
    print(
        card(
            "Session",
            [
                ("provider", result.provider),
                ("model", result.model),
                ("rounds", str(len(result.rounds))),
                ("experiment", result.experiment_name),
                ("format_valid", str(result.format_check.get("valid"))),
            ],
            width=88,
        )
    )
    print(section("Deliberation"))
    for round_result in result.rounds:
        print(
            panel(
                f"{round_result.mode.upper()} :: {round_result.title}",
                f"statements: {len(round_result.statements)} | transcript: {round_result.transcript_visibility}",
                width=88,
            )
        )
        for statement in round_result.statements:
            marker = "!" if statement.event_type == "interrupt" else ">"
            label = f"{marker} {statement.agent_name} [{statement.event_type} turn {statement.turn_number}]"
            preview = " ".join(statement.content.split())[:180]
            print("  " + color(label, Style.BOLD))
            print("    " + wrap(preview, width=82))


if __name__ == "__main__":
    raise SystemExit(main())
