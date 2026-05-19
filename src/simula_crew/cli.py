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
    BUILDATHON_DEFAULT_TASK_PROMPT,
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


SUPPORTED_DOCUMENT_EXTENSIONS = {".csv", ".md", ".txt"}


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
              simulacrew chat
              simulacrew chat --provider claude --interruption-classifier llm
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
        default=BUILDATHON_DEFAULT_TASK_PROMPT,
    )
    ingest_parser.add_argument("--name", default="survey-team")
    ingest_parser.add_argument(
        "--document-dir",
        default=None,
        help=(
            "Optional normalized local document directory. Supports <person-id>.txt, "
            "<person-id>/*.txt, .md, and .csv files."
        ),
    )
    ingest_parser.add_argument(
        "--document-text-dir",
        default=None,
        help=argparse.SUPPRESS,
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
        default=BUILDATHON_DEFAULT_TASK_PROMPT,
    )
    google_ingest_parser.add_argument("--name", default="survey-team")
    google_ingest_parser.add_argument(
        "--document-dir",
        default=None,
        help=(
            "Optional normalized local document directory. Supports <person-id>.txt, "
            "<person-id>/*.txt, .md, and .csv files."
        ),
    )
    google_ingest_parser.add_argument(
        "--document-text-dir",
        default=None,
        help=argparse.SUPPRESS,
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

    chat_parser = subparsers.add_parser(
        "chat",
        help="Open a chat-style prompt loop for running simulations.",
    )
    chat_parser.add_argument(
        "experiment",
        nargs="?",
        default="configs/experiments/simulacra.yaml",
        help="Experiment file to run. Defaults to configs/experiments/simulacra.yaml.",
    )
    chat_parser.add_argument(
        "--message",
        default=None,
        help="Run one chat message and exit. Useful for scripts and tests.",
    )
    chat_parser.add_argument("--var", action="append", default=None, metavar="KEY=VALUE")
    chat_parser.add_argument(
        "--provider",
        choices=["dry-run", "openai", "claude"],
        default="dry-run",
    )
    chat_parser.add_argument("--model", default=None)
    chat_parser.add_argument(
        "--interruption-classifier",
        choices=["deterministic", "llm"],
        default="deterministic",
        help="Use deterministic personality scoring or an LLM classifier for interruption decisions.",
    )
    chat_parser.add_argument(
        "--classifier-model",
        default=None,
        help="Model for LLM interruption classification. Defaults to --model.",
    )
    chat_parser.add_argument("--temperature", type=float, default=0.2)
    chat_parser.add_argument("--max-agents", type=int, default=None)
    chat_parser.add_argument("--output-dir", default="runs")
    chat_parser.add_argument(
        "--quiet",
        action="store_true",
        help="Only print output file paths after each run.",
    )
    chat_parser.add_argument(
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
    if args.command == "chat":
        return _chat(args, parser)

    parser.print_help()
    return 0


def _ingest_survey(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    try:
        documents = _load_document_dir(_document_dir_arg(args))
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
        documents = _load_document_dir(_document_dir_arg(args))
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


def _document_dir_arg(args: argparse.Namespace) -> str | None:
    return args.document_dir or args.document_text_dir


def _load_document_dir(path: str | None) -> dict[str, list[DocumentText]]:
    if not path:
        return {}
    root = Path(path)
    if not root.exists():
        raise OSError(f"document directory does not exist: {root}")
    if not root.is_dir():
        raise OSError(f"document path is not a directory: {root}")
    documents: dict[str, list[DocumentText]] = {}
    for document_path in _iter_document_files(root):
        person_id = _person_id_for_document(root, document_path)
        documents.setdefault(person_id, []).append(
            DocumentText(
                source=document_path.relative_to(root).as_posix(),
                content=document_path.read_text(encoding="utf-8"),
            )
        )
    return documents


def _iter_document_files(root: Path):
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_DOCUMENT_EXTENSIONS
    )


def _person_id_for_document(root: Path, document_path: Path) -> str:
    relative = document_path.relative_to(root)
    if len(relative.parts) > 1:
        return relative.parts[0]
    return document_path.stem


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


def _chat(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    if args.max_agents is not None and args.max_agents <= 0:
        parser.error("--max-agents must be greater than zero")
    if args.temperature < 0 or args.temperature > 2:
        parser.error("--temperature must be between 0 and 2")

    try:
        base_experiment = load_experiment(args.experiment)
        variables = parse_variable_assignments(args.var)
        client = create_client(args.provider)
        model = args.model or _default_model(args.provider)
        interruption_classifier = None
        if args.interruption_classifier == "llm":
            interruption_classifier = LLMInterruptionClassifier(
                client=client,
                model=args.classifier_model or model,
            )
    except (ConfigError, RuntimeError, ValueError) as exc:
        parser.exit(2, f"error: {exc}\n")

    if args.quiet and args.message is None:
        parser.error("--quiet is only supported with --message")
    if args.message is None and not sys.stdin.isatty():
        parser.error("chat mode needs a terminal, or pass --message")

    if not args.quiet:
        print(logo())
        print(_chat_welcome(base_experiment.name, base_experiment.task.title, args.provider, model))

    messages = [args.message] if args.message is not None else None
    while True:
        if messages is None:
            message = _chat_input()
        else:
            message = messages.pop(0) if messages else None
        if message is None:
            break
        message = message.strip()
        if not message:
            continue
        if message.lower() in {"exit", "quit", ":q"}:
            if not args.quiet:
                print(_chat_system("Session closed."))
            break

        try:
            json_path, conversation_path = _run_chat_message(
                base_experiment=base_experiment,
                experiment_path=Path(args.experiment),
                message=message,
                variables=variables,
                client=client,
                model=model,
                temperature=args.temperature,
                max_agents=args.max_agents,
                interruption_classifier=interruption_classifier,
                output_dir=args.output_dir,
                provider=args.provider,
                classifier_mode=args.interruption_classifier,
                show_interruption_notes=args.show_interruption_notes,
                quiet=args.quiet,
            )
        except (ConfigError, RuntimeError, ValueError) as exc:
            parser.exit(2, f"error: {exc}\n")

        if args.quiet:
            print(json_path)
            print(conversation_path)
        else:
            print(_chat_artifacts(json_path, conversation_path))

        if messages is not None:
            break
    return 0


def _run_chat_message(
    *,
    base_experiment,
    experiment_path: Path,
    message: str,
    variables: dict[str, str],
    client,
    model: str,
    temperature: float,
    max_agents: int | None,
    interruption_classifier,
    output_dir: str,
    provider: str,
    classifier_mode: str,
    show_interruption_notes: bool,
    quiet: bool,
) -> tuple[Path, Path]:
    experiment = apply_runtime_inputs(
        base_experiment,
        prompt=message,
        variables=variables,
    )
    event_callback = None
    if not quiet:
        print(_chat_bubble("You", message, accent=Style.CORAL))
        print(_chat_system("Crew is deliberating. Live transcript follows."))
        event_callback = _live_event_printer(
            show_interruption_notes=show_interruption_notes,
            chat_style=True,
        )
    result = run_experiment(
        experiment=experiment,
        client=client,
        model=model,
        temperature=temperature,
        max_agents=max_agents,
        interruption_classifier=interruption_classifier,
        event_callback=event_callback,
        run_metadata={
            "experiment_path": str(experiment_path),
            "output_dir": str(Path(output_dir)),
            "interruption_classifier": classifier_mode,
            "chat_interface": True,
            "provider": provider,
        },
    )
    return save_experiment_result(result, output_dir)


def _chat_welcome(
    experiment_name: str,
    task_title: str,
    provider: str,
    model: str,
) -> str:
    rows = [
        color("SimulaCrew Chat", Style.BOLD + Style.CORAL),
        color(task_title, Style.MUTED),
        "",
        f"{color('experiment', Style.BOLD):<20} {experiment_name}",
        f"{color('provider', Style.BOLD):<20} {provider}",
        f"{color('model', Style.BOLD):<20} {model}",
        "",
        color("Type a simulation prompt. Use exit, quit, or :q to leave.", Style.MUTED),
    ]
    return _rounded_box(rows, title="chat")


def _chat_input() -> str:
    try:
        return input(color("\nsimulacrew ", Style.BOLD + Style.CORAL) + color("› ", Style.BOLD))
    except EOFError:
        return "exit"


def _chat_system(message: str) -> str:
    return _chat_bubble("SimulaCrew", message, accent=Style.MUTED)


def _chat_artifacts(json_path: Path, conversation_path: Path) -> str:
    return _rounded_box(
        [
            color("Saved artifacts", Style.BOLD + Style.CORAL),
            f"{color('json', Style.BOLD):<14} {json_path}",
            f"{color('conversation', Style.BOLD):<14} {conversation_path}",
        ],
        title="done",
    )


def _chat_bubble(
    speaker: str,
    body: str,
    *,
    accent: str,
    width: int = 88,
) -> str:
    content_width = max(36, width - 6)
    lines = [
        color(f"{speaker}", Style.BOLD + accent),
        "",
    ]
    for paragraph in body.strip().splitlines() or [""]:
        wrapped = wrap_text(paragraph, width=content_width) if paragraph else [""]
        lines.extend(wrapped)
    return _rounded_box(lines, title="message", width=width, accent=accent)


def _rounded_box(
    lines: list[str],
    *,
    title: str,
    width: int = 88,
    accent: str = Style.CORAL,
) -> str:
    width = max(44, width)
    title_text = f" {title} "
    top = "╭─" + title_text + "─" * max(0, width - len(title_text) - 3) + "╮"
    bottom = "╰" + "─" * (width - 2) + "╯"
    rendered = [color(top, accent)]
    inner_width = width - 4
    for line in lines:
        plain_len = _visible_len(line)
        padding = " " * max(0, inner_width - plain_len)
        rendered.append(color("│ ", accent) + line + padding + color(" │", accent))
    rendered.append(color(bottom, accent))
    return "\n".join(rendered)


def _visible_len(text: str) -> int:
    length = 0
    in_escape = False
    for char in text:
        if char == "\033":
            in_escape = True
            continue
        if in_escape:
            if char == "m":
                in_escape = False
            continue
        length += 1
    return length


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


def _live_event_printer(*, show_interruption_notes: bool = False, chat_style: bool = False):
    speaker_styles: dict[str, str] = {}
    indicator: TypingIndicator | None = None
    last_alignment_payload: dict | None = None

    def handle(event_type: str, payload: dict) -> None:
        nonlocal indicator, last_alignment_payload
        if event_type == "round_start":
            if indicator:
                indicator.stop()
                indicator = None
            if chat_style and last_alignment_payload:
                print(_alignment_status(last_alignment_payload, speaker_styles), flush=True)
                last_alignment_payload = None
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
                (
                    _chat_statement(
                        statement,
                        style,
                        show_interruption_notes=show_interruption_notes,
                    )
                    if chat_style
                    else _conversation_statement(
                        statement,
                        style,
                        show_interruption_notes=show_interruption_notes,
                    )
                ),
                flush=True,
            )
            return

        if event_type == "alignment_update":
            if chat_style and not _should_show_chat_alignment(payload):
                last_alignment_payload = payload
                return
            last_alignment_payload = None
            print(_alignment_status(payload, speaker_styles), flush=True)
            return

        if event_type == "goal_aligned":
            if indicator:
                indicator.stop()
                indicator = None
            if chat_style and last_alignment_payload:
                print(_alignment_status(last_alignment_payload, speaker_styles), flush=True)
                last_alignment_payload = None
            print(color("Goal convergence reached. Moving to PRD.", Style.DIM), flush=True)
            return

        if event_type == "discussion_time_limit_reached":
            if indicator:
                indicator.stop()
                indicator = None
            if chat_style and last_alignment_payload:
                print(_alignment_status(last_alignment_payload, speaker_styles), flush=True)
                last_alignment_payload = None
            print(color("Discussion time limit reached. Moving to PRD.", Style.DIM), flush=True)
            return

    return handle


def _should_show_chat_alignment(payload: dict) -> bool:
    if payload.get("aligned"):
        return True
    try:
        turn_number = int(payload.get("turn_number", 0))
    except (TypeError, ValueError):
        return False
    return turn_number > 0 and turn_number % 4 == 0


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


def _chat_statement(
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
    title = f"{statement.agent_name} · {label.lower()} · {time_text}{score_text}"
    if statement.event_type == "thought":
        body = "[stays quiet]"
    else:
        body = statement.content.strip()
    return _chat_bubble(title, body, accent=speaker_style)


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
