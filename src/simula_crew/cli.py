from __future__ import annotations

import argparse
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from textwrap import dedent, wrap as wrap_text

from simula_crew.clients import create_client
from simula_crew.engine import run_crew
from simula_crew.io import list_configs, load_config, save_result
from simula_crew.runtime import apply_runtime_inputs, parse_variable_assignments
from simula_crew.scoring import LLMInterruptionClassifier
from simula_crew.schema import ConfigError, CrewResult
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
              simulacrew inspect configs/simulacra.json
              simulacrew run configs/simulacra.json
              simulacrew run configs/simulacra.json --prompt "What should the crew decide?"
              simulacrew run configs/simulacra.json --provider claude --model sonnet --interruption-classifier llm --prompt-file challenge.txt
            """
        ),
    )
    subparsers = parser.add_subparsers(dest="command")

    list_parser = subparsers.add_parser("list", help="List available presets.")
    list_parser.add_argument("--config-dir", default="configs")

    inspect_parser = subparsers.add_parser("inspect", help="Show preset details.")
    inspect_parser.add_argument("config")

    run_parser = subparsers.add_parser("run", help="Run a crew preset.")
    run_parser.add_argument("config")
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
        return _list_presets(args.config_dir)
    if args.command == "inspect":
        return _inspect(args.config)
    if args.command == "run":
        return _run(args, parser)

    parser.print_help()
    return 0


def _list_presets(config_dir: str) -> int:
    print(logo())
    print(banner("Agent Debate CLI", "personality-driven crews with interruption mechanics"))
    configs = list_configs(config_dir)
    if not configs:
        print("No presets found.")
        return 0
    print(section("Presets"))
    for config_path in configs:
        config = load_config(config_path)
        print(panel(config.name, f"{config.description}\npath: {config_path}", width=88))
    return 0


def _inspect(config_path: str) -> int:
    config = load_config(config_path)
    print(logo())
    print(banner(config.name, config.description))
    print(
        card(
            "Preset",
            [
                ("file", config_path),
                ("agents", str(len(config.agents))),
                ("rounds", str(len(config.rounds))),
                ("topic", config.topic.title),
            ],
            width=88,
        )
    )
    print(section("Topic"))
    print(wrap(config.topic.prompt))
    print(section("Agents"))
    for agent in config.agents:
        print(panel(agent.name, agent.base_prompt, width=88))
        if agent.personality:
            traits = ", ".join(f"{key}={value}" for key, value in sorted(agent.personality.items()))
            print("  " + color(traits, Style.DIM))
    print(section("Rounds"))
    print(
        tree(
            [
                (
                    round_spec.title,
                    f"[{round_spec.mode}, max_turns={round_spec.max_turns or 'n/a'}]",
                )
                for round_spec in config.rounds
            ]
        )
    )
    return 0


def _run(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    if args.provider != "dry-run" and not args.model:
        parser.error("--model is required when --provider is not dry-run")
    if args.max_agents is not None and args.max_agents <= 0:
        parser.error("--max-agents must be greater than zero")
    if args.temperature < 0 or args.temperature > 2:
        parser.error("--temperature must be between 0 and 2")

    try:
        config = apply_runtime_inputs(
            load_config(args.config),
            prompt=args.prompt,
            prompt_file=args.prompt_file,
            variables=parse_variable_assignments(args.var),
        )
        client = create_client(args.provider)
        model = args.model or ("dry-run-model" if args.provider == "dry-run" else "sonnet")
        interruption_classifier = None
        if args.interruption_classifier == "llm":
            interruption_classifier = LLMInterruptionClassifier(
                client=client,
                model=args.classifier_model or model,
            )
        event_callback = None
        if not args.quiet and not args.no_live:
            _print_run_header(config.name, config.topic.title, args.provider, model)
            event_callback = _live_event_printer(
                show_interruption_notes=args.show_interruption_notes,
            )
        result = run_crew(
            config=config,
            client=client,
            model=model,
            temperature=args.temperature,
            max_agents=args.max_agents,
            interruption_classifier=interruption_classifier,
            event_callback=event_callback,
            run_metadata={
                "config_path": str(Path(args.config)),
                "output_dir": str(Path(args.output_dir)),
                "interruption_classifier": args.interruption_classifier,
            },
        )
        json_path, conversation_path = save_result(result, args.output_dir)
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
                ("mode", "█ live per-agent output"),
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
            if sys.stdout.isatty():
                if indicator:
                    indicator.stop()
                indicator = TypingIndicator(message)
                indicator.start()
            rationale = payload.get("interruption_rationale")
            if rationale and show_interruption_notes:
                if indicator:
                    indicator.stop()
                    indicator = None
                print(color(f"░ note: {rationale}", Style.DIM), flush=True)
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

    return handle


def _conversation_phase(mode: str, title: str) -> str:
    line = "█" * 88
    phase = "GROUP CHAT" if mode == "DISCUSSION" else mode
    return "\n".join(
        [
            "",
            color(line, Style.DIM),
            color(f"█ {phase} / {title}", Style.BOLD + Style.MAGENTA),
            color(line, Style.DIM),
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
    marker = "█" if statement.event_type == "interrupt" else "▓"
    header = (
        color(f"{marker} {statement.agent_name}", speaker_style + Style.BOLD)
        + color(f"  {label.lower()}  {time_text}{score_text}", Style.DIM)
    )
    lines = [header]
    for paragraph in statement.content.strip().splitlines() or [""]:
        wrapped = wrap_text(paragraph, width=82) if paragraph else [""]
        for line in wrapped:
            lines.append(color("░ ", speaker_style) + line)
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
        sys.stdout.write("\r" + " " * 100 + "\r")
        sys.stdout.flush()

    def _run(self) -> None:
        frames = ["░  ", "▒  ", "▓  ", "█  "]
        index = 0
        while not self._stop.is_set():
            sys.stdout.write("\r" + color(f"{frames[index % len(frames)]} {self.message}", Style.DIM))
            sys.stdout.flush()
            index += 1
            time.sleep(0.25)

def _print_run_summary(result: CrewResult) -> None:
    print(logo())
    print(banner("SimulaCrew Run Complete", result.topic_title))
    print(section("Run"))
    print(
        card(
            "Session",
            [
                ("provider", result.provider),
                ("model", result.model),
                ("rounds", str(len(result.rounds))),
                ("config", result.config_name),
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
