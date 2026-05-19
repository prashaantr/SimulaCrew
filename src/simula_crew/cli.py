from __future__ import annotations

import argparse
from pathlib import Path

from simula_crew.clients import create_client
from simula_crew.engine import run_crew
from simula_crew.io import list_configs, load_config, save_result
from simula_crew.runtime import apply_runtime_inputs, parse_variable_assignments
from simula_crew.scoring import LLMInterruptionClassifier
from simula_crew.schema import ConfigError, CrewResult
from simula_crew.terminal import Style, banner, color, key_value, section, wrap


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="simulacrew",
        description="Run personality-driven agent crews that debate, interrupt, and synthesize.",
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
    print(banner("SimulaCrew", "personality-driven agent debate harness"))
    configs = list_configs(config_dir)
    if not configs:
        print("No presets found.")
        return 0
    print(section("Presets"))
    for config_path in configs:
        config = load_config(config_path)
        print(key_value(config.name, str(config_path)))
        print("  " + wrap(config.description, width=84))
    return 0


def _inspect(config_path: str) -> int:
    config = load_config(config_path)
    print(banner(config.name, config.description))
    print(section("Topic"))
    print(wrap(config.topic.prompt))
    print(section("Agents"))
    for agent in config.agents:
        print(color(agent.name, Style.BOLD + Style.GREEN))
        print("  " + wrap(agent.base_prompt, width=84))
        if agent.personality:
            traits = ", ".join(f"{key}={value}" for key, value in sorted(agent.personality.items()))
            print("  " + color(traits, Style.DIM))
    print(section("Rounds"))
    for round_spec in config.rounds:
        detail = f"{round_spec.mode}, max_turns={round_spec.max_turns or 'n/a'}"
        print(key_value(round_spec.title, detail))
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
        result = run_crew(
            config=config,
            client=client,
            model=model,
            temperature=args.temperature,
            max_agents=args.max_agents,
            interruption_classifier=interruption_classifier,
            run_metadata={
                "config_path": str(Path(args.config)),
                "output_dir": str(Path(args.output_dir)),
                "interruption_classifier": args.interruption_classifier,
            },
        )
        json_path, markdown_path = save_result(result, args.output_dir)
    except (ConfigError, RuntimeError, ValueError) as exc:
        parser.exit(2, f"error: {exc}\n")

    if args.quiet:
        print(json_path)
        print(markdown_path)
        return 0

    _print_run_summary(result)
    print(section("Artifacts"))
    print(key_value("JSON", str(json_path)))
    print(key_value("Markdown", str(markdown_path)))
    return 0


def _print_run_summary(result: CrewResult) -> None:
    print(banner("SimulaCrew Run Complete", result.topic_title))
    print(section("Deliberation"))
    for round_result in result.rounds:
        print(color(round_result.title, Style.BOLD + Style.CYAN))
        for statement in round_result.statements:
            label = f"{statement.agent_name} [{statement.event_type} turn {statement.turn_number}]"
            preview = " ".join(statement.content.split())[:180]
            print("  " + color(label, Style.BOLD))
            print("  " + wrap(preview, width=84))
