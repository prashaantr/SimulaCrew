# SimulaCrew

Personality-driven agent debate harness with realistic interruptions, a runnable CLI, and research-backed interaction presets.

```text
┌──────────────────────┐
│  Preset JSON          │
│  topic + agents       │
│  personality fields   │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  Persona Harness      │
│  builds private       │
│  character prompts    │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  Deliberation Engine  │
│  private → debate →   │
│  interruption → final │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  Outputs              │
│  transcript.md        │
│  structured.json      │
└──────────────────────┘
```

## What It Does

SimulaCrew runs small crews of LLM agents whose behavior is shaped by configurable personality/questionnaire fields. Agents first reason independently, then debate, then enter an interruption window where the harness decides who should interrupt based on personality and context.

It is intentionally not just a loop of chat messages. The default `simulacra` preset combines:

- independent ideation from nominal group technique,
- multi-agent debate rounds,
- personality-weighted interruption scoring,
- optional LLM interruption classification,
- neutral final synthesis that preserves dissent.

## Install

```bash
git clone https://github.com/prashaantr/SimulaCrew.git
cd SimulaCrew

python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

Claude Agent SDK support:

```bash
pip install -e '.[claude]'
export ANTHROPIC_API_KEY='your-key'
```

Do not commit real keys. Use `.env.example` only as a template.

## Run The CLI

List presets:

```bash
simulacrew list
```

Inspect the test preset:

```bash
simulacrew inspect configs/simulacra.json
```

Run a no-API dry run:

```bash
simulacrew run configs/simulacra.json
```

Run with a custom prompt:

```bash
simulacrew run configs/simulacra.json \
  --prompt "Should we build a moral deliberation simulator or a general agent harness first?"
```

Run with Claude Agent SDK:

```bash
simulacrew run configs/simulacra.json \
  --provider claude \
  --model sonnet \
  --prompt-file challenge.txt
```

Use Claude for interruption classification too:

```bash
simulacrew run configs/simulacra.json \
  --provider claude \
  --model sonnet \
  --interruption-classifier llm \
  --prompt-file challenge.txt
```

Outputs are written to `runs/`:

- `*.md`: readable transcript
- `*.json`: structured run data

## CLI Map

```text
simulacrew
├── list
│   └── show available presets
├── inspect <config>
│   └── show topic, agents, traits, and rounds
└── run <config>
    ├── --prompt "..."
    ├── --prompt-file challenge.txt
    ├── --provider dry-run|claude|openai
    ├── --interruption-classifier deterministic|llm
    ├── --max-agents 2
    └── --output-dir runs/demo
```

## Where Personalities Go

Personality lives in the preset under `agents[].personality`.

```json
{
  "id": "niko",
  "name": "Niko",
  "base_prompt": "You are a careful critic. You look for failure modes.",
  "personality": {
    "assertiveness": 7,
    "skepticism": 9,
    "urgency": 5,
    "extraversion": 5,
    "agreeableness": 3,
    "conscientiousness": 8,
    "openness": 6,
    "neuroticism": 5
  },
  "goals": ["prevent shallow simulation", "force testable claims"],
  "constraints": ["must offer a replacement when blocking an idea"]
}
```

The harness turns those fields into private character prompts with `harness.character_prompt_template`.

```text
Preset personality fields
      │
      ▼
Character prompt template
      │
      ▼
Agent-specific system prompt
      │
      ▼
Statement in debate
```

## Interruption Architecture

```text
Current transcript
      │
      ├──────────────┐
      ▼              ▼
Personality       Optional LLM
formula           classifier
      │              │
      └──────┬───────┘
             ▼
   interruption score
             │
             ▼
  speaker selected for turn
```

The deterministic score uses questionnaire-like fields:

- assertiveness
- skepticism
- urgency
- extraversion
- agreeableness
- conscientiousness
- neuroticism
- recent speaking frequency

The optional LLM classifier sees the agent persona, transcript, round rules, and topic. It returns JSON with `score`, `should_interrupt`, and `rationale`. If classification fails, SimulaCrew falls back to the deterministic score.

## Research Basis

SimulaCrew's default preset draws from:

- Multi-agent debate for improved factuality and reasoning: Du et al., 2023, <https://arxiv.org/abs/2305.14325>
- Divergent thinking through multi-agent debate: Liang et al., 2023, <https://arxiv.org/abs/2305.19118>
- Nominal group technique for independent idea generation and structured sharing: ASQ overview, <https://asq.org/quality-resources/nominal-group-technique>
- Delphi-style controlled feedback and anonymity principles: Delphi Academy, <https://www.durvey.org/academy/fundamentals/key-principles>
- Claude Agent SDK Python docs: <https://docs.claude.com/en/docs/agent-sdk/python>

The preset uses these as design constraints, not as a claim that LLM agents perfectly reproduce human group behavior.

## Test

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
python3 -m compileall -q src tests
simulacrew run configs/simulacra.json --max-agents 2 --output-dir runs/smoke
```

## Project Layout

```text
SimulaCrew/
├── configs/
│   └── simulacra.json
├── src/simula_crew/
│   ├── cli.py
│   ├── clients.py
│   ├── engine.py
│   ├── io.py
│   ├── prompts.py
│   ├── runtime.py
│   ├── schema.py
│   ├── scoring.py
│   └── terminal.py
├── tests/
│   └── test_simula_crew.py
└── README.md
```

## Contact Information

Emily  
Email: xehu@mit.edu  
GitHub: <https://github.com/xehu>

Prashaant  
Email: prashaant.rn@gmail.com, prashran@stanford.edu  
GitHub: <https://github.com/prashaantr>

Andy  
Email: h4upt@stanford.edu  
GitHub: <https://github.com/indraos>

David  
Email: davemholtz@gmail.com  
GitHub: <https://github.com/daveholtz>
