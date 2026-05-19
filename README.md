# SimulaCrew

Personality-driven agent debate harness with realistic interruptions, a runnable CLI, and research-backed interaction presets.

```text
+-----------------------+
| Preset JSON           |
| topic + agents        |
| personality fields    |
+-----------+-----------+
            |
            v
+-----------------------+
| Persona Harness       |
| builds private        |
| character prompts     |
+-----------+-----------+
            |
            v
+-----------------------+
| Conversation Engine   |
| read -> group chat -> |
| final synthesis       |
+-----------+-----------+
            |
            v
+-----------------------+
| Outputs               |
| conversation.txt      |
| structured.json       |
+-----------------------+
```

## What It Does

SimulaCrew runs small crews of LLM agents whose behavior is shaped by configurable personality/questionnaire fields. Agents first read the prompt independently, then talk in a normal group chat where a high-interruption persona can cut in naturally.

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

If you see this error:

```text
error: Claude support requires: pip install -e '.[claude]'
```

run:

```bash
source .venv/bin/activate
pip install -e '.[claude]'
```

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

By default the CLI prints a live, jury-sim-style conversation: each agent briefly thinks, then the group chats normally, with occasional natural cut-ins. The recorder only wraps up at the end. This is not token-by-token streaming, but it prints before and after each agent call so long Claude runs do not look frozen.

To disable live output and print only the final summary:

```bash
simulacrew run configs/simulacra.json --no-live
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
  --model opus \
  --prompt-file challenge.txt
```

Run with Claude and an inline one-line prompt:

```bash
simulacrew run configs/simulacra.json \
  --provider claude \
  --model opus \
  --interruption-classifier llm \
  --prompt "Should we build the moral deliberation simulator first or the general agent harness first?"
```

Keep inline prompts on one line unless you intentionally want a newline inside the prompt.

Use Claude for interruption classification too:

```bash
simulacrew run configs/simulacra.json \
  --provider claude \
  --model opus \
  --interruption-classifier llm \
  --prompt-file challenge.txt
```

Use the exact Claude model alias that your Claude Agent SDK install supports. `opus` is shown here because this harness benefits from stronger reasoning during agent turns.

Outputs are written to `runs/`:

- `*.txt`: compact conversation log
- `*.json`: structured run data

## CLI Map

```text
simulacrew
|-- list
|   `-- show available presets
|-- inspect <config>
|   `-- show topic, agents, traits, and rounds
`-- run <config>
    |-- --prompt "..."
    |-- --prompt-file challenge.txt
    |-- --provider dry-run|claude|openai
    |-- --interruption-classifier deterministic|llm
    |-- --no-live
    |-- --max-agents 2
    `-- --output-dir runs/demo
```

## How SimulaCrew Interactions Work

The terminal conversation is the main experience. Saved files are support artifacts: JSON for machines and a compact text log for quick review.

```text
1. Private thinking
   Each agent forms a short starting position before the group chat.

2. Structured debate
   Agents talk in short turns. Each turn should do one thing:
   propose, challenge, clarify, merge, or decide. The goal is
   consensus, but not fake agreement.

3. Natural interruptions
   During the group chat, each turn scores who is most likely to cut in.
   If the score is high enough and the agent is not just repeating the
   last speaker, that agent takes the next message as a cut-in.

4. Final synthesis
   A neutral recorder writes the decision, dissent, risks, and next steps.
```

Live CLI output is intentionally shaped like the `jurysim` room transcript:

```text
----------------------------------------------------------------------------------------
GROUP CHAT / Group Chat
----------------------------------------------------------------------------------------
... Mara typing (turn 1)
Mara  says  15:46:12
| We should build the harness first, but constrain it to one preset
| so it proves something today.

... Niko typing (turn 2)
Niko  says  15:46:18
| I disagree with a bare harness. Add the inspection trail, or we
| cannot tell simulation from roleplay.

... Mara cuts in (turn 3)
Mara  cuts in  15:46:25
| Wait, we are turning this into architecture talk again. What is the
| smallest harness that lets us run one moral scenario today?
```

Internal interruption scores and classifier notes are hidden by default because they make the run feel less like a real group chat. To debug them:

```bash
simulacrew run configs/simulacra.json --show-interruption-notes
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
    "neuroticism": 5,
    "interruptiveness": 7,
    "disagreement_sensitivity": 9,
    "patience": 4
  },
  "goals": ["prevent shallow simulation", "force testable claims"],
  "constraints": ["must offer a replacement when blocking an idea"]
}
```

The harness turns those fields into private character prompts with `harness.character_prompt_template`.

```text
Preset personality fields
      |
      v
Character prompt template
      |
      v
Agent-specific system prompt
      |
      v
Statement in debate
```

## Interruption Architecture

```text
Current transcript
      |
      +--------------+
      |              |
      v              v
Personality       Optional LLM
formula           classifier
      |              |
      +------+-------+
             v
   interruption score
             |
             v
  speaker selected for turn
```

The deterministic score is not a generic dominance score. It estimates the agent's current tendency to interrupt right now.

The scheduler works like this:

1. Score every agent against the current transcript.
2. Add pressure for assertiveness, skepticism, urgency, interruptiveness, and disagreement sensitivity.
3. Reduce pressure for patience, agreeableness, and repeated recent speaking.
4. Penalize the last speaker so one loud agent does not dominate.
5. Pick the highest current cut-in score. If there is a near tie, rotate away from the last speaker.

The score uses questionnaire-like fields:

- assertiveness
- skepticism
- urgency
- extraversion
- agreeableness
- conscientiousness
- neuroticism
- interruptiveness
- disagreement sensitivity
- patience
- recent speaking frequency
- whether the agent spoke last
- whether the agent has been silent recently

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
|-- configs/
|   `-- simulacra.json
|-- src/simula_crew/
|   |-- cli.py
|   |-- clients.py
|   |-- engine.py
|   |-- io.py
|   |-- prompts.py
|   |-- runtime.py
|   |-- schema.py
|   |-- scoring.py
|   `-- terminal.py
|-- tests/
|   `-- test_simula_crew.py
`-- README.md
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
