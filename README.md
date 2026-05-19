# SimulaCrew

Personality-driven agent simulation harness with realistic group chat, natural interruptions, convergence tracking, a runnable CLI, and PRD output.

```text
+--------------------------------------------------------------------------------+
|                                SIMULACREW                                      |
+--------------------------------------------------------------------------------+
|                                                                                |
|  configs/*.json                                                                |
|  topic + agents + personas + phases + output contract                          |
|          |                                                                     |
|          v                                                                     |
|  src/simula_crew/io.py                                                         |
|  load_config() validates JSON into schema objects                              |
|          |                                                                     |
|          v                                                                     |
|  src/simula_crew/runtime.py                                                    |
|  applies --prompt, --prompt-file, and --var overrides                          |
|          |                                                                     |
|          v                                                                     |
|  src/simula_crew/engine.py                                                     |
|  runs private thinking, dynamic group chat, interruptions, convergence, PRD    |
|          |                                                                     |
|          v                                                                     |
|  src/simula_crew/clients.py                                                    |
|  dry-run / Claude / OpenAI completion calls                                    |
|          |                                                                     |
|          v                                                                     |
|  runs/*.json + runs/*.txt                                                      |
|  structured state + readable conversation                                      |
|                                                                                |
+--------------------------------------------------------------------------------+
```

## What It Does

SimulaCrew runs small crews of LLM agents whose behavior is shaped by configurable personality/questionnaire fields. Agents first read the prompt independently, then talk in a normal group chat where a high-interruption persona can cut in naturally. The default preset is a hackathon-style crew: it has a 20-minute discussion cap, tries to converge on one project idea, then outputs a compact PRD.

It is intentionally not just a loop of chat messages. The default `simulacra` preset combines:

- independent ideation from nominal group technique,
- multi-agent debate rounds,
- personality-weighted interruption scoring,
- optional LLM interruption classification,
- convergence tracking toward a shared goal,
- neutral PRD synthesis that preserves dissent.

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

By default the CLI prints a live, [jury-sim-style conversation](https://github.com/prashaantr/jurysim): each agent briefly thinks, then the group chats normally, with occasional natural cut-ins. The recorder writes the PRD only after discussion ends. This is not token-by-token streaming, but it prints before and after each agent call so long Claude runs do not look frozen.

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
  --prompt-file challenge.txt
```

Run with Claude and an inline one-line prompt:

```bash
simulacrew run configs/simulacra.json \
  --provider claude \
  --interruption-classifier llm \
  --prompt "Should we build the moral deliberation simulator first or the general agent harness first?"
```

Keep inline prompts on one line unless you intentionally want a newline inside the prompt.

Use Claude for interruption classification too:

```bash
simulacrew run configs/simulacra.json \
  --provider claude \
  --interruption-classifier llm \
  --prompt-file challenge.txt
```

When `--provider claude` is used without `--model`, SimulaCrew defaults to `haiku` for faster agent turns. You can override it with the exact Claude model alias your Claude Agent SDK install supports:

```bash
simulacrew run configs/simulacra.json \
  --provider claude \
  --model opus \
  --interruption-classifier llm \
  --prompt-file challenge.txt
```

Outputs are written to `runs/`:

- `*.txt`: compact conversation log
- `*.json`: structured run data

The final recorder turn contains the PRD. The conversation log is intentionally compact text, not a giant Markdown transcript.

## CLI Map

```text
+--------------------------------------------------------------------------------+
| CLI COMMANDS                                                                   |
+--------------------------------------------------------------------------------+
| simulacrew list                                                                |
|   show available presets                                                       |
|                                                                                |
| simulacrew inspect configs/simulacra.json                                      |
|   show topic, agents, traits, and rounds                                       |
|                                                                                |
| simulacrew run configs/simulacra.json                                          |
|   run dry-run mode without an API                                              |
|                                                                                |
| simulacrew run configs/simulacra.json --provider claude --model opus           |
|   run live agents through Claude Agent SDK                                     |
+--------------------------------------------------------------------------------+

+--------------------------------------------------------------------------------+
| RUN FLAGS                                                                      |
+--------------------------------------------------------------------------------+
| --prompt "..."                         replace the preset topic                 |
| --prompt-file challenge.txt            load the topic from a file               |
| --var key=value                        override topic.variables                 |
| --provider dry-run|claude|openai       choose model backend                     |
| --interruption-classifier deterministic|llm choose cut-in scoring mode          |
| --show-interruption-notes              print hidden scores/rationales           |
| --no-live                              print summary after completion           |
| --max-agents 2                         use the first N agents                   |
| --output-dir runs/demo                 choose output directory                  |
+--------------------------------------------------------------------------------+
```

## How SimulaCrew Interactions Work

The terminal conversation is the main experience. Saved files are support artifacts: JSON for machines and a compact text log for quick review.

```text
+-----------------------+     +-----------------------+     +----------------------+
| 1. PRIVATE THINKING   | --> | 2. GROUP CHAT         | --> | 3. CONVERGENCE       |
| each agent reads      |     | dynamic speaker order |     | per-agent alignment  |
| alone before debate   |     | says / cuts in / quiet|     | stop or continue     |
+-----------------------+     +-----------------------+     +----------+-----------+
                                                                      |
                                                                      v
                                                        +--------------------------+
                                                        | 4. PRD SYNTHESIS         |
                                                        | recorder writes product  |
                                                        | requirements document    |
                                                        +--------------------------+
```

## Phases And Actions

```text
+----------------------+-------------------------+------------------------------+
| phase                | visible action          | what the agent is doing       |
+----------------------+-------------------------+------------------------------+
| Independent Positions| thinking                | reads the prompt alone        |
| Group Chat           | says                    | responds normally             |
| Group Chat           | cuts in                 | interrupts to contest/merge   |
| Group Chat           | stays quiet             | stores a private thought      |
| Convergence Check    | hidden by default       | updates goal alignment state  |
| Final Synthesis      | wraps up                | writes the PRD                |
+----------------------+-------------------------+------------------------------+
```

The human running the CLI can see the independent thinking phase. The agents do not see each other's private thinking. Those private turns are stored in that agent's private memory and fed back only to the same agent on later turns.

## Terms

```text
+--------------------------+-----------------------------------------------------+
| term                     | meaning                                             |
+--------------------------+-----------------------------------------------------+
| preset                   | a JSON scenario in configs/                         |
| topic                    | the problem/question the crew is trying to solve    |
| agent                    | one simulated participant with a persona            |
| persona                  | base prompt, history, skills, interests, traits     |
| round                    | one phase of the run                                |
| private round            | agents think independently before group chat        |
| parallel thinking        | private round calls run concurrently                |
| private thinking worker  | thread slot used for parallel private calls         |
| private memory           | notes only the same agent can see later             |
| public transcript        | messages agents actually said in the group chat     |
| public turn              | a spoken group-chat message, not private thinking   |
| group chat               | normal conversation where agents respond in turn    |
| normal turn              | a non-interrupting "says" message                   |
| protected opening        | early public turns where cut-ins are disabled       |
| cut-in                   | an interruption rendered as "cuts in"               |
| contestation pressure    | how strongly an agent wants to challenge direction  |
| interruption score       | private estimate of how likely an agent is to cut in|
| buy-in score             | how aligned each agent is with the emerging idea    |
| current idea             | the idea the group appears to be converging on      |
| agent idea view          | what one agent seems to think the current idea is   |
| convergence              | internal estimate that agents are aligned on goal   |
| alignment evaluator      | LLM or fallback process that updates buy-in state   |
| fallback steps           | deterministic buy-in updates for dry-run/testing    |
| discussion cap           | hard time limit before moving to PRD synthesis      |
| show-interruption-notes  | CLI flag for classifier notes and agent idea views  |
| model provider           | dry-run, Claude, or OpenAI backend                  |
| Claude Haiku default     | default Claude model when --model is omitted        |
| PRD recorder             | neutral final writer that turns discussion into PRD |
| artifact                 | saved JSON and text output in runs/                 |
+--------------------------+-----------------------------------------------------+
```

## What Each Phase Means

```text
+--------------------------------------------------------------------------------+
| 1. PRIVATE ROUND / INDEPENDENT POSITIONS                                       |
+--------------------------------------------------------------------------------+
| Each agent receives the topic and its own persona. Agents do not see each       |
| other's private answers. This phase creates each agent's starting mental state. |
| SimulaCrew runs these calls in parallel because they are independent.           |
+--------------------------------------------------------------------------------+

+--------------------------------------------------------------------------------+
| 2. GROUP CHAT                                                                  |
+--------------------------------------------------------------------------------+
| Agents see the public transcript only: messages that were actually spoken.      |
| The engine chooses who speaks next from interruption pressure, recent silence,  |
| and recent speaking history. Early turns are protected so agents build on the   |
| prior message before cut-ins begin. A turn can become a normal "says", a        |
| "cuts in", or a hidden "stays quiet" private note.                             |
+--------------------------------------------------------------------------------+

+--------------------------------------------------------------------------------+
| 3. CONVERGENCE CHECK                                                           |
+--------------------------------------------------------------------------------+
| After public turns, the engine updates internal alignment toward the shared     |
| goal. Live Claude runs can use an LLM evaluator; dry-run uses configured        |
| fallback steps. The CLI prints buy-in scores and the current idea after turns.  |
+--------------------------------------------------------------------------------+

+--------------------------------------------------------------------------------+
| 4. FINAL SYNTHESIS / PRD RECORDER                                              |
+--------------------------------------------------------------------------------+
| Once the crew converges or the time limit is reached, a neutral recorder reads  |
| the public transcript plus hidden private notes and writes the final PRD.       |
+--------------------------------------------------------------------------------+
```

## Code Map

```text
SimulaCrew/
|
|-- configs/
|   `-- simulacra.json
|       Preset definition. Put topics, agent personas, phase prompts,
|       interruption rules, convergence settings, and PRD contract here.
|
|-- src/simula_crew/
|   |
|   |-- cli.py
|   |   Argument parser and terminal experience. Builds the client, starts
|   |   run_crew(), prints live agent turns, and saves artifacts.
|   |
|   |-- engine.py
|   |   Core simulator. Selects speakers, decides quiet vs speak vs cut-in,
|   |   stores private thoughts, updates convergence, and calls the recorder.
|   |
|   |-- scoring.py
|   |   Interruption scoring. Deterministic classifier for tests/dry-run and
|   |   optional LLM classifier for transcript-aware interruption decisions.
|   |
|   |-- clients.py
|   |   Model backends. DryRunClient, ClaudeSDKClient, and OpenAIClient all
|   |   expose the same complete(...) method.
|   |
|   |-- prompts.py
|   |   Prompt rendering. Injects topic variables, transcript, persona fields,
|   |   private memory, and convergence state into round prompts.
|   |
|   |-- schema.py
|   |   Typed config objects and validation for topics, harness, agents,
|   |   rounds, statements, and results.
|   |
|   |-- runtime.py
|   |   Runtime overrides from --prompt, --prompt-file, and --var.
|   |
|   |-- io.py
|   |   Loads config JSON and writes run outputs to runs/*.json and runs/*.txt.
|   |
|   `-- terminal.py
|       Colors, banners, panels, wrapping, and logo helpers.
|
|-- tests/
|   `-- test_simula_crew.py
|       Config, prompt, runtime, scoring, engine, and artifact smoke tests.
|
`-- README.md
```

## Runtime Call Graph

```text
+--------------------+
| simulacrew run ... |
+---------+----------+
          |
          v
+--------------------+      +----------------------+
| cli._run()         | ---> | load_config()        |
| parse args         |      | schema validation    |
+---------+----------+      +----------+-----------+
          |                            |
          v                            v
+--------------------+      +----------------------+
| create_client()    |      | apply_runtime_inputs |
| dry-run/Claude/API |      | prompt + variables   |
+---------+----------+      +----------+-----------+
          |                            |
          +-------------+--------------+
                        v
              +------------------+
              | run_crew()       |
              | engine loop      |
              +--------+---------+
                       |
        +--------------+---------------+
        |              |               |
        v              v               v
+---------------+ +--------------+ +----------------+
| _call_agent() | | _score_agents| | GoalTracker    |
| LLM turn      | | cut-in score | | convergence    |
+-------+-------+ +------+-------+ +-------+--------+
        |                |                 |
        +----------------+-----------------+
                         v
                  +--------------+
                  | save_result()|
                  | json + txt   |
                  +--------------+
```

## Mental Model

```text
+--------------------------------------------------------------------------------+
|                               ONE SIMULATION RUN                               |
+--------------------------------------------------------------------------------+
|                                                                                |
|  PRIVATE MINDS                           PUBLIC ROOM                           |
|  each agent has its own state            only spoken turns are visible          |
|                                                                                |
|  +----------------------+                +----------------------------------+   |
|  | Mara memory          |                | group transcript                  |   |
|  | - private thinking   |                | - says                            |   |
|  | - stayed quiet notes |                | - cuts in                         |   |
|  +----------+-----------+                | - final PRD                       |   |
|             |                            +----------------+-----------------+   |
|  +----------v-----------+                                 |                     |
|  | Niko memory          |                                 v                     |
|  | - private thinking   |                +----------------------------------+   |
|  | - stayed quiet notes |                | scheduler                         |   |
|  +----------+-----------+                | who has reason to speak now?      |   |
|             |                            +----------------+-----------------+   |
|  +----------v-----------+                                 |                     |
|  | Sol / June memory    |                                 v                     |
|  +----------------------+                +----------------------------------+   |
|                                          | interruption classifier            |   |
|                                          | should this be a normal turn       |   |
|                                          | or a cut-in?                       |   |
|                                          +----------------+-----------------+   |
|                                                           |                     |
|                                                           v                     |
|                                          +----------------------------------+   |
|                                          | convergence evaluator             |   |
|                                          | are all agents aligned enough      |   |
|                                          | to stop and write the PRD?         |   |
|                                          +----------------------------------+   |
|                                                                                |
+--------------------------------------------------------------------------------+
```

Private thinking is visible to the human in the terminal so you can inspect the run. It is not visible to the other agents. During group chat, each agent only sees the public transcript plus its own private memory.

## Parallel Thinking

The independent thinking phase is parallelized because each private thought depends only on the topic and the agent's own persona. This makes Claude runs faster without changing what agents know.

```text
+-------------------+   +-------------------+   +-------------------+
| Mara private call |   | Niko private call |   | Sol private call  |
+---------+---------+   +---------+---------+   +---------+---------+
          |                       |                       |
          +-----------+-----------+-----------+-----------+
                      v                       v
             ordered private round     same-agent memory
```

In live mode, SimulaCrew prints a thinking line for every agent as soon as that private call starts:

```text
░ Mara thinking (turn 1)
░ Niko thinking (turn 2)
░ Sol thinking (turn 3)
░ June thinking (turn 4)
```

Those private calls may finish in a different order, but the saved private round is kept in configured agent order.

Configure the worker count in `configs/simulacra.json`:

```json
{
  "topic": {
    "variables": {
      "private_thinking_workers": 4
    }
  }
}
```

The group chat still runs turn by turn because each message depends on the public transcript so far.

## State Flow

```text
+------------------+       +------------------+       +------------------+
| Persona Fields   |       | Conversation     |       | Hidden State     |
+------------------+       +------------------+       +------------------+
| base_prompt      |       | public turns     |       | private thoughts |
| backstory        | ----> | says/cuts in     | ----> | interrupt scores |
| skills           |       | questions        |       | convergence      |
| interests        |       | objections       |       | LLM rationales   |
| history          |       | concessions      |       | agent alignment  |
| knowledge        |       | proposal merges  |       | PRD inputs       |
| speaking_style   |       | final decision   |       |                  |
+------------------+       +------------------+       +------------------+
```

Private state is intentionally split from the public transcript:

```text
+--------------------------+       +----------------------------+
| Agent's own private turn | ----> | same agent's private_memory|
+--------------------------+       +----------------------------+
              |
              v
       not included in
       public group transcript
```

That means Niko can use Niko's own earlier thinking when he speaks, but Mara cannot see Niko's private notes unless Niko says the idea out loud.

Possible group-chat moves are configured in the preset prompt and harness rules:

- ask a clarifying question,
- answer the prior speaker,
- challenge a weak assumption,
- concede a point,
- merge two ideas,
- propose a concrete project direction,
- stay quiet and keep a private note for a later turn,
- cut in when the persona would realistically interrupt.

The agents are instructed to act like a small working group, not like essay writers. Normal turns should be one to three lines. Internal scores, classifier rationales, and private thoughts are hidden from the CLI by default so the transcript reads like a group chat.

Live CLI output is intentionally shaped like the [`jurysim`](https://github.com/prashaantr/jurysim) room transcript:

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

... Sol thinking (turn 3)
Sol  stays quiet  15:46:21

... Mara cuts in (turn 4)
Mara  cuts in  15:46:25
| Wait, we are turning this into architecture talk again. What is the
| smallest harness that lets us run one moral scenario today?
```

Internal interruption scores and classifier notes are hidden by default because they make the run feel less like a real group chat. To debug them:

```bash
simulacrew run configs/simulacra.json --show-interruption-notes
```

The normal CLI shows buy-in and current idea state after public group-chat turns:

```text
◇ buy-in  june 43%  mara 56%  niko 37%  sol 45%
◇ idea    voice-first job-skills matcher for displaced workers
```

With `--show-interruption-notes`, the CLI also prints each agent's internal view of what the idea is. This helps catch cases where agents appear to agree but are actually imagining different products.

## How The Idea And Percentages Work

SimulaCrew does not pick the idea by taking the last sentence someone said. The intended model is closer to a real group:

```text
+------------------+       +------------------+       +------------------+
| agent idea view  |       | buy-in score     |       | public transcript|
+------------------+       +------------------+       +------------------+
| Mara's concept   |       | Mara: 0.56       |       | what was said    |
| Niko's concept   | ----> | Niko: 0.37       | ----> | questions        |
| Sol's concept    |       | Sol: 0.45        |       | objections       |
| June's concept   |       | June: 0.43       |       | concessions      |
+------------------+       +------------------+       +------------------+
          |                         |                         |
          +-------------------------+-------------------------+
                                    v
                         +----------------------+
                         | convergence evaluator|
                         | updates room state   |
                         +----------+-----------+
                                    |
                                    v
                         +----------------------+
                         | shared current idea  |
                         | PRD when aligned     |
                         +----------------------+
```

Each agent has two pieces of internal state:

- `agent idea view`: what that agent currently thinks the proposal is.
- `buy-in score`: how bought in that agent is to moving forward with the shared goal and emerging idea.

The CLI line:

```text
◇ buy-in  mara 56%  niko 37%  sol 45%  june 43%
◇ idea    voice-first job-skills matcher for displaced workers
```

means:

- Mara is 56% bought in to the current direction.
- Niko is 37% bought in, so he is likely to keep contesting or asking for proof.
- The `idea` line is the evaluator's current best read of the shared proposal in the room.

With `--show-interruption-notes`, the CLI can also show each agent's view:

```text
◇ agent idea views
  mara: voice-first job-skills matcher for informal workers
  niko: skills matcher, but only valid if job data is grounded
  sol: worker mobility tool that translates experience into next roles
  june: simple demo where a worker speaks history and sees next steps
```

The final idea is decided when the group reaches convergence:

```text
all agents buy-in >= goal_alignment_threshold
AND public turns >= goal_alignment_min_turns
```

If the 20-minute discussion cap hits first, the recorder writes the PRD from the best available shared idea and preserves unresolved disagreement.

### Live Claude Runs

When running with Claude and `goal_alignment_evaluator: "llm"`, the evaluator receives:

- the shared goal,
- all active personas,
- current buy-in scores,
- the latest public statement,
- the public transcript.

It returns compact JSON:

```json
{
  "by_agent": {
    "mara": 0.56,
    "niko": 0.37,
    "sol": 0.45,
    "june": 0.43
  },
  "current_idea": "voice-first job-skills matcher for displaced workers",
  "agent_views": {
    "mara": "fast demo for displaced workers",
    "niko": "skills matcher needing grounded job data",
    "sol": "mobility tool connecting experience to next roles",
    "june": "simple worker-facing demo"
  },
  "aligned": false,
  "rationale": "The group has a candidate idea, but Niko still needs evidence quality resolved."
}
```

Those values become the next turn's private context and the CLI buy-in display.

### Dry-Run Mode

Dry-run mode does not call an LLM evaluator. It uses deterministic fallback values so tests and demos can run without an API key:

- the speaker's buy-in increases by `goal_alignment_step_self`,
- listeners increase by `goal_alignment_step_listener`,
- interruptions are damped by `goal_alignment_interrupt_factor`,
- the visible `idea` line is a simple placeholder derived from the latest public statement.

Dry-run percentages are useful for testing the mechanics. For realistic idea tracking, use Claude with the LLM convergence evaluator.

## Convergence And The 20-Minute Cap

The default task is:

```text
Converge on one concrete hackathon project idea and then produce a useful PRD for building it.
```

This is configured in `configs/simulacra.json` under `topic.variables`:

```json
{
  "shared_goal": "Converge on one concrete hackathon project idea and then produce a useful PRD for building it.",
  "discussion_time_limit_minutes": 20,
  "goal_alignment_threshold": 0.86,
  "goal_alignment_min_turns": 8,
  "goal_alignment_evaluator": "llm",
  "default_goal_alignment_start": 0.42,
  "goal_alignment_step_self": 0.055,
  "goal_alignment_step_listener": 0.025,
  "goal_alignment_interrupt_factor": 0.65,
  "min_public_turns_before_interruptions": 3
}
```

The discussion stops when either:

- every active agent reaches the configured alignment threshold after the minimum number of public turns, or
- the 20-minute discussion cap is reached between turns.

Convergence is not based on hard-coded words like "agree" or "risk." In live Claude runs, the LLM evaluator reads the transcript, personas, shared goal, and current state, then returns per-agent alignment as JSON. In dry-run mode, SimulaCrew uses the configured fallback step sizes so local tests can run without an API key.

Cut-ins are delayed by `min_public_turns_before_interruptions` so the first few group-chat turns build on what was said before. After that, high-interruption personas can cut in when the classifier says there is enough contestation pressure.

Per-agent starting alignment goes in `agents[].personality.goal_alignment_start`:

```json
{
  "id": "niko",
  "personality": {
    "skepticism": 9,
    "interruptiveness": 7,
    "goal_alignment_start": 0.34
  }
}
```

The alignment state is written into the JSON run metadata for debugging. It is not printed in the normal CLI transcript unless you pass `--show-interruption-notes`.

## Where Personalities Go

Personality, backstory, skills, interests, history, knowledge, and speaking style live in the preset under each item in `agents[]`.

```json
{
  "id": "niko",
  "name": "Niko",
  "base_prompt": "You are a careful critic. You look for failure modes.",
  "backstory": "Evaluation-minded researcher who has seen polished demos fail.",
  "speaking_style": "Precise, skeptical, calm but firm.",
  "knowledge": ["evaluation harnesses", "failure mode analysis"],
  "skills": ["test design", "red-team critique", "rubric construction"],
  "interests": ["observable mechanisms", "simulation validity"],
  "history": ["has reviewed agent demos that looked realistic but had no inspection trail"],
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
    "patience": 4,
    "goal_alignment_start": 0.34
  },
  "goals": ["prevent shallow simulation", "force testable claims"],
  "constraints": ["must offer a replacement when blocking an idea"]
}
```

The harness turns those fields into private character prompts with `harness.character_prompt_template`. This is the place to put agent creation instructions. If you want different characters, edit the agent objects and the template in `configs/simulacra.json`.

Skills, interests, and past experience are considered through the character prompt, not through a separate hard-coded rule. They affect:

- what the agent notices in the topic,
- which ideas they are naturally drawn toward,
- what kinds of objections they raise,
- when they feel credible enough to cut in,
- what private thoughts they keep when staying quiet,
- what they contribute to the final convergence.

For example, an agent with CLI usability skills and a history of confusing setup failures will push harder on install commands and live terminal feedback. An agent interested in evaluation validity will push harder on traceability, test harnesses, and failure modes.

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
Statement in group chat
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

The scheduler works like this during group chat:

1. Score every agent against the current transcript.
2. Add pressure for assertiveness, skepticism, urgency, interruptiveness, and disagreement sensitivity.
3. Add pressure for agents who have been quiet recently.
4. Reduce pressure for patience, agreeableness, repeated recent speaking, and the last speaker.
5. Pick the next speaker dynamically. If the score is high enough, the turn is rendered as a cut-in.
6. If a quieter persona decides not to speak, SimulaCrew records a private thought and feeds it back into that agent's later turns.

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

## PRD Output

The final round is a recorder round. It receives:

- the full public transcript,
- the hidden private notes that agents kept when they stayed quiet,
- the final convergence state,
- the configured `harness.output_contract`.

The default output contract asks for:

- product idea,
- target user,
- problem,
- proposed solution,
- core user flow,
- MVP scope,
- out of scope,
- agent/personality mechanics if relevant,
- risks,
- open questions,
- immediate build plan.

Edit `harness.output_contract` in `configs/simulacra.json` to change the final artifact.

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
