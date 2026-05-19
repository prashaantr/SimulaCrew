"use client";

import { FormEvent, useMemo, useState } from "react";

type Statement = {
  agent_id: string;
  agent_name: string;
  round_id: string;
  turn_number: number | string;
  event_type: string;
  content: string;
  metadata?: Record<string, unknown>;
};

type RoundResult = {
  id: string;
  title: string;
  mode: string;
  statements: Statement[];
};

type SimulationResult = {
  experiment_name: string;
  task_title: string;
  task_prompt: string;
  provider: string;
  model: string;
  task_result: string;
  raw_task_result: string;
  rounds: RoundResult[];
  transcript: Statement[];
  format_check: {
    valid: boolean;
    errors: string[];
  };
};

type ApiResponse = {
  result: SimulationResult;
  artifacts: {
    json: string;
    conversation: string | null;
  };
};

type Provider = "dry-run" | "claude" | "openai";

const starterPrompt =
  "What project should we build for a future-of-work hackathon?";

const agentColors = [
  "#ff385c",
  "#30d5a5",
  "#7cc4ff",
  "#ffb86b",
  "#b58cff",
  "#f973a9"
];

export function ChatWorkspace() {
  const [prompt, setPrompt] = useState(starterPrompt);
  const [provider, setProvider] = useState<Provider>("dry-run");
  const [maxAgents, setMaxAgents] = useState(4);
  const [result, setResult] = useState<SimulationResult | null>(null);
  const [artifacts, setArtifacts] = useState<ApiResponse["artifacts"] | null>(null);
  const [isRunning, setIsRunning] = useState(false);
  const [error, setError] = useState("");

  const messages = useMemo(() => flattenRounds(result), [result]);
  const roomState = useMemo(() => latestRoomState(messages), [messages]);
  const agents = useMemo(() => agentSummaries(messages), [messages]);

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const text = prompt.trim();
    if (!text || isRunning) {
      return;
    }
    setIsRunning(true);
    setError("");
    setArtifacts(null);
    try {
      const response = await fetch("/api/simulate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          prompt: text,
          provider,
          maxAgents
        })
      });
      const payload = await response.json();
      if (!response.ok) {
        throw new Error(payload.error || "Simulation failed.");
      }
      setResult(payload.result);
      setArtifacts(payload.artifacts);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Simulation failed.");
    } finally {
      setIsRunning(false);
    }
  }

  return (
    <div className="workspace-grid">
      <section className="chat-card" aria-label="SimulaCrew group chat">
        <header className="topbar">
          <div>
            <p className="eyebrow">SimulaCrew</p>
            <h1>Group Chat</h1>
          </div>
          <div className="status-pill">
            <span className={isRunning ? "status-dot running" : "status-dot"} />
            {isRunning ? "running" : result ? "complete" : "ready"}
          </div>
        </header>

        <div className="conversation">
          {!result && (
            <div className="empty-state">
              <div className="empty-avatar">S</div>
              <div className="empty-copy">
                <strong>Start a crew run</strong>
                <span>The agents will deliberate here as a terminal group chat.</span>
              </div>
            </div>
          )}

          {result && (
            <div className="prompt-banner">
              <span>You</span>
              <p>{result.task_prompt}</p>
            </div>
          )}

          {messages.map((message, index) => (
            <MessageBubble
              key={`${message.round_id}-${message.agent_id}-${message.turn_number}-${index}`}
              statement={message}
              color={colorFor(message.agent_id)}
            />
          ))}

          {isRunning && (
            <div className="typing-row">
              <span />
              <span />
              <span />
            </div>
          )}
        </div>

        {error && <div className="error">{error}</div>}

        <form className="composer" onSubmit={submit}>
          <textarea
            value={prompt}
            onChange={(event) => setPrompt(event.target.value)}
            rows={2}
            placeholder="Ask the crew what to build..."
          />
          <div className="composer-actions">
            <label>
              Provider
              <select
                value={provider}
                onChange={(event) => setProvider(event.target.value as Provider)}
              >
                <option value="dry-run">dry-run</option>
                <option value="claude">claude</option>
                <option value="openai">openai</option>
              </select>
            </label>
            <label>
              Agents
              <input
                type="number"
                min={1}
                max={8}
                value={maxAgents}
                onChange={(event) => setMaxAgents(Number(event.target.value))}
              />
            </label>
            <button type="submit" disabled={isRunning || !prompt.trim()}>
              {isRunning ? "Running" : "Run"}
            </button>
          </div>
        </form>
      </section>

      <aside className="side-panel" aria-label="Room state">
        <section className="panel-section">
          <p className="eyebrow">Room State</p>
          <h2>{roomState.idea || "No idea yet"}</h2>
          <div className="buyin-list">
            {roomState.buyIn.length ? (
              roomState.buyIn.map((item) => (
                <div className="buyin-row" key={item.agent}>
                  <span>{displayName(item.agent)}</span>
                  <div className="meter">
                    <i style={{ width: `${item.score}%` }} />
                  </div>
                  <b>{item.score}%</b>
                </div>
              ))
            ) : (
              <p className="muted">Room state appears after the first public turn.</p>
            )}
          </div>
        </section>

        <section className="panel-section">
          <p className="eyebrow">Agents</p>
          <div className="agent-list">
            {agents.length ? (
              agents.map((agent) => (
                <div className="agent-row" key={agent.id}>
                  <span
                    className="agent-mark"
                    style={{ backgroundColor: colorFor(agent.id) }}
                  >
                    {initials(agent.name)}
                  </span>
                  <div>
                    <strong>{agent.name}</strong>
                    <small>{agent.count} turns</small>
                  </div>
                </div>
              ))
            ) : (
              <p className="muted">Agents appear after a run starts.</p>
            )}
          </div>
        </section>

        {artifacts && (
          <section className="panel-section">
            <p className="eyebrow">Artifacts</p>
            <code>{artifacts.json}</code>
            {artifacts.conversation && <code>{artifacts.conversation}</code>}
          </section>
        )}

        {result?.raw_task_result && (
          <section className="panel-section final-output">
            <p className="eyebrow">Final PRD</p>
            <pre>{result.raw_task_result}</pre>
          </section>
        )}
      </aside>
    </div>
  );
}

function MessageBubble({
  statement,
  color
}: {
  statement: Statement;
  color: string;
}) {
  const quiet = statement.event_type === "thought";
  const privateTurn = statement.event_type === "private";
  const synthesis = statement.event_type === "synthesis";
  return (
    <article
      className={[
        "message-row",
        privateTurn ? "private" : "",
        quiet ? "quiet" : "",
        synthesis ? "synthesis" : ""
      ].join(" ")}
    >
      <div className="avatar" style={{ backgroundColor: color }}>
        {initials(statement.agent_name)}
      </div>
      <div className="bubble">
        <div className="message-meta">
          <strong>{statement.agent_name}</strong>
          <span>{labelFor(statement.event_type)}</span>
        </div>
        <p>{quiet ? "stays quiet" : statement.content}</p>
      </div>
    </article>
  );
}

function flattenRounds(result: SimulationResult | null) {
  if (!result) {
    return [];
  }
  return result.rounds.flatMap((round) => round.statements);
}

function latestRoomState(messages: Statement[]) {
  const withState = [...messages]
    .reverse()
    .find((statement) => statement.metadata?.goal_alignment_by_agent);
  const metadata = withState?.metadata || {};
  const scores = metadata.goal_alignment_by_agent as Record<string, number> | undefined;
  const views = metadata.agent_idea_views as Record<string, string> | undefined;
  return {
    idea: String(metadata.current_idea || ""),
    views: views || {},
    buyIn: scores
      ? Object.entries(scores).map(([agent, score]) => ({
          agent,
          score: Math.round(score * 100)
        }))
      : []
  };
}

function agentSummaries(messages: Statement[]) {
  const seen = new Map<string, { id: string; name: string; count: number }>();
  for (const message of messages) {
    const item = seen.get(message.agent_id) || {
      id: message.agent_id,
      name: message.agent_name,
      count: 0
    };
    item.count += 1;
    seen.set(message.agent_id, item);
  }
  return Array.from(seen.values());
}

function colorFor(agentId: string) {
  let total = 0;
  for (const character of agentId) {
    total += character.charCodeAt(0);
  }
  return agentColors[total % agentColors.length];
}

function initials(name: string) {
  return name
    .split(/\s+/)
    .filter(Boolean)
    .slice(0, 2)
    .map((part) => part[0]?.toUpperCase())
    .join("");
}

function displayName(agentId: string) {
  return agentId.replace(/[-_]/g, " ").replace(/\b\w/g, (char) => char.toUpperCase());
}

function labelFor(eventType: string) {
  if (eventType === "private") {
    return "thinking";
  }
  if (eventType === "interrupt") {
    return "cuts in";
  }
  if (eventType === "synthesis") {
    return "wraps up";
  }
  if (eventType === "thought") {
    return "quiet";
  }
  return "says";
}
