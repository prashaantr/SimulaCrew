"use client";

import { useEffect, useMemo, useState } from "react";

type BuyIn = { agent: string; score: number };

type AgentSummary = { id: string; name: string; publicTurns: number };

type TranscriptEntry = {
  agentId: string;
  agentName: string;
  eventType: string;
  content: string;
};

type RunSummary = {
  file: string;
  team: string | null;
  experimentName: string;
  taskTitle: string;
  taskPrompt: string;
  startedAt: string;
  provider: string;
  model: string;
  formatCheck: { valid: boolean; errors: string[]; formatType: string };
  finalIdea: string;
  buyIn: BuyIn[];
  agents: AgentSummary[];
  totalPublicTurns: number;
  roundCount: number;
  rawTaskResult: string;
  transcript: TranscriptEntry[];
};

type ApiResponse = {
  runs: RunSummary[];
  source: string;
  missing: boolean;
  errors?: { file: string; error: string }[];
};

type TeamInfo = {
  team_number: number;
  members: string[];
  project?: string;
};

type TeamApiResponse = {
  teams: TeamInfo[];
};

const agentColors = [
  "#ff385c",
  "#30d5a5",
  "#7cc4ff",
  "#ffb86b",
  "#b58cff",
  "#f973a9"
];

export function RunsList() {
  const [runs, setRuns] = useState<RunSummary[]>([]);
  const [teamsByNumber, setTeamsByNumber] = useState<Map<number, TeamInfo>>(new Map());
  const [source, setSource] = useState("");
  const [missing, setMissing] = useState(false);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [expanded, setExpanded] = useState<string | null>(null);
  const [chatExpanded, setChatExpanded] = useState<string | null>(null);
  const [providerFilter, setProviderFilter] = useState<string>("all");
  const [teamFilter, setTeamFilter] = useState<string>("all");

  useEffect(() => {
    let cancelled = false;
    Promise.all([
      fetch("/api/runs").then((response) => response.json()),
      fetch("/api/teams")
        .then((response) => response.json())
        .catch(() => ({ teams: [] }))
    ])
      .then(([runsPayload, teamsPayload]: [ApiResponse, TeamApiResponse]) => {
        if (cancelled) {
          return;
        }
        setRuns(Array.isArray(runsPayload.runs) ? runsPayload.runs : []);
        setSource(String(runsPayload.source || ""));
        setMissing(Boolean(runsPayload.missing));
        const teamMap = new Map<number, TeamInfo>();
        for (const team of Array.isArray(teamsPayload?.teams) ? teamsPayload.teams : []) {
          if (typeof team?.team_number === "number") {
            teamMap.set(team.team_number, team);
          }
        }
        setTeamsByNumber(teamMap);
        setLoading(false);
      })
      .catch((caught) => {
        if (cancelled) {
          return;
        }
        setError(caught instanceof Error ? caught.message : "Failed to load runs.");
        setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const providers = useMemo(() => {
    const found = new Set<string>();
    for (const run of runs) {
      if (run.provider) {
        found.add(run.provider);
      }
    }
    return Array.from(found).sort();
  }, [runs]);

  const teams = useMemo(() => {
    const found = new Set<string>();
    for (const run of runs) {
      if (run.team) {
        found.add(run.team);
      }
    }
    return Array.from(found).sort(naturalCompare);
  }, [runs]);

  const visible = useMemo(() => {
    return runs.filter((run) => {
      if (providerFilter !== "all" && run.provider !== providerFilter) {
        return false;
      }
      if (teamFilter !== "all" && run.team !== teamFilter) {
        return false;
      }
      return true;
    });
  }, [providerFilter, teamFilter, runs]);

  const aggregate = useMemo(() => {
    if (!visible.length) {
      return null;
    }
    const totalTurns = visible.reduce((sum, run) => sum + run.totalPublicTurns, 0);
    const validCount = visible.filter((run) => run.formatCheck.valid).length;
    const averageBuyIn =
      visible.reduce((sum, run) => {
        if (!run.buyIn.length) {
          return sum;
        }
        const mean =
          run.buyIn.reduce((acc, item) => acc + item.score, 0) / run.buyIn.length;
        return sum + mean;
      }, 0) / Math.max(1, visible.filter((run) => run.buyIn.length).length);
    return {
      totalRuns: visible.length,
      totalTurns,
      validCount,
      averageBuyIn: Number.isFinite(averageBuyIn) ? Math.round(averageBuyIn) : 0
    };
  }, [visible]);

  return (
    <section className="runs-dashboard" aria-label="Simulation run outcomes">
      <div className="dashboard-heading">
        <div>
          <p className="eyebrow">Run Outcomes</p>
          <h1>Recent Runs</h1>
        </div>
        <div className="run-filters">
          {teams.length > 0 && (
            <select
              value={teamFilter}
              onChange={(event) => setTeamFilter(event.target.value)}
            >
              <option value="all">All teams</option>
              {teams.map((team) => (
                <option key={team} value={team}>
                  {team}
                </option>
              ))}
            </select>
          )}
          <select
            value={providerFilter}
            onChange={(event) => setProviderFilter(event.target.value)}
          >
            <option value="all">All providers</option>
            {providers.map((provider) => (
              <option key={provider} value={provider}>
                {provider}
              </option>
            ))}
          </select>
        </div>
      </div>

      {aggregate && (
        <div className="run-aggregate">
          <Stat label="Runs" value={String(aggregate.totalRuns)} />
          <Stat label="Public turns" value={String(aggregate.totalTurns)} />
          <Stat
            label="Format-check pass"
            value={`${aggregate.validCount}/${aggregate.totalRuns}`}
          />
          <Stat label="Avg buy-in" value={`${aggregate.averageBuyIn}%`} />
        </div>
      )}

      {loading && <p className="muted">Loading runs…</p>}
      {error && <div className="error">{error}</div>}

      {!loading && missing && (
        <div className="team-empty">
          <strong>No runs directory yet</strong>
          <span>
            Run the chat workspace once or point SIMULACREW_RUNS_DIR at an existing
            artifact directory.
          </span>
          {source && <code>{source}</code>}
        </div>
      )}

      {!loading && !missing && visible.length === 0 && (
        <div className="team-empty">
          <strong>No runs match this filter</strong>
          <span>Run a simulation in /chat or clear the provider filter.</span>
          {source && <code>{source}</code>}
        </div>
      )}

      <div className="run-grid">
        {visible.map((run) => (
          <RunCard
            key={run.file}
            run={run}
            teamInfo={lookupTeam(run.team, teamsByNumber)}
            expanded={expanded === run.file}
            chatExpanded={chatExpanded === run.file}
            onToggle={() =>
              setExpanded((current) => (current === run.file ? null : run.file))
            }
            onToggleChat={() =>
              setChatExpanded((current) =>
                current === run.file ? null : run.file
              )
            }
          />
        ))}
      </div>
    </section>
  );
}

function RunCard({
  run,
  teamInfo,
  expanded,
  chatExpanded,
  onToggle,
  onToggleChat
}: {
  run: RunSummary;
  teamInfo: TeamInfo | null;
  expanded: boolean;
  chatExpanded: boolean;
  onToggle: () => void;
  onToggleChat: () => void;
}) {
  const subtitle = formatTimestamp(run.startedAt);
  const members = teamInfo?.members ?? [];
  return (
    <article className={`run-card${expanded ? " expanded" : ""}`}>
      <header className="run-card-top">
        <div>
          <div className="run-card-tags">
            {run.team && <span className="run-team">{run.team}</span>}
            <span className="run-experiment">{run.experimentName}</span>
          </div>
          <small>{subtitle}</small>
        </div>
        <span
          className={`format-pill ${run.formatCheck.valid ? "ok" : "bad"}`}
          title={
            run.formatCheck.valid
              ? "Final output matched the declared format"
              : run.formatCheck.errors.join("; ") || "Format check failed"
          }
        >
          {run.formatCheck.valid ? "format ok" : "format fail"}
          <small>{run.formatCheck.formatType || "—"}</small>
        </span>
      </header>

      {members.length > 0 ? (
        <div className="member-chips run-members">
          {members.map((member) => (
            <span key={member}>{member}</span>
          ))}
        </div>
      ) : (
        <h2 className="run-title">{run.finalIdea || "No converged idea"}</h2>
      )}

      {run.rawTaskResult ? (
        <pre className="run-prd">{run.rawTaskResult}</pre>
      ) : (
        <p className="muted">No final artifact recorded.</p>
      )}

      <footer className="run-card-foot">
        <div className="run-card-actions">
          <button type="button" onClick={onToggle}>
            {expanded ? "Hide context" : "Show context"}
          </button>
          <button
            type="button"
            onClick={onToggleChat}
            disabled={!run.transcript.length}
            title={
              run.transcript.length
                ? undefined
                : "No public turns recorded for this run"
            }
          >
            {chatExpanded ? "Hide chat" : "Show chat"}
          </button>
        </div>
        <code>{run.file}</code>
      </footer>

      {chatExpanded && run.transcript.length > 0 && (
        <div className="run-chat">
          {run.transcript.map((entry, index) => (
            <article
              key={`${entry.agentId}-${index}`}
              className={`chat-line ${
                entry.eventType === "interrupt"
                  ? "interrupt"
                  : entry.eventType === "synthesis"
                    ? "synthesis"
                    : ""
              }`}
            >
              <span
                className="chat-avatar"
                style={{ backgroundColor: colorFor(entry.agentId) }}
              >
                {initials(entry.agentName)}
              </span>
              <div className="chat-bubble">
                <div className="chat-meta">
                  <strong>{entry.agentName}</strong>
                  <span>{labelFor(entry.eventType)}</span>
                </div>
                <p>{entry.content}</p>
              </div>
            </article>
          ))}
        </div>
      )}

      {expanded && (
        <div className="run-context">
          <p className="run-prompt">{run.taskPrompt}</p>

          <dl className="run-stats">
            <div>
              <dt>Provider</dt>
              <dd>{run.provider}</dd>
            </div>
            <div>
              <dt>Model</dt>
              <dd>{run.model || "—"}</dd>
            </div>
            <div>
              <dt>Public turns</dt>
              <dd>{run.totalPublicTurns}</dd>
            </div>
            <div>
              <dt>Rounds</dt>
              <dd>{run.roundCount}</dd>
            </div>
          </dl>

          <div className="run-section">
            <p className="eyebrow">Buy-in at finish</p>
            <div className="buyin-list">
              {run.buyIn.length ? (
                run.buyIn.map((item) => (
                  <div className="buyin-row" key={item.agent}>
                    <span>{displayName(item.agent)}</span>
                    <div className="meter">
                      <i
                        style={{
                          width: `${Math.max(0, Math.min(100, item.score))}%`,
                          background: colorFor(item.agent)
                        }}
                      />
                    </div>
                    <b>{item.score}%</b>
                  </div>
                ))
              ) : (
                <p className="muted">No buy-in scores recorded.</p>
              )}
            </div>
          </div>

          <div className="run-section">
            <p className="eyebrow">Agent participation</p>
            <div className="agent-tags">
              {run.agents.length ? (
                run.agents
                  .slice()
                  .sort((a, b) => b.publicTurns - a.publicTurns)
                  .map((agent) => (
                    <span
                      key={agent.id}
                      className="agent-tag"
                      style={{ borderColor: colorFor(agent.id) }}
                    >
                      <i style={{ backgroundColor: colorFor(agent.id) }} />
                      {agent.name}
                      <b>{agent.publicTurns}</b>
                    </span>
                  ))
              ) : (
                <p className="muted">No public turns recorded.</p>
              )}
            </div>
          </div>

          {run.formatCheck.errors.length > 0 && (
            <div className="run-section">
              <p className="eyebrow">Format errors</p>
              <ul className="format-errors">
                {run.formatCheck.errors.map((err, index) => (
                  <li key={`${err}-${index}`}>{err}</li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}
    </article>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="run-stat">
      <small>{label}</small>
      <strong>{value}</strong>
    </div>
  );
}

function colorFor(agentId: string) {
  let total = 0;
  for (const character of agentId) {
    total += character.charCodeAt(0);
  }
  return agentColors[total % agentColors.length];
}

function displayName(agentId: string) {
  return agentId.replace(/[-_]/g, " ").replace(/\b\w/g, (char) => char.toUpperCase());
}

function initials(name: string) {
  return name
    .split(/\s+/)
    .filter(Boolean)
    .slice(0, 2)
    .map((part) => part[0]?.toUpperCase() ?? "")
    .join("");
}

function labelFor(eventType: string) {
  if (eventType === "interrupt") {
    return "cuts in";
  }
  if (eventType === "synthesis") {
    return "wraps up";
  }
  return "says";
}

function lookupTeam(
  teamSlug: string | null,
  teams: Map<number, TeamInfo>
): TeamInfo | null {
  if (!teamSlug) {
    return null;
  }
  const match = teamSlug.match(/(\d+)/);
  if (!match) {
    return null;
  }
  const number = Number(match[1]);
  return teams.get(number) ?? null;
}

function naturalCompare(a: string, b: string) {
  return a.localeCompare(b, undefined, { numeric: true, sensitivity: "base" });
}

function formatTimestamp(value: string) {
  if (!value) {
    return "";
  }
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return value;
  }
  return date.toLocaleString();
}
