"use client";

import { useEffect, useMemo, useState } from "react";

type TeamInfo = {
  team_number: number;
  members: string[];
  project?: string;
  notes?: string;
  links?: Record<string, string>;
  resources?: { name: string; url: string }[];
};

export function TeamDashboard() {
  const [teams, setTeams] = useState<TeamInfo[]>([]);
  const [selectedTeam, setSelectedTeam] = useState<number | "all">("all");
  const [teamSource, setTeamSource] = useState("");

  const filteredTeams = useMemo(() => {
    if (selectedTeam === "all") {
      return teams;
    }
    return teams.filter((team) => team.team_number === selectedTeam);
  }, [selectedTeam, teams]);

  useEffect(() => {
    let cancelled = false;
    fetch("/api/teams")
      .then((response) => response.json())
      .then((payload) => {
        if (cancelled) {
          return;
        }
        setTeams(Array.isArray(payload.teams) ? payload.teams : []);
        setTeamSource(String(payload.source || ""));
      })
      .catch(() => {
        if (!cancelled) {
          setTeams([]);
        }
      });
    return () => {
      cancelled = true;
    };
  }, []);

  return (
    <section className="team-dashboard" aria-label="Buildathon team dashboard">
      <div className="dashboard-heading">
        <div>
          <p className="eyebrow">Buildathon Teams</p>
          <h1>Team Dashboard</h1>
        </div>
        <select
          value={selectedTeam}
          onChange={(event) =>
            setSelectedTeam(
              event.target.value === "all" ? "all" : Number(event.target.value)
            )
          }
        >
          <option value="all">All teams</option>
          {teams.map((team) => (
            <option key={team.team_number} value={team.team_number}>
              Team {team.team_number}
            </option>
          ))}
        </select>
      </div>
      <div className="team-grid">
        {filteredTeams.length ? (
          filteredTeams.map((team) => <TeamCard key={team.team_number} team={team} />)
        ) : (
          <div className="team-empty">
            <strong>No team data loaded</strong>
            <span>
              Set SIMULACREW_TEAM_INFO_PATH or place the local JSON at the configured
              path.
            </span>
            {teamSource && <code>{teamSource}</code>}
          </div>
        )}
      </div>
    </section>
  );
}

function TeamCard({ team }: { team: TeamInfo }) {
  const links = Object.entries(team.links || {});
  const resources = team.resources || [];
  return (
    <article className="team-card">
      <div className="team-card-top">
        <span>Team {team.team_number}</span>
        <b>{team.members.length} members</b>
      </div>
      <h2>{team.project || "Untitled project"}</h2>
      {team.notes && <p className="team-notes">{team.notes}</p>}
      <div className="member-chips">
        {team.members.map((member) => (
          <span key={member}>{member}</span>
        ))}
      </div>
      {(links.length > 0 || resources.length > 0) && (
        <div className="team-links">
          {links.map(([name, url]) => (
            <a href={url} key={name} rel="noreferrer" target="_blank">
              {name.replace(/_/g, " ")}
            </a>
          ))}
          {resources.map((resource) => (
            <a href={resource.url} key={resource.url} rel="noreferrer" target="_blank">
              {resource.name}
            </a>
          ))}
        </div>
      )}
    </article>
  );
}
