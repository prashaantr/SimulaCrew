import { existsSync } from "node:fs";
import { readFile } from "node:fs/promises";
import path from "node:path";
import { NextResponse } from "next/server";

export const runtime = "nodejs";

const DEFAULT_TEAM_INFO_PATH = path.join(
  process.cwd(),
  "data",
  "buildathon_team_info.json"
);

export async function GET() {
  const configuredPath =
    process.env.SIMULACREW_TEAM_INFO_PATH || DEFAULT_TEAM_INFO_PATH;
  const teamInfoPath = path.resolve(configuredPath);

  if (!existsSync(teamInfoPath)) {
    return NextResponse.json({
      teams: [],
      source: teamInfoPath,
      missing: true
    });
  }

  try {
    const raw = await readFile(teamInfoPath, "utf-8");
    const parsed = JSON.parse(raw);
    return NextResponse.json({
      teams: Array.isArray(parsed.teams) ? parsed.teams : [],
      source: teamInfoPath,
      missing: false
    });
  } catch (error) {
    return NextResponse.json(
      {
        error: error instanceof Error ? error.message : "Unable to read team info.",
        teams: [],
        source: teamInfoPath,
        missing: false
      },
      { status: 500 }
    );
  }
}
