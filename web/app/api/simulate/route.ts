import { spawn } from "node:child_process";
import { existsSync } from "node:fs";
import { readFile } from "node:fs/promises";
import path from "node:path";
import { NextRequest, NextResponse } from "next/server";

export const runtime = "nodejs";
export const maxDuration = 300;

type SimulateRequest = {
  prompt?: string;
  provider?: "dry-run" | "claude" | "openai";
  experiment?: string;
  maxAgents?: number;
  model?: string;
};

const DEFAULT_EXPERIMENT = "configs/experiments/simulacra.yaml";

export async function POST(request: NextRequest) {
  const payload = (await request.json()) as SimulateRequest;
  const prompt = String(payload.prompt || "").trim();
  if (!prompt) {
    return NextResponse.json({ error: "Prompt is required." }, { status: 400 });
  }

  const repoRoot = path.resolve(process.cwd(), "..");
  const experiment = safeExperimentPath(
    repoRoot,
    payload.experiment || DEFAULT_EXPERIMENT
  );
  const provider = payload.provider || "dry-run";
  const python = resolvePython(repoRoot);
  const outputDir = path.join(repoRoot, "runs", "web");

  const args = [
    "-m",
    "simula_crew",
    "chat",
    experiment,
    "--message",
    prompt,
    "--provider",
    provider,
    "--quiet",
    "--output-dir",
    outputDir
  ];

  if (payload.model) {
    args.push("--model", payload.model);
  }
  if (payload.maxAgents && Number.isFinite(payload.maxAgents)) {
    args.push("--max-agents", String(Math.max(1, Math.floor(payload.maxAgents))));
  }

  const completed = await runCommand(python, args, repoRoot);
  if (completed.exitCode !== 0) {
    return NextResponse.json(
      {
        error: "Simulation failed.",
        stderr: completed.stderr,
        stdout: completed.stdout
      },
      { status: 500 }
    );
  }

  const artifactPaths = completed.stdout
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean);
  const jsonPath = artifactPaths.find((item) => item.endsWith(".json"));
  if (!jsonPath) {
    return NextResponse.json(
      {
        error: "Simulation did not return a JSON artifact.",
        stdout: completed.stdout
      },
      { status: 500 }
    );
  }

  const absoluteJsonPath = path.isAbsolute(jsonPath)
    ? jsonPath
    : path.join(repoRoot, jsonPath);
  const result = JSON.parse(await readFile(absoluteJsonPath, "utf-8"));

  return NextResponse.json({
    result,
    artifacts: {
      json: jsonPath,
      conversation: artifactPaths.find((item) => item.endsWith(".txt")) || null
    }
  });
}

function resolvePython(repoRoot: string) {
  const venvPython = path.join(repoRoot, ".venv", "bin", "python");
  return existsSync(venvPython) ? venvPython : "python3";
}

function safeExperimentPath(repoRoot: string, requested: string) {
  const normalized = requested.replace(/^\/+/, "");
  const resolved = path.resolve(repoRoot, normalized);
  if (!resolved.startsWith(repoRoot)) {
    return DEFAULT_EXPERIMENT;
  }
  return path.relative(repoRoot, resolved);
}

function runCommand(command: string, args: string[], cwd: string) {
  return new Promise<{
    exitCode: number | null;
    stdout: string;
    stderr: string;
  }>((resolve) => {
    const child = spawn(command, args, {
      cwd,
      env: {
        ...process.env,
        PYTHONPATH: [path.join(cwd, "src"), process.env.PYTHONPATH]
          .filter(Boolean)
          .join(":")
      }
    });
    let stdout = "";
    let stderr = "";
    const timer = setTimeout(() => {
      child.kill("SIGTERM");
      stderr += "\nTimed out after 5 minutes.";
    }, 5 * 60 * 1000);

    child.stdout.on("data", (chunk) => {
      stdout += chunk.toString();
    });
    child.stderr.on("data", (chunk) => {
      stderr += chunk.toString();
    });
    child.on("close", (exitCode) => {
      clearTimeout(timer);
      resolve({ exitCode, stdout, stderr });
    });
  });
}
