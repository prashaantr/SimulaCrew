from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from simula_crew.schema import AgentPersona


NAME_COLUMNS = [
    "Please tell us the name you use in the Hackpad (we want to match you to your real team!)",
    "Name",
    "name",
]

DEMOGRAPHIC_COLUMNS = {
    "region": "Which part of the United States do you currently live in?",
    "sex_assigned_at_birth": "What is the sex that you were assigned at birth?",
    "age": "How old are you?",
    "race_or_origin": "What is your race or origin?",
    "relationship_status": "Which of these best describes you?",
    "religion": "What is your present religion, if any?",
    "religious_attendance": "Aside from weddings and funerals, how often do you attend religious services?",
    "political_views": "In general, would you describe your political views as:",
}

BACKGROUND_COLUMNS = {
    "primary_role": "Which best describes your primary role?",
    "occupation": "What is your occupation?",
    "duties": "In this occupation, what kind of work do you do and what are the most important activities or duties?",
    "education": "What is the highest level of schooling or degree that you have completed?",
    "background": "In a few sentences, describe your educational and occupational background. Do you have a disciplinary affiliation or approach (e.g., economics, computer science)?",
    "motivation": "In a few sentences, what draws you to the question of how AI is reshaping jobs and the economy?",
    "skills": "How would you describe the skill set that you contribute to the team?",
    "data_sources": "How would you describe the data sources you are most excited to work with?",
    "values": "What do you value the most in your life?",
    "hopes": "Imagine yourself a few years from now. Maybe you want your life to be the same in some ways as it is now. Maybe you want it to be different in some ways. What do you hope for?",
    "uploaded_documents": "Please upload any additional unstructed information that you think describes you and your interests/expertise (e.g., like a resume or CV).",
}

BUILDATHON_DEFAULT_TASK_PROMPT = (
    "You are participating in the Gates/Wharton Build-a-thon on AI, jobs, labor, "
    "organizations, and the economy. As a team, deliberate openly and decide what "
    "project, product, research prototype, or tool you would build today.\n\n"
    "Use only this shared hackpad project-idea menu as context. You may combine, "
    "refine, or reject ideas, but do not introduce unrelated themes:\n"
    "- Visualize the state of an applied economics research project: what is done, "
    "what is missing, and what should happen next.\n"
    "- Translate skills, jobs, and experience across taxonomies to reduce reliance "
    "on proprietary black-box mappings.\n"
    "- Connect higher education to graduates' labor-market outcomes using syllabi, "
    "course content, and career/profile data.\n"
    "- Train or release open LLMs or datasets based on job postings data for "
    "research use.\n"
    "- Build an open occupation-level or task-level demand-elasticity dataset/model "
    "linking cost impacts and labor demand.\n"
    "- Simulate divide-and-conquer work: agents with different characteristics "
    "complete realistic task steps, then evaluate whether isolated work recombines "
    "into a good final output.\n"
    "- Build a harness for sampling realistic tasks, deploying agents, and observing "
    "how they subdivide, allocate, and coordinate work.\n"
    "- Build a front-end conversation analysis tool for human-AI conversations, "
    "prompt coaching, communication-pattern detection, or concerning-pattern flags.\n"
    "- Build a prompt annotation tool that suggests how to make a user's prompt "
    "sharper.\n"
    "- Build a reverse task generator: start from skills or characteristics and "
    "generate relevant tasks or jobs.\n"
    "- Model supply-side skill-complement dynamics for healthy jobs, organizations, "
    "or economies.\n"
    "- Build and backtest a multi-factor labor displacement model using demand "
    "elasticity, adaptability, firm adoption, exposure, and recent labor data.\n"
    "- Build a New Work Tracker that detects novel task language emerging in job "
    "postings and tracks first appearances by occupation.\n"
    "- Build a job-description rewriter and bias measurement tool that emphasizes "
    "skills and experience over degrees, explains each change, and compares old/new "
    "ads using synthetic resume pools and pass-rate shifts.\n"
    "- Model augmentation-versus-automation scenarios in archetypal firms or "
    "industries with different roles, org structures, and AI adoption maturity.\n"
    "- Estimate the cost of performing the same task bundle with human, AI/agentic, "
    "or hybrid systems under different compute and labor prices.\n"
    "- Improve, automate, or add to jobsdata.ai.\n"
    "- Study people's anxieties and perceptions about AI's impact on their own "
    "careers, and how those beliefs shape career-transition behavior.\n\n"
    "Do not assume any known real team project or observed build-a-thon outcome. "
    "The goal is to predict which project this team would naturally choose from "
    "this idea space, and how they would shape it, using the survey-derived "
    "evidence about its members."
)


@dataclass(frozen=True)
class DocumentText:
    source: str
    content: str


def load_survey_csv(path: str | Path) -> list[dict[str, str]]:
    csv_path = Path(path)
    with csv_path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        return [
            _clean_row(row)
            for row in reader
            if any(_clean_value(value) for value in row.values())
        ]


def agents_from_survey_rows(
    rows: Iterable[dict[str, Any]],
    *,
    documents_by_person: dict[str, list[DocumentText]] | None = None,
) -> list[AgentPersona]:
    documents_by_person = documents_by_person or {}
    agents: list[AgentPersona] = []
    seen_ids: set[str] = set()
    for index, raw_row in enumerate(rows, start=1):
        row = _clean_row(raw_row)
        name = _first_value(row, NAME_COLUMNS) or f"Participant {index}"
        agent_id = _unique_id(_slugify(name) or f"participant-{index}", seen_ids)
        documents = documents_by_person.get(agent_id, [])
        agents.append(_agent_from_row(row=row, name=name, agent_id=agent_id, documents=documents))
    return agents


def build_survey_experiment_bundle(
    agents: list[AgentPersona],
    *,
    name: str = "survey-team",
    task_prompt: str = BUILDATHON_DEFAULT_TASK_PROMPT,
    description: str = "Survey-derived open-ended SimulaCrew team.",
) -> dict[str, Any]:
    """Build a four-file experiment bundle from survey-derived agents.

    Returns a dict with keys 'experiment', 'task', 'process', 'agents'. The
    'experiment' value references the others by relative filename, suitable for
    writing out as experiment.yaml + task.json + process.json + agents.json.
    """
    task = {
        "title": "Build-a-thon Project Simulation",
        "prompt": task_prompt,
        "target_user": "the simulation runner",
        "success_criteria": (
            "predict the concrete build-a-thon project this team would likely choose "
            "after open-ended deliberation, weighting the members' personal and "
            "professional backgrounds more heavily than generic task appeal"
        ),
        "output_format": {
            "type": "markdown",
            "description": (
                "Final synthesis must describe the team's predicted build-a-thon project. "
                "Include the main artifact, rationale, rejected alternatives, unresolved "
                "dissent, risks, immediate next steps, and why this specific team's "
                "backgrounds, skills, interests, and values make the project plausible."
            ),
            "required_sections": [
                "artifact",
                "rationale",
                "risks",
                "next steps",
            ],
        },
        "variables": {
            "time_budget": "20 minutes",
            "discussion_time_limit_minutes": 20,
            "decision_pressure": (
                "The team is trying to converge on a project they could build today, "
                "but personal fit should matter more than generic hackathon strategy."
            ),
            "shared_goal": (
                "Converge on one concrete build-a-thon project that this exact team "
                "would plausibly choose from its members' backgrounds."
            ),
            "default_goal_alignment_start": 0.42,
            "goal_alignment_threshold": 0.86,
            "goal_alignment_min_turns": 8,
            "goal_alignment_evaluator": "deterministic",
            "goal_alignment_step_self": 0.055,
            "goal_alignment_step_listener": 0.025,
            "goal_alignment_interrupt_factor": 0.65,
            "private_thinking_workers": min(max(len(agents), 1), 4),
        },
    }
    process = {
        "shared_instructions": (
            "You are part of a realistic working group. Your persona comes from "
            "survey-derived evidence, including demographic context, professional "
            "background, stated interests, collaboration preferences, and optional "
            "document evidence. Use those details as the primary decision lens; the "
            "task prompt is the shared arena, not the agent's whole identity. When "
            "generic task strategy conflicts with personal background, let the "
            "survey-derived background dominate. Do not make unsupported stereotyped "
            "inferences. Speak naturally and keep turns concise."
        ),
        "character_prompt_template": (
            "survey-derived character prompt for {agent_name}:\n\n"
            "Base behavior:\n{agent_base_prompt}\n\n"
            "Backstory and demographic/professional context:\n{agent_backstory}\n\n"
            "Past experience / document evidence:\n{agent_history}\n\n"
            "Skills:\n{agent_skills}\n\n"
            "Interests:\n{agent_interests}\n\n"
            "Knowledge they can draw from:\n{agent_knowledge}\n\n"
            "Speaking style:\n{agent_speaking_style}\n\n"
            "Survey questionnaire responses:\n{agent_personality}\n\n"
            "Goals:\n{agent_goals}\n\n"
            "Constraints:\n{agent_constraints}\n\n"
            "Use the survey and document evidence to decide what this person notices, "
            "where they are credible, what they may push for, when they speak, and how "
            "they collaborate. Weight this person's background, values, skills, and "
            "stated interests more heavily than the task description itself. Do not "
            "recite the survey mechanically."
        ),
        "interaction_rules": [
            "Start with independent reasoning before seeing peer arguments.",
            "During group chat, make one move per turn: ask, answer, challenge, clarify, concede, or propose.",
            "Draw naturally on professional background, skills, interests, values, and document evidence.",
            "Prefer ideas that feel personally plausible for these exact members over ideas that merely fit the topic.",
            "Avoid false consensus. If disagreement remains, preserve it.",
        ],
        "interruption_rules": [
            "Interrupt when a hidden assumption would change the decision.",
            "Interrupt when the group is converging too early.",
            "Interrupt when a proposal is vague enough that it cannot be tested.",
            "Do not interrupt merely to restate your preference.",
        ],
        "rounds": _default_rounds(),
    }
    agents_payload = {"agents": [_agent_to_config(agent) for agent in agents]}
    experiment = {
        "name": name,
        "description": description,
        "metadata": {
            "source": "survey-ingestion",
            "agent_count": len(agents),
        },
        "task": "task.json",
        "process": "process.json",
        "agents": "agents.json",
    }
    return {
        "experiment": experiment,
        "task": task,
        "process": process,
        "agents": agents_payload,
    }


def _agent_from_row(
    *,
    row: dict[str, str],
    name: str,
    agent_id: str,
    documents: list[DocumentText],
) -> AgentPersona:
    demographics = _field_values(row, DEMOGRAPHIC_COLUMNS)
    background = _field_values(row, BACKGROUND_COLUMNS)
    questionnaire = _questionnaire_from_row(row)
    skills = _split_phrases(background.get("skills", ""))
    interests = _compact_list([background.get("data_sources", ""), background.get("motivation", "")])
    knowledge = _compact_list(
        [
            background.get("occupation", ""),
            background.get("primary_role", ""),
            background.get("background", ""),
            _document_summary(documents),
        ]
    )
    history = _compact_list(
        [
            *[
                f"{document.source}: {_trim_words(document.content, 50)}"
                for document in documents
                if document.content.strip()
            ],
            background.get("duties", ""),
            background.get("education", ""),
        ]
    )
    base_prompt = _base_prompt(name=name, background=background, demographics=demographics)
    backstory = _backstory(background=background, demographics=demographics)
    return AgentPersona(
        id=agent_id,
        name=name,
        base_prompt=base_prompt,
        personality=questionnaire,
        backstory=backstory,
        speaking_style="",
        knowledge=knowledge,
        skills=skills,
        interests=interests,
        history=history,
        goals=_compact_list(
            [
                background.get("motivation", ""),
                f"Values: {background['values']}" if background.get("values") else "",
                f"Hopes: {background['hopes']}" if background.get("hopes") else "",
            ]
        ),
        constraints=[],
    )


def _clean_row(row: dict[str, Any]) -> dict[str, str]:
    cleaned: dict[str, str] = {}
    for key, value in row.items():
        if key is None:
            continue
        cleaned[str(key).strip()] = _clean_value(value)
    return cleaned


def _clean_value(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _first_value(row: dict[str, str], columns: list[str]) -> str:
    for column in columns:
        value = row.get(column, "").strip()
        if value:
            return value
    return ""


def _field_values(row: dict[str, str], columns: dict[str, str]) -> dict[str, str]:
    return {
        label: row[column].strip()
        for label, column in columns.items()
        if row.get(column, "").strip()
    }


def _questionnaire_from_row(row: dict[str, str]) -> dict[str, str]:
    non_question_columns = {
        "Timestamp",
        *NAME_COLUMNS,
        *DEMOGRAPHIC_COLUMNS.values(),
        *BACKGROUND_COLUMNS.values(),
    }
    return {
        column: value
        for column, value in row.items()
        if column not in non_question_columns and value.strip()
    }


def _base_prompt(
    *,
    name: str,
    background: dict[str, str],
    demographics: dict[str, str],
) -> str:
    role = background.get("occupation") or background.get("primary_role") or "participant"
    context = _sentence_join(
        [
            f"{name} is represented as a survey-derived participant",
            f"their role is {role}",
            f"their background includes {background.get('background')}" if background.get("background") else "",
            f"demographic context includes {_format_mapping_inline(demographics)}" if demographics else "",
        ]
    )
    return (
        f"{context}. Deliberate from this person's stated background, skills, "
        "interests, values, and collaboration preferences. Avoid unsupported stereotypes."
    )


def _backstory(*, background: dict[str, str], demographics: dict[str, str]) -> str:
    parts = [
        f"Demographics: {_format_mapping_inline(demographics)}" if demographics else "",
        f"Role: {background.get('primary_role', '')}" if background.get("primary_role") else "",
        f"Occupation: {background.get('occupation', '')}" if background.get("occupation") else "",
        f"Duties: {background.get('duties', '')}" if background.get("duties") else "",
        f"Education/background: {background.get('background', '')}" if background.get("background") else "",
    ]
    return "\n".join(part for part in parts if part)


def _document_summary(documents: list[DocumentText]) -> str:
    snippets = [
        _trim_words(document.content, 35)
        for document in documents
        if document.content.strip()
    ]
    return " ".join(snippets)


def _split_phrases(value: str) -> list[str]:
    parts = re.split(r";|,|\n", value)
    return _compact_list(parts)


def _compact_list(items: Iterable[str]) -> list[str]:
    return [item.strip() for item in items if item and item.strip()]


def _sentence_join(parts: Iterable[str]) -> str:
    return "; ".join(part.strip().rstrip(".") for part in parts if part and part.strip())


def _format_mapping_inline(mapping: dict[str, str]) -> str:
    return "; ".join(f"{key}={value}" for key, value in mapping.items())


def _trim_words(text: str, limit: int) -> str:
    words = text.split()
    if len(words) <= limit:
        return " ".join(words)
    return " ".join(words[:limit]).rstrip() + "..."


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower()).strip("-")
    return slug or "participant"


def _unique_id(base: str, seen: set[str]) -> str:
    if base not in seen:
        seen.add(base)
        return base
    index = 2
    while f"{base}-{index}" in seen:
        index += 1
    unique = f"{base}-{index}"
    seen.add(unique)
    return unique


def _agent_to_config(agent: AgentPersona) -> dict[str, Any]:
    return {
        "id": agent.id,
        "name": agent.name,
        "base_prompt": agent.base_prompt,
        "personality": agent.personality,
        "backstory": agent.backstory,
        "speaking_style": agent.speaking_style,
        "knowledge": agent.knowledge,
        "skills": agent.skills,
        "interests": agent.interests,
        "history": agent.history,
        "goals": agent.goals,
        "constraints": agent.constraints,
    }


def _default_rounds() -> list[dict[str, Any]]:
    return [
        {
            "id": "private_positions",
            "title": "Independent Positions",
            "mode": "private",
            "transcript_visibility": "hidden",
            "prompt": (
                "Topic: {topic_prompt}\n\nTarget user: {target_user}\n"
                "Success criteria: {success_criteria}\nTime budget: {time_budget}\n\n"
                "As {agent_name}, quietly read the task and give your initial take as "
                "two or three plain spoken sentences. Treat your survey-derived "
                "background, skills, values, interests, and document evidence as the "
                "primary source of what you notice and want. The task is only the "
                "shared prompt. No headings or bullets."
            ),
        },
        {
            "id": "group_chat",
            "title": "Group Chat",
            "mode": "discussion",
            "transcript_visibility": "named",
            "max_turns": 24,
            "turn_strategy": "round_robin",
            "prompt": (
                "You are {current_speaker}. Turn {turn_number}; {turns_remaining} "
                "turn slots remain. Discussion hard limit: {discussion_time_limit}.\n\n"
                "Topic: {topic_prompt}\nShared objective: {alignment_goal}\n"
                "Turn type for your behavior only: {event_type}\n"
                "Private interruption tendency if relevant: {interruption_score}/10\n"
                "Private reason if relevant: {interruption_rationale}\n"
                "Private notes from times you stayed quiet:\n{private_memory}\n\n"
                "Transcript:\n{transcript}\n\n"
                "Talk like a normal person in a small group trying to produce a useful "
                "deliverable. Before choosing what to say, anchor in your personal "
                "survey evidence: your background, skills, interests, values, and "
                "work history should shape the proposal more than generic task fit. "
                "Reply to the last useful point, ask a question, push back, concede, "
                "or propose a merge. Do not write the words interrupt, score, tendency, "
                "rationale, alignment, or threshold. One or two plain sentences."
            ),
        },
        {
            "id": "final_synthesis",
            "title": "Final Synthesis",
            "mode": "synthesis",
            "transcript_visibility": "named",
            "speaker_id": "recorder",
            "speaker_name": "Recorder",
            "speaker_prompt": (
                "You are a neutral recorder. Do not invent consensus. Preserve dissent "
                "and identify the best next step."
            ),
            "prompt": (
                "Topic: {topic_prompt}\nShared objective: {alignment_goal}\n"
                "Private notes that informed but did not appear in chat:\n{private_memory}\n\n"
                "Full transcript:\n{transcript}\n\n{output_contract}\n\n"
                "Write the final artifact. Explain why this particular team, given "
                "its members' backgrounds, skills, interests, values, and stated "
                "motivations, would choose this project. Do not include the raw transcript."
            ),
        },
    ]
