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

LIKERT_COLUMNS = {
    "openness": [
        "I enjoy being unique and different from others in many ways.",
        "I generally solve problems creatively",
        "I have an active imagination.",
        "I would prefer complex to simple problems.",
        "I really enjoy a task that involves coming up with new solutions to problems",
    ],
    "conscientiousness": [
        "For me, it is very important to carry out the obligations placed on me",
        "I do a thorough job.",
        "-I tend to be lazy.",
    ],
    "extraversion": [
        "I am outgoing, sociable.",
        "-I am reserved.",
    ],
    "agreeableness": [
        "The well-being of my coworkers is important to me.",
        "I feel good when I cooperate with others.",
        "I am generally trusting.",
        "-I tend to find fault with others.",
    ],
    "assertiveness": [
        "I often do “my own thing.”",
        "I prefer to work without instructions from others",
        "When faced with a difficult person problem, it is better to decide what to do yourself rather than follow the advice of others.",
    ],
    "skepticism": [
        "I tend to find fault with others.",
        "It's enough for me that something gets the job done; I don't care how or why it works.",
    ],
    "neuroticism": [
        "I get nervous easily.",
        "-I am relaxed, handle stress well.",
    ],
    "patience": [
        "I will stay in a group if they need me, even when I'm not happy with the group.",
        "-If the group is slowing me down, it is better to leave it and work alone.",
    ],
}

LIKERT_VALUES = {
    "strongly disagree": 1.0,
    "disagree": 3.0,
    "neutral": 5.0,
    "neither agree nor disagree": 5.0,
    "agree": 7.0,
    "strongly agree": 9.0,
}


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
    task_prompt: str = "Deliberate as a team and produce the best final artifact for the task.",
    description: str = "Survey-derived open-ended SimulaCrew team.",
) -> dict[str, Any]:
    """Build a four-file experiment bundle from survey-derived agents.

    Returns a dict with keys 'experiment', 'task', 'process', 'agents'. The
    'experiment' value references the others by relative filename, suitable for
    writing out as experiment.yaml + task.json + process.json + agents.json.
    """
    task = {
        "title": "Survey-Derived Team Simulation",
        "prompt": task_prompt,
        "target_user": "the simulation runner",
        "success_criteria": "produce a concrete final artifact that reflects the team's likely deliberation",
        "output_format": {
            "type": "markdown",
            "description": (
                "Final synthesis must describe the team's predicted final deliverable. "
                "Include the main artifact, rationale, rejected alternatives, unresolved "
                "dissent, risks, and immediate next steps."
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
            "decision_pressure": "The team is trying to converge on a useful output today.",
            "shared_goal": "Converge on one concrete final deliverable for the task.",
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
            "document evidence. Use those details as grounded context; do not make "
            "unsupported stereotyped inferences. Speak naturally and keep turns concise."
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
            "Personality questionnaire fields:\n{agent_personality}\n\n"
            "Goals:\n{agent_goals}\n\n"
            "Constraints:\n{agent_constraints}\n\n"
            "Use the survey and document evidence to decide what this person notices, "
            "where they are credible, what they may push for, when they speak, and how "
            "they collaborate. Do not recite the survey mechanically."
        ),
        "interaction_rules": [
            "Start with independent reasoning before seeing peer arguments.",
            "During group chat, make one move per turn: ask, answer, challenge, clarify, concede, or propose.",
            "Draw naturally on professional background, skills, interests, values, and document evidence.",
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
    personality = _personality_from_row(row)
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
        personality=personality,
        backstory=backstory,
        speaking_style=_speaking_style(personality),
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
        constraints=_constraints(personality),
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


def _personality_from_row(row: dict[str, str]) -> dict[str, float]:
    personality: dict[str, float] = {}
    for trait, columns in LIKERT_COLUMNS.items():
        values: list[float] = []
        for column in columns:
            reverse = column.startswith("-")
            actual_column = column[1:] if reverse else column
            score = _likert_score(row.get(actual_column, ""))
            if score is None:
                continue
            values.append(10.0 - score if reverse else score)
        if values:
            personality[trait] = round(sum(values) / len(values), 2)
    personality.setdefault("openness", 5.0)
    personality.setdefault("conscientiousness", 5.0)
    personality.setdefault("extraversion", 5.0)
    personality.setdefault("agreeableness", 5.0)
    personality.setdefault("assertiveness", 5.0)
    personality.setdefault("skepticism", 5.0)
    personality.setdefault("neuroticism", 5.0)
    personality.setdefault("urgency", 5.0)
    personality.setdefault("patience", 5.0)
    personality["interruptiveness"] = round(
        (personality["assertiveness"] + personality["skepticism"] + (10.0 - personality["patience"])) / 3,
        2,
    )
    personality["disagreement_sensitivity"] = round(
        (personality["skepticism"] + (10.0 - personality["agreeableness"]) + personality["conscientiousness"]) / 3,
        2,
    )
    personality["goal_alignment_start"] = round(
        min(0.7, max(0.25, (personality["agreeableness"] + personality["conscientiousness"]) / 25)),
        2,
    )
    return personality


def _likert_score(value: str) -> float | None:
    normalized = value.strip().lower()
    if not normalized:
        return None
    if normalized in LIKERT_VALUES:
        return LIKERT_VALUES[normalized]
    try:
        number = float(normalized)
    except ValueError:
        return None
    return max(1.0, min(9.0, number))


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


def _speaking_style(personality: dict[str, float]) -> str:
    fragments: list[str] = []
    fragments.append("more direct" if personality["assertiveness"] >= 6.5 else "more tentative")
    fragments.append("sociable" if personality["extraversion"] >= 6.5 else "reserved")
    fragments.append("detail-oriented" if personality["conscientiousness"] >= 6.5 else "flexible")
    fragments.append("collaborative" if personality["agreeableness"] >= 6.5 else "willing to challenge")
    return ", ".join(fragments)


def _constraints(personality: dict[str, float]) -> list[str]:
    constraints: list[str] = []
    if personality["assertiveness"] >= 7:
        constraints.append("may push strongly for their own framing")
    if personality["extraversion"] <= 4:
        constraints.append("may wait to speak unless the point feels important")
    if personality["skepticism"] >= 7:
        constraints.append("may focus on failure modes and assumptions")
    if personality["agreeableness"] >= 7:
        constraints.append("may smooth over disagreement unless prompted to preserve dissent")
    return constraints or ["respond from the survey evidence without overclaiming"]


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
                "two or three plain spoken sentences. Draw on your survey-derived "
                "background and document evidence. No headings or bullets."
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
                "Your private current alignment with the objective: {goal_alignment}\n"
                "Private group alignment snapshot: {goal_alignment_summary}\n"
                "Turn type for your behavior only: {event_type}\n"
                "Private interruption tendency if relevant: {interruption_score}/10\n"
                "Private reason if relevant: {interruption_rationale}\n"
                "Private notes from times you stayed quiet:\n{private_memory}\n\n"
                "Transcript:\n{transcript}\n\n"
                "Talk like a normal person in a small group trying to produce a useful "
                "deliverable. Reply to the last useful point, ask a question, push back, "
                "concede, or propose a merge. Do not write the words interrupt, score, "
                "tendency, rationale, alignment, or threshold. One or two plain sentences."
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
                "Final alignment state: {goal_alignment_summary}\n"
                "Private notes that informed but did not appear in chat:\n{private_memory}\n\n"
                "Full transcript:\n{transcript}\n\n{output_contract}\n\n"
                "Write the final artifact. Do not include the raw transcript."
            ),
        },
    ]
