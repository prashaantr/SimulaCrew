from __future__ import annotations

import asyncio
import os
from typing import Any, Protocol


class ModelClient(Protocol):
    provider: str

    def complete(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        model: str,
        temperature: float = 0.2,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Return one model completion."""


class DryRunClient:
    provider = "dry-run"

    def complete(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        model: str,
        temperature: float = 0.2,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        metadata = metadata or {}
        agent_name = metadata.get("agent_name", "Recorder")
        event_type = metadata.get("event_type", "statement")
        turn_number = metadata.get("turn_number", "?")
        system_preview = " ".join(system_prompt.split())[:220]
        prompt_preview = " ".join(user_prompt.split())[:320]
        return (
            f"Dry-run {event_type} from {agent_name} on turn {turn_number}.\n\n"
            f"System prompt focus: {system_preview}\n\n"
            f"User prompt focus: {prompt_preview}"
        )


class OpenAIClient:
    provider = "openai"

    def __init__(self, api_key: str | None = None) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "OpenAI support requires: pip install -e '.[openai]'"
            ) from exc

        resolved_api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not resolved_api_key:
            raise RuntimeError("OPENAI_API_KEY is required for --provider openai.")
        self._client = OpenAI(api_key=resolved_api_key)

    def complete(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        model: str,
        temperature: float = 0.2,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        response = self._client.chat.completions.create(
            model=model,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        content = response.choices[0].message.content
        if not content:
            raise RuntimeError("OpenAI returned an empty response.")
        return content


def create_client(provider: str) -> ModelClient:
    if provider == "dry-run":
        return DryRunClient()
    if provider == "openai":
        return OpenAIClient()
    if provider == "claude":
        return ClaudeSDKClient()
    raise ValueError(f"Unsupported provider: {provider}")


class ClaudeSDKClient:
    provider = "claude"

    def __init__(self) -> None:
        if not os.getenv("ANTHROPIC_API_KEY"):
            raise RuntimeError("ANTHROPIC_API_KEY is required for --provider claude.")
        try:
            from claude_agent_sdk import ClaudeAgentOptions, query
        except ImportError as exc:
            raise RuntimeError(
                "Claude support requires: pip install -e '.[claude]'"
            ) from exc
        self._options_cls = ClaudeAgentOptions
        self._query = query

    def complete(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        model: str,
        temperature: float = 0.2,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        return asyncio.run(
            self._complete_async(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                model=model,
            )
        )

    async def _complete_async(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        model: str,
    ) -> str:
        options = self._options_cls(
            allowed_tools=[],
            permission_mode="default",
            system_prompt=system_prompt,
            model=model,
            max_turns=1,
        )
        result_text = ""
        async for message in self._query(prompt=user_prompt, options=options):
            if hasattr(message, "result") and message.result:
                result_text = str(message.result)
            elif hasattr(message, "content"):
                chunks: list[str] = []
                for block in message.content:
                    text = getattr(block, "text", None)
                    if text:
                        chunks.append(str(text))
                if chunks:
                    result_text = "\n".join(chunks)

        if not result_text.strip():
            raise RuntimeError("Claude SDK returned an empty response.")
        return result_text.strip()
