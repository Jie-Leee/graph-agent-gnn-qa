"""LangChain ChatOpenAI adapter plus a lightweight compatible HTTP client."""
from __future__ import annotations

import json
import math
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Protocol
from urllib.parse import urlsplit


class ProviderError(RuntimeError):
    """Safe to display: never contains response bodies or request credentials."""


@dataclass(frozen=True)
class Completion:
    message: dict
    input_tokens: int = 0
    output_tokens: int = 0
    usage_reported: bool = False
    served_model: str | None = None


class ChatProvider(Protocol):
    def complete(self, messages: list[dict], *, tools: list[dict] | None = None) -> Completion: ...


@dataclass(frozen=True)
class LLMSettings:
    api_base: str
    api_key: str = field(repr=False)
    model: str = "deepseek-flash"
    timeout_seconds: float = 60
    max_retries: int = 2
    max_output_tokens: int = 1024
    temperature: float = 0

    def __post_init__(self):
        parsed = urlsplit(self.api_base)
        if (parsed.scheme not in ("https", "http") or not parsed.hostname
                or parsed.username or parsed.password or parsed.query or parsed.fragment):
            raise ValueError("LLM_API_BASE must be an HTTP(S) base URL without credentials/query")
        if not self.api_key.strip() or not self.model.strip():
            raise ValueError("Set LLM_API_KEY and LLM_MODEL in your local .env")
        if not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0:
            raise ValueError("LLM_TIMEOUT_SECONDS must be positive and finite")
        if type(self.max_retries) is not int or not 0 <= self.max_retries <= 5:
            raise ValueError("max_retries must be between 0 and 5")
        if type(self.max_output_tokens) is not int or self.max_output_tokens < 1:
            raise ValueError("max_output_tokens must be positive")
        if not math.isfinite(self.temperature) or not 0 <= self.temperature <= 2:
            raise ValueError("temperature must be between 0 and 2")

    @classmethod
    def from_env(cls, env_file: Path | None = None) -> LLMSettings:
        # Only the explicitly selected local file is read. Process env wins.
        values = {}
        if env_file is not None and env_file.is_file():
            for number, line in enumerate(env_file.read_text(encoding="utf-8-sig").splitlines(), 1):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" not in line:
                    raise ValueError(f"Invalid .env assignment at line {number}")
                key, value = line.split("=", 1)
                value = value.strip()
                if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                    value = value[1:-1]
                values[key.strip()] = value
        values.update(os.environ)
        try:
            return cls(
                api_base=values.get("LLM_API_BASE") or "https://api.deepseek.com",
                api_key=values.get("LLM_API_KEY") or values.get("DEEPSEEK_API_KEY", ""),
                model=values.get("LLM_MODEL") or "deepseek-flash",
                timeout_seconds=float(values.get("LLM_TIMEOUT_SECONDS", "60")),
            )
        except (TypeError, ValueError):
            raise ValueError("Invalid LLM configuration: check local API base/key/model/timeout") from None


def _transport(request: urllib.request.Request, timeout: float) -> dict:
    # No redirects: an API key must not follow a redirect to a different host.
    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, headers, newurl):
            return None

    with urllib.request.build_opener(NoRedirect).open(request, timeout=timeout) as response:
        return json.load(response)


class OpenAICompatibleClient:
    def __init__(self, settings: LLMSettings, *, transport: Callable = _transport,
                 sleep: Callable = time.sleep):
        self.settings = settings
        self._transport = transport
        self._sleep = sleep

    def complete(self, messages: list[dict], *, tools: list[dict] | None = None) -> Completion:
        settings = self.settings
        payload = {"model": settings.model, "messages": messages, "stream": False,
                   "temperature": settings.temperature, "max_tokens": settings.max_output_tokens}
        if urlsplit(settings.api_base).hostname == "api.deepseek.com":
            # Keep routing/answers in non-thinking mode with a bounded output budget.
            payload["thinking"] = {"type": "disabled"}
        if tools:
            payload.update(tools=tools, tool_choice="auto")
        request = urllib.request.Request(
            settings.api_base.rstrip("/") + "/chat/completions",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Authorization": "Bearer " + settings.api_key, "Content-Type": "application/json"},
            method="POST",
        )
        for attempt in range(settings.max_retries + 1):
            try:
                data = self._transport(request, settings.timeout_seconds)
                return self._parse(data)
            except urllib.error.HTTPError as error:
                status = error.code
                error.close()
                retryable = status in (408, 429, 500, 502, 503, 504)
                if not retryable or attempt == settings.max_retries:
                    raise ProviderError(f"LLM HTTP {status}; check credentials, quota and API configuration") from None
            except (urllib.error.URLError, TimeoutError, OSError):
                if attempt == settings.max_retries:
                    raise ProviderError("LLM network request failed or timed out") from None
            except (ValueError, TypeError, KeyError, IndexError):
                raise ProviderError("LLM returned malformed JSON or an unsupported completion") from None
            self._sleep(min(2 ** attempt, 4))
        raise AssertionError("unreachable")

    @staticmethod
    def _parse(data: dict) -> Completion:
        if not isinstance(data, dict) or not isinstance(data.get("choices"), list):
            raise ValueError("Expected choices")
        choice = data["choices"][0]
        if not isinstance(choice, dict) or not isinstance(choice.get("message"), dict):
            raise ValueError("Expected message")
        if choice.get("finish_reason") in ("length", "content_filter"):
            raise ProviderError("LLM response was truncated or filtered; no prediction was produced")
        message = choice["message"]
        if message.get("role") != "assistant":
            raise ValueError("Expected assistant message")
        content, calls = message.get("content"), message.get("tool_calls")
        if content is not None and not isinstance(content, str):
            raise ValueError("Expected text content")
        if calls is not None and not isinstance(calls, list):
            raise ValueError("Expected tool_calls array")
        if not content and not calls:
            raise ValueError("Empty completion")
        clean = {"role": "assistant", "content": content}
        if calls:
            clean["tool_calls"] = calls
        usage = data.get("usage") or {}
        if not isinstance(usage, dict):
            raise ValueError("Invalid usage")
        reported = all(type(usage.get(k)) is int and usage[k] >= 0
                       for k in ("prompt_tokens", "completion_tokens"))
        return Completion(clean, usage["prompt_tokens"] if reported else 0,
                          usage["completion_tokens"] if reported else 0, reported,
                          data.get("model") if isinstance(data.get("model"), str) else None)


class LangChainClient:
    """Default backend for the project's Anaconda langchain environment."""
    def __init__(self, settings: LLMSettings, *, model=None):
        try:
            from langchain_core.messages import convert_to_messages, convert_to_openai_messages
            from langchain_openai import ChatOpenAI
        except ImportError:
            raise ProviderError("Use conda environment langchain; install requirements-agent.txt if needed") from None
        self.settings = settings
        self._to_messages = convert_to_messages
        self._to_openai = convert_to_openai_messages
        # ChatOpenAI rewrites its max_tokens field to max_completion_tokens.
        # Compatible providers such as DeepSeek expect max_tokens; extra_body
        # preserves that wire name when the OpenAI SDK merges the request.
        extra = {"max_tokens": settings.max_output_tokens}
        if urlsplit(settings.api_base).hostname == "api.deepseek.com":
            extra["thinking"] = {"type": "disabled"}
        self._model = model if model is not None else ChatOpenAI(
            model=settings.model, api_key=settings.api_key, base_url=settings.api_base.rstrip("/"),
            timeout=settings.timeout_seconds, max_retries=settings.max_retries,
            temperature=settings.temperature,
            extra_body=extra, use_responses_api=False,
        )

    def complete(self, messages: list[dict], *, tools: list[dict] | None = None) -> Completion:
        try:
            runnable = self._model.bind_tools(tools, tool_choice="auto") if tools else self._model
            response = runnable.invoke(self._to_messages(messages))
            if response.invalid_tool_calls:
                raise ProviderError("LLM returned invalid tool-call arguments")
            metadata = response.response_metadata
            usage = response.usage_metadata
            token_usage = ({"prompt_tokens": usage["input_tokens"],
                            "completion_tokens": usage["output_tokens"]} if usage
                           else metadata.get("token_usage"))
            return OpenAICompatibleClient._parse({"choices": [{
                "finish_reason": metadata.get("finish_reason"),
                "message": self._to_openai(response),
            }], "usage": token_usage, "model": metadata.get("model_name", metadata.get("model"))})
        except ProviderError:
            raise
        except Exception as error:
            status = getattr(error, "status_code", None)
            if type(status) is int:
                raise ProviderError(f"LLM HTTP {status}; check credentials, quota and API configuration") from None
            raise ProviderError("LangChain LLM request failed or returned an unsupported completion") from None
