from __future__ import annotations

import io
import json
import os
import sys
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from graph_agent.agent.provider import LLMSettings, OpenAICompatibleClient, ProviderError


def response(content="hello", usage=True):
    data = {"choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": content}}]}
    if usage:
        data["usage"] = {"prompt_tokens": 12, "completion_tokens": 3}
    return data


class ProviderTests(unittest.TestCase):
    def settings(self, **kwargs):
        return LLMSettings("https://api.deepseek.com/v1/", "secret-key", **kwargs)

    def test_deepseek_request_and_real_usage(self):
        captured = []
        def transport(request, timeout):
            captured.append((request, timeout))
            return response()
        client = OpenAICompatibleClient(self.settings(), transport=transport)
        result = client.complete([{"role": "user", "content": "hi"}], tools=[{"type": "function"}])
        request, timeout = captured[0]
        self.assertEqual(request.full_url, "https://api.deepseek.com/v1/chat/completions")
        self.assertEqual(timeout, 60)
        self.assertEqual(request.get_header("Authorization"), "Bearer secret-key")
        payload = json.loads(request.data)
        self.assertEqual(payload["tool_choice"], "auto")
        self.assertEqual(payload["thinking"], {"type": "disabled"})
        self.assertEqual((result.input_tokens, result.output_tokens, result.usage_reported), (12, 3, True))
        self.assertNotIn("secret-key", repr(self.settings()))

    def test_generic_api_receives_no_deepseek_extension_or_tools_when_disabled(self):
        def transport(request, timeout):
            payload = json.loads(request.data)
            self.assertNotIn("tools", payload)
            self.assertNotIn("thinking", payload)
            return response()
        OpenAICompatibleClient(LLMSettings("https://example.org/v1", "key"), transport=transport).complete([])

    def test_retry_rate_limit_then_success(self):
        calls, waits = [], []
        def transport(request, timeout):
            calls.append(1)
            if len(calls) == 1:
                raise urllib.error.HTTPError(request.full_url, 429, "secret-key", {}, io.BytesIO(b"secret-key"))
            return response()
        result = OpenAICompatibleClient(self.settings(), transport=transport, sleep=waits.append).complete([])
        self.assertEqual(len(calls), 2)
        self.assertEqual(waits, [1])
        self.assertEqual(result.message["content"], "hello")

    def test_auth_error_is_not_retried_and_does_not_expose_response(self):
        calls = []
        def transport(request, timeout):
            calls.append(1)
            raise urllib.error.HTTPError(request.full_url, 401, "secret-key", {}, io.BytesIO(b"secret-key"))
        with self.assertRaises(ProviderError) as context:
            OpenAICompatibleClient(self.settings(), transport=transport).complete([])
        self.assertEqual(len(calls), 1)
        self.assertNotIn("secret-key", str(context.exception))

    def test_network_retries_are_bounded(self):
        calls = []
        def transport(request, timeout):
            calls.append(1)
            raise TimeoutError("secret-key")
        with self.assertRaises(ProviderError) as context:
            OpenAICompatibleClient(self.settings(), transport=transport, sleep=lambda _: None).complete([])
        self.assertEqual(len(calls), 3)
        self.assertNotIn("secret-key", str(context.exception))

    def test_missing_usage_is_marked_unknown(self):
        result = OpenAICompatibleClient(self.settings(), transport=lambda *_: response(usage=False)).complete([])
        self.assertFalse(result.usage_reported)
        self.assertEqual(result.input_tokens, 0)

    def test_invalid_and_truncated_completions_fail(self):
        truncated = response()
        truncated["choices"][0]["finish_reason"] = "length"
        for bad in ({}, [], {"choices": []}, {"choices": [None]}, {"choices": [{"message": []}]}, truncated):
            with self.subTest(bad=bad), self.assertRaises(ProviderError):
                OpenAICompatibleClient(self.settings(), transport=lambda *_, d=bad: d).complete([])

    def test_env_precedence_quotes_and_missing_key(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / ".env"
            path.write_text('LLM_API_KEY="local-key"\nLLM_MODEL=local-model\n', encoding="utf-8")
            with patch.dict(os.environ, {"LLM_MODEL": "process-model"}, clear=True):
                settings = LLMSettings.from_env(path)
                self.assertEqual(settings.model, "process-model")
                self.assertEqual(settings.api_key, "local-key")
            with patch.dict(os.environ, {}, clear=True), self.assertRaises(ValueError):
                LLMSettings.from_env(Path(directory) / "missing")

    def test_invalid_endpoints_and_timeouts_fail_locally(self):
        for base in ("file:///tmp/key", "https://user:key@example.org", "https://example.org?key=secret"):
            with self.subTest(base=base), self.assertRaises(ValueError):
                LLMSettings(base, "key")
        for timeout in (0, -1, float("nan"), float("inf")):
            with self.subTest(timeout=timeout), self.assertRaises(ValueError):
                self.settings(timeout_seconds=timeout)


if __name__ == "__main__":
    unittest.main()
