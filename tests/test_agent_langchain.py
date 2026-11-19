"""Exercise real LangChain conversions/HTTP requests with a mocked API transport."""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
try:
    import httpx
    from langchain_openai import ChatOpenAI
    HAS_LANGCHAIN = True
except ImportError:
    HAS_LANGCHAIN = False

from graph_agent.agent.demo import DEMO_CATALOG, DEMO_QUESTION, DEMO_TRIPLES, answer_message, query_message
from graph_agent.agent.provider import LangChainClient, LLMSettings, ProviderError
from graph_agent.agent.runner import AgentRunner
from graph_agent.retrieval.graph import GraphIndex


@unittest.skipUnless(HAS_LANGCHAIN, "Run in the Anaconda langchain environment")
class LangChainIntegrationTests(unittest.TestCase):
    def client(self, messages, *, status=200):
        requests = []
        queue = list(messages)
        def handler(request):
            requests.append(json.loads(request.content))
            if status != 200:
                return httpx.Response(status, json={"error": {"message": "secret-key"}})
            return httpx.Response(200, json={
                "id": "mock-completion", "object": "chat.completion", "created": 1,
                "model": "deepseek-flash", "choices": [{"index": 0, "finish_reason": "stop",
                    "message": queue.pop(0)}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
            })
        http_client = httpx.Client(transport=httpx.MockTransport(handler))
        self.addCleanup(http_client.close)
        # Run the production constructor with only its HTTP transport replaced.
        with patch("langchain_openai.ChatOpenAI", side_effect=lambda **kwargs: ChatOpenAI(http_client=http_client, **kwargs)):
            client = LangChainClient(LLMSettings("https://api.deepseek.com", "secret-key", max_retries=0))
        return client, requests

    def test_real_langchain_tool_loop_and_usage_with_mocked_HTTP(self):
        client, requests = self.client([
            query_message(), {"role": "assistant", "content": "Ready"},
            answer_message(ids=("city:sample",), text="Sample City", evidence=(DEMO_TRIPLES[0], DEMO_TRIPLES[2]))])
        runner = AgentRunner(client, retriever=GraphIndex(DEMO_TRIPLES), catalog=DEMO_CATALOG)
        prediction = runner.run(DEMO_QUESTION, "text")
        self.assertEqual(prediction.answer_ids, ("city:sample",))
        self.assertEqual((prediction.input_tokens, prediction.output_tokens), (30, 6))
        self.assertEqual(requests[0]["tool_choice"], "auto")
        self.assertEqual(requests[0]["thinking"], {"type": "disabled"})
        self.assertEqual(requests[0]["max_tokens"], 1024)
        self.assertNotIn("max_completion_tokens", requests[0])
        self.assertEqual(requests[1]["messages"][-1]["role"], "tool")
        self.assertNotIn("tools", requests[-1])
        self.assertNotIn("secret-key", json.dumps(runner.last_trace))
        self.assertEqual(runner.last_trace["provider"]["backend"], "LangChainClient")
        self.assertEqual(runner.last_trace["events"][-1]["served_model"], "deepseek-flash")

    def test_real_langchain_skip_and_pure_llm_do_not_query(self):
        for mode, messages in (("llm", [answer_message(text="Unknown")]),
                               ("text", [{"role": "assistant", "content": "Skip"}, answer_message(text="Unknown")])):
            with self.subTest(mode=mode):
                client, requests = self.client(messages)
                runner = AgentRunner(client, retriever=GraphIndex(DEMO_TRIPLES), catalog=DEMO_CATALOG)
                prediction = runner.run(DEMO_QUESTION, mode)
                self.assertEqual(prediction.tool_calls, 0)
                if mode == "llm":
                    prompt = json.loads(requests[0]["messages"][1]["content"])
                    self.assertEqual(prompt, {"question": DEMO_QUESTION.text})

    def test_sdk_HTTP_errors_are_sanitized(self):
        client, requests = self.client([], status=401)
        with self.assertRaises(ProviderError) as context:
            client.complete([{"role": "user", "content": "Hi"}])
        self.assertIn("401", str(context.exception))
        self.assertNotIn("secret-key", str(context.exception))
        self.assertEqual(len(requests), 1)

    def test_framework_invalid_tool_arguments_are_rejected(self):
        message = query_message()
        message["tool_calls"][0]["function"]["arguments"] = "not JSON"
        client, _ = self.client([message])
        with self.assertRaises(ProviderError):
            client.complete([{"role": "user", "content": "Hi"}])

    def test_default_constructor_targets_chat_completions_and_disables_thinking(self):
        client = LangChainClient(LLMSettings("https://api.deepseek.com", "secret-key"))
        self.assertEqual(client._model.extra_body, {"thinking": {"type": "disabled"}, "max_tokens": 1024})
        self.assertFalse(client._model.use_responses_api)
        self.assertEqual(client._model.model_name, "deepseek-flash")


if __name__ == "__main__":
    unittest.main()
