from django.test import SimpleTestCase

from .context_manager import (
    _build_compaction_text,
    _build_history_text,
    _group_history_messages,
    _sliding_window_compress,
    consolidate_messages,
    estimate_messages_tokens,
    is_context_overflow_error,
)


class ContextOverflowDetectionTests(SimpleTestCase):
    def test_detects_common_provider_context_overflow_errors(self):
        messages = (
            "Range of input length should be [1, 98304]",
            "Request exceeds the context window",
            "maximum context length is 8192 tokens",
        )
        for message in messages:
            with self.subTest(message=message):
                self.assertTrue(is_context_overflow_error(Exception(message)))

    def test_excludes_throttling_and_rate_limit_errors(self):
        messages = (
            "ThrottlingException: Too many tokens, please wait",
            "Rate limit",
        )
        for message in messages:
            with self.subTest(message=message):
                self.assertFalse(is_context_overflow_error(Exception(message)))


class CurrentTurnCompactionTests(SimpleTestCase):
    def _messages_with_tool_groups(self, group_count, tokens_per_group=3000):
        messages = [
            {"role": "system", "content": "system prompt"},
            {"role": "user", "content": "question"},
        ]
        for index in range(group_count):
            call_id = f"call-{index}"
            messages.extend(
                [
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": call_id,
                                "type": "function",
                                "function": {"name": "knowledge_search", "arguments": "{}"},
                            }
                        ],
                    },
                    {
                        "role": "tool",
                        "tool_call_id": call_id,
                        "name": "knowledge_search",
                        "content": f"group-{index} " + ("word " * tokens_per_group),
                    },
                ]
            )
        return messages

    def test_compacts_current_turn_at_tool_group_boundaries(self):
        messages = self._messages_with_tool_groups(10)

        compacted = consolidate_messages(
            messages,
            max_tokens=16000,
            llm_caller=lambda _: "A compacted summary of earlier tool traffic.",
            extract_key_info=False,
        )

        self.assertLess(estimate_messages_tokens(compacted), 16000 * 0.8)
        self.assertIn({"role": "user", "content": "question"}, compacted)
        self.assertIn("group-9", "\n".join(message.get("content", "") for message in compacted))
        self.assertNotIn("group-0", "\n".join(message.get("content", "") for message in compacted))

        preceding_call_ids = set()
        for message in compacted:
            if message.get("role") == "assistant" and message.get("tool_calls"):
                preceding_call_ids.update(call["id"] for call in message["tool_calls"])
            elif message.get("role") == "tool":
                call_id = message["tool_call_id"]
                self.assertIn(call_id, preceding_call_ids)
                preceding_call_ids.remove(call_id)

        self.assertTrue(_group_history_messages(compacted))

    def test_current_turn_summary_preserves_history_and_output_order(self):
        messages = self._messages_with_tool_groups(10)
        old_history = [
            {"role": "user", "content": "older question"},
            {"role": "assistant", "content": "older answer"},
        ]
        messages[1:1] = old_history

        compacted = consolidate_messages(
            messages,
            max_tokens=16000,
            llm_caller=lambda _: "A compacted summary of earlier tool traffic.",
            extract_key_info=False,
        )

        self.assertEqual(compacted[0], messages[0])
        self.assertEqual(compacted[1]["role"], "system")
        self.assertEqual(compacted[2:4], old_history)
        self.assertEqual(compacted[4], {"role": "user", "content": "question"})
        self.assertEqual(compacted[5]["role"], "assistant")

    def test_current_turn_evidence_precedes_long_old_history_in_compaction_prompts(self):
        old_history = [
            {"role": "user", "content": "older question"},
            {"role": "assistant", "content": "old history " * 5000},
        ]
        self.assertGreater(len(old_history[1]["content"]), 8000)
        self.assertEqual(_build_compaction_text(old_history, []), _build_history_text(old_history))

        messages = [
            {"role": "system", "content": "system prompt"},
            *old_history,
            {"role": "user", "content": "current question"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call-old",
                        "type": "function",
                        "function": {"name": "knowledge_search", "arguments": "{}"},
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call-old",
                "name": "knowledge_search",
                "content": "CURRENT_TURN_MARKER " + ("evidence " * 6000),
            },
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call-new",
                        "type": "function",
                        "function": {"name": "knowledge_search", "arguments": "{}"},
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call-new",
                "name": "knowledge_search",
                "content": "recent evidence",
            },
        ]
        prompts = []
        responses = iter(
            [
                "Key information from earlier results.",
                "A concise summary preserving the current-turn evidence.",
            ]
        )

        def llm_caller(llm_messages):
            prompts.append(llm_messages[-1]["content"])
            return next(responses)

        consolidate_messages(messages, max_tokens=16000, llm_caller=llm_caller)

        self.assertEqual(len(prompts), 2)
        self.assertIn("CURRENT_TURN_MARKER", prompts[0])
        self.assertIn("CURRENT_TURN_MARKER", prompts[1])

    def test_prior_summary_precedes_long_current_turn_in_compaction_prompts(self):
        messages = self._messages_with_tool_groups(12)
        messages.insert(
            1,
            {
                "role": "system",
                "content": "PRIOR_SUMMARY_MARKER " + ("prior summary " * 12000),
            },
        )
        self.assertGreater(len(_build_history_text(messages[3:-4])), 8000)

        prompts = []
        responses = iter(
            [
                "Key information from the prior summary.",
                "A concise summary preserving prior and current evidence.",
            ]
        )

        def llm_caller(llm_messages):
            prompts.append(llm_messages[-1]["content"])
            return next(responses)

        consolidate_messages(messages, max_tokens=16000, llm_caller=llm_caller, force=True)

        self.assertEqual(len(prompts), 2)
        self.assertIn("PRIOR_SUMMARY_MARKER", prompts[0])
        self.assertIn("PRIOR_SUMMARY_MARKER", prompts[1])

    def test_force_skips_threshold_without_changing_default_under_threshold_behavior(self):
        messages = self._messages_with_tool_groups(2)
        self.assertLessEqual(estimate_messages_tokens(messages), int(16000 * 0.5))

        unchanged = consolidate_messages(
            messages,
            max_tokens=16000,
            llm_caller=lambda _: "A compacted summary of earlier tool traffic.",
            extract_key_info=False,
        )
        compacted = consolidate_messages(
            messages,
            max_tokens=16000,
            llm_caller=lambda _: "A compacted summary of earlier tool traffic.",
            extract_key_info=False,
            force=True,
        )

        self.assertIs(unchanged, messages)
        self.assertLess(len(compacted), len(messages))
        self.assertIn("group-1", "\n".join(message.get("content", "") for message in compacted))

    def test_sliding_window_compresses_current_turn_without_splitting_tool_groups(self):
        messages = self._messages_with_tool_groups(3)

        compacted = _sliding_window_compress(messages, max_tokens=4000)

        content = "\n".join(message.get("content", "") for message in compacted)
        self.assertIn({"role": "user", "content": "question"}, compacted)
        self.assertIn("group-2", content)
        self.assertNotIn("group-0", content)
        self.assertNotIn("group-1", content)

        preceding_call_ids = set()
        for message in compacted:
            if message.get("role") == "assistant" and message.get("tool_calls"):
                preceding_call_ids.update(call["id"] for call in message["tool_calls"])
            elif message.get("role") == "tool":
                call_id = message["tool_call_id"]
                self.assertIn(call_id, preceding_call_ids)
                preceding_call_ids.remove(call_id)
