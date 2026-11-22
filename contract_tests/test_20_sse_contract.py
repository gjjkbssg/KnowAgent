import json
import os
import unittest

from contract_tests.assertions import assert_snapshot, assert_value_snapshot, capture
from contract_tests.fake_llm import FIXED_ANSWER, received_requests, reset_requests
from contract_tests.harness import admin, new_client


API = "/api/v1"


def parse_sse(text):
    frames = []
    for raw_frame in text.split("\n\n"):
        if not raw_frame.strip():
            continue
        fields = {}
        data_lines = []
        for line in raw_frame.splitlines():
            if line.startswith("data:"):
                data_lines.append(line.partition(":")[2].lstrip())
            elif ":" in line:
                key, value = line.split(":", 1)
                fields[key] = value.lstrip()
        payload = json.loads("\n".join(data_lines)) if data_lines else {}
        raw_id = fields.get("id")
        try:
            frame_id = int(raw_id) if raw_id is not None else None
        except ValueError:
            frame_id = raw_id
        frames.append(
            {"id": frame_id, "event": fields.get("event", "message"), "data": payload}
        )
    return frames


def assert_monotonic_ids(testcase, frames):
    ids = [frame["id"] for frame in frames]
    testcase.assertTrue(all(isinstance(frame_id, int) for frame_id in ids), ids)
    testcase.assertEqual(ids, sorted(set(ids)), "SSE ids must be unique and increasing")


class SseApiContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = new_client(admin()["token"], timeout=180)

    @classmethod
    def tearDownClass(cls):
        cls.client.close()

    def exercise_chat(self, responses, kind):
        created = self.client.post(
            f"{API}/sessions",
            json={"title": f"Contract {kind} stream"},
        )
        created_summary = capture(self, f"create_{kind}_session", created, 201)
        session_id = created_summary["body"]["data"]["id"]
        endpoint = "agent-chat" if kind == "agent" else "knowledge-chat"
        response = self.client.post(
            f"{API}/{endpoint}/{session_id}",
            headers={"Accept": "text/event-stream"},
            json={
                "query": "Explain the contract fixture.",
                "stream": True,
                "enable_memory": False,
                "knowledge_base_ids": [],
            },
        )
        self.assertEqual(response.status_code, 200, response.text[:500])
        self.assertTrue(
            response.headers.get("content-type", "").startswith("text/event-stream"),
            response.headers.get("content-type"),
        )
        frames = parse_sse(response.text)
        self.assertTrue(frames, f"{kind} chat returned no SSE frames")
        assert_monotonic_ids(self, frames)
        event_ids = [frame["id"] for frame in frames if isinstance(frame["id"], int)]
        self.assertEqual(frames[-1]["event"], "done")
        response_types = list(
            dict.fromkeys(
                frame["data"]["response_type"]
                for frame in frames
                if isinstance(frame["data"], dict)
                and frame["data"].get("response_type")
            )
        )
        answer_text = "".join(
            frame["data"].get("content", "")
            for frame in frames
            if isinstance(frame["data"], dict)
            and frame["data"].get("response_type") == "answer"
        )
        self.assertIn(FIXED_ANSWER, answer_text)

        done_frame = frames[-1]
        message_id = done_frame["data"].get("message_id")
        self.assertTrue(message_id)
        responses[f"{kind}_chat"] = {
            "status": response.status_code,
            "content_type": response.headers.get("content-type", ""),
            "frames": [
                {
                    "id": frame["id"],
                    "event": frame["event"],
                    "data": frame["data"],
                }
                for frame in frames
            ],
            "distinct_response_types": response_types,
        }

        loaded = self.client.get(f"{API}/messages/{session_id}/load")
        responses[f"{kind}_messages_load"] = capture(
            self, f"{kind}_messages_load", loaded, 200
        )
        loaded_messages = loaded.json()["data"]["messages"]
        self.assertTrue(
            any(message["role"] == "assistant" for message in loaded_messages),
            f"{kind} assistant message missing from message history",
        )

        last_event_id = 0
        resumed = self.client.get(
            f"{API}/sessions/continue-stream/{session_id}",
            params={"message_id": message_id},
            headers={"Last-Event-ID": str(last_event_id)},
        )
        self.assertEqual(resumed.status_code, 200, resumed.text[:500])
        self.assertTrue(
            resumed.headers.get("content-type", "").startswith("text/event-stream"),
            resumed.headers.get("content-type"),
        )
        resumed_frames = parse_sse(resumed.text)
        assert_monotonic_ids(self, resumed_frames)
        resumed_ids = [
            frame["id"] for frame in resumed_frames if isinstance(frame["id"], int)
        ]
        self.assertTrue(all(frame_id > last_event_id for frame_id in resumed_ids))
        self.assertEqual(resumed_frames[-1]["event"], "done")
        for resume_after in [None] + event_ids:
            replay = self.client.get(
                f"{API}/sessions/continue-stream/{session_id}",
                params={"message_id": message_id},
                headers={} if resume_after is None else {"Last-Event-ID": str(resume_after)},
            )
            self.assertEqual(replay.status_code, 200, replay.text[:500])
            expected = frames if resume_after is None else [
                frame for frame in frames if frame["id"] > resume_after
            ]
            self.assertEqual(
                parse_sse(replay.text),
                expected,
                f"{kind} replay after Last-Event-ID {resume_after} differs from the live stream",
            )
        responses[f"{kind}_event_ids"] = {
            "chat": event_ids,
            "continue_stream": resumed_ids,
        }
        responses[f"{kind}_continue_stream"] = {
            "status": resumed.status_code,
            "content_type": resumed.headers.get("content-type", ""),
            "frames": resumed_frames,
        }

    def test_knowledge_chat_and_agent_chat_stream_contract(self):
        reset_requests()
        responses = {}
        self.exercise_chat(responses, "knowledge")
        self.exercise_chat(responses, "agent")
        chat_requests = [
            request
            for request in received_requests()
            if request["path"].endswith("/chat/completions")
        ]
        if not os.environ.get("CONTRACT_BASE_URL"):
            self.assertTrue(chat_requests, "The fake LLM did not receive a chat request")
            self.assertIn(True, [request["payload"].get("stream") for request in chat_requests])
        assert_snapshot(self, "sse", responses)
        assert_value_snapshot(
            self,
            "sse_response_types",
            {
                "knowledge": responses["knowledge_chat"]["distinct_response_types"],
                "agent": responses["agent_chat"]["distinct_response_types"],
            },
        )
        assert_value_snapshot(
            self,
            "sse_event_ids",
            {
                "knowledge": responses["knowledge_event_ids"],
                "agent": responses["agent_event_ids"],
            },
        )
