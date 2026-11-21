import json

from django.test import Client, TestCase

from .models import Message, Session, Tenant
from .stream_manager import stream_manager


def parse_sse(response):
    """Return (id, event, payload) for every frame of a streaming response."""
    frames = []
    for raw in b"".join(response.streaming_content).decode().strip().split("\n\n"):
        fields = dict(line.split(": ", 1) for line in raw.split("\n"))
        frames.append((int(fields["id"]) if "id" in fields else None, fields["event"], json.loads(fields["data"])))
    return frames


class StreamResumeTests(TestCase):
    def setUp(self):
        self.client = Client()
        response = self.client.post(
            "/api/v1/auth/register",
            data=json.dumps({"username": "streamer", "email": "streamer@example.com", "password": "password123"}),
            content_type="application/json",
        )
        data = response.json()["data"]
        self.auth = {"HTTP_AUTHORIZATION": f"Bearer {data['token']}"}
        self.session = Session.objects.create(tenant=Tenant.objects.get(id=data["tenant"]["id"]), title="s")
        self.message = Message.objects.create(session=self.session, request_id="r1", role="assistant", content="", is_completed=False)
        stream = stream_manager.create_stream(self.message.id, self.session.id)
        self.addCleanup(stream_manager.remove_stream, self.message.id)
        stream.append_event("tool_call", {"name": "knowledge_search", "arguments": {"query": "q"}, "iteration": 1})
        stream.append_event("tool_result", {"name": "knowledge_search", "output": "hit", "duration_ms": 5})
        stream.append_event("thinking", {"content": "partial"})
        stream.set_final_result(content="final answer", refs=[{"chunk_id": "c-1"}])
        stream.append_event("complete", {"done": True, "content": "final answer"})

    def continue_stream(self, **headers):
        url = f"/api/v1/sessions/continue-stream/{self.session.id}?message_id={self.message.id}"
        return parse_sse(self.client.get(url, **self.auth, **headers))

    def test_first_connection_replays_every_event_with_its_offset_as_id(self):
        frames = self.continue_stream()

        self.assertEqual([event for _, event, _ in frames[:2]], ["message_start", "message"])
        self.assertEqual(frames[1][2]["response_type"], "agent_query")
        self.assertEqual([frame_id for frame_id, _, _ in frames[2:]], [0, 1, 2, 3, 3, 3])
        self.assertEqual([payload.get("response_type") for _, _, payload in frames[2:6]], ["tool_call", "tool_result", "answer", "answer"])

    def test_reconnect_with_last_event_id_only_sends_what_was_missed(self):
        frames = self.continue_stream(HTTP_LAST_EVENT_ID="1")

        # No start frames (they would reset what the client already rendered) and no tool events again.
        self.assertEqual([frame_id for frame_id, _, _ in frames], [2, 3, 3, 3])
        self.assertEqual(frames[0][2]["content"], "partial")
        self.assertEqual(frames[1][2]["content"], "final answer")
        self.assertEqual(frames[1][2]["knowledge_references"], [{"chunk_id": "c-1"}])
        self.assertEqual(frames[-1][1], "done")

    def test_reconnect_after_the_stream_is_gone_falls_back_to_the_saved_message(self):
        stream_manager.remove_stream(self.message.id)
        Message.objects.filter(id=self.message.id).update(content="saved answer")

        unfinished = self.continue_stream(HTTP_LAST_EVENT_ID="2")
        self.assertEqual([event for _, event, _ in unfinished], ["done"])

        Message.objects.filter(id=self.message.id).update(is_completed=True)
        finished = self.continue_stream(HTTP_LAST_EVENT_ID="2")
        self.assertEqual(finished[0][2]["content"], "saved answer")
        self.assertEqual(finished[-1][1], "done")

    def test_other_tenants_cannot_replay_the_stream(self):
        response = self.client.post(
            "/api/v1/auth/register",
            data=json.dumps({"username": "intruder", "email": "intruder@example.com", "password": "password123"}),
            content_type="application/json",
        )
        token = response.json()["data"]["token"]
        url = f"/api/v1/sessions/continue-stream/{self.session.id}?message_id={self.message.id}"

        self.assertEqual(self.client.get(url, HTTP_AUTHORIZATION=f"Bearer {token}").status_code, 404)
