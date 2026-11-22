import json

from django.test import Client, TestCase

from .models import Message, Session, Tenant
from .stream_manager import stream_manager


def parse_sse(response):
    """Return (id, event, payload) for every frame of a streaming response."""
    frames = []
    for raw in b"".join(response.streaming_content).decode().strip().split("\n\n"):
        if not raw:
            continue
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

    def test_first_connection_numbers_every_frame_uniquely(self):
        frames = self.continue_stream()

        self.assertEqual([frame_id for frame_id, _, _ in frames], list(range(8)))
        self.assertEqual([event for _, event, _ in frames[:2]], ["message_start", "message"])
        self.assertEqual(frames[1][2]["response_type"], "agent_query")
        self.assertEqual(
            [payload.get("response_type") for _, _, payload in frames[2:7]],
            ["tool_call", "tool_result", "answer", "answer", "complete"],
        )
        self.assertEqual(frames[-1][1], "done")

    def test_reconnect_with_last_event_id_only_sends_what_was_missed(self):
        frames = self.continue_stream(HTTP_LAST_EVENT_ID="3")

        # No start frames (they would reset what the client already rendered) and no tool events again.
        self.assertEqual([frame_id for frame_id, _, _ in frames], [4, 5, 6, 7])
        self.assertEqual(frames[0][2]["content"], "partial")
        self.assertEqual(frames[1][2]["content"], "final answer")
        self.assertEqual(frames[1][2]["knowledge_references"], [{"chunk_id": "c-1"}])
        self.assertEqual(frames[-1][1], "done")

    def test_resuming_from_any_id_returns_exactly_the_frames_after_it(self):
        full = self.continue_stream()
        for completed in (False, True):
            Message.objects.filter(id=self.message.id).update(is_completed=completed)
            for last_id in [-1] + [frame_id for frame_id, _, _ in full]:
                with self.subTest(completed=completed, last_event_id=last_id):
                    headers = {} if last_id < 0 else {"HTTP_LAST_EVENT_ID": str(last_id)}
                    self.assertEqual(self.continue_stream(**headers), full[last_id + 1:])

    def test_reconnect_after_the_stream_is_gone_falls_back_to_the_saved_message(self):
        stream_manager.remove_stream(self.message.id)
        Message.objects.filter(id=self.message.id).update(content="saved answer")

        unfinished = self.continue_stream(HTTP_LAST_EVENT_ID="2")
        self.assertEqual([(frame_id, event) for frame_id, event, _ in unfinished], [(3, "done")])

        Message.objects.filter(id=self.message.id).update(is_completed=True)
        finished = self.continue_stream(HTTP_LAST_EVENT_ID="2")
        self.assertEqual([frame_id for frame_id, _, _ in finished], [3, 4, 5])
        self.assertEqual(finished[0][2]["content"], "saved answer")
        self.assertEqual(finished[-1][1], "done")

        fresh = self.continue_stream()
        self.assertEqual([frame_id for frame_id, _, _ in fresh], [0, 1, 2])

    def test_pending_approval_replay_keeps_live_ids(self):
        from datetime import timedelta

        from django.utils import timezone

        from .models import AgentToolApproval

        message = Message.objects.create(session=self.session, request_id="r2", role="assistant", content="", is_completed=False)
        stream = stream_manager.create_stream(message.id, self.session.id)
        self.addCleanup(stream_manager.remove_stream, message.id)
        stream.append_event("thinking", {"content": "need a tool"})
        approval = AgentToolApproval.objects.create(
            tenant=self.session.tenant,
            session=self.session,
            message=message,
            tool_calls=[{"id": "c1", "name": "tool"}],
            checkpoint={},
            expires_at=timezone.now() + timedelta(minutes=5),
        )
        stream.append_event("approval_required", {"approval": {"id": approval.id, "status": "pending"}})
        url = f"/api/v1/sessions/continue-stream/{self.session.id}?message_id={message.id}"

        full = parse_sse(self.client.get(url, **self.auth))
        self.assertEqual([frame_id for frame_id, _, _ in full], [0, 1, 2, 3, 4])
        self.assertEqual(full[3][2]["response_type"], "tool_approval_required")
        resumed = parse_sse(self.client.get(url, **self.auth, HTTP_LAST_EVENT_ID="2"))
        self.assertEqual(resumed, full[3:])

        stream_manager.remove_stream(message.id)
        durable = parse_sse(self.client.get(url, **self.auth, HTTP_LAST_EVENT_ID="2"))
        self.assertEqual([(frame_id, event) for frame_id, event, _ in durable], [(3, "message"), (4, "done")])
        self.assertEqual(durable[0][2]["approval"]["id"], approval.id)

    def test_idle_timeout_frames_do_not_move_the_resume_point(self):
        from chat.streaming import stream_message_events

        message = Message.objects.create(session=self.session, request_id="r3", role="assistant", content="", is_completed=False)
        stream = stream_manager.create_stream(message.id, self.session.id)
        self.addCleanup(stream_manager.remove_stream, message.id)
        stream.append_event("thinking", {"content": "slow"})

        frames = "".join(stream_message_events(message, idle_timeout=0.15)).strip().split("\n\n")
        ids = [line[4:] for frame in frames for line in frame.split("\n") if line.startswith("id: ")]
        self.assertEqual(ids, ["0", "1", "2"])
        self.assertIn("等待超时", frames[-2])
        self.assertNotIn("id:", frames[-2] + frames[-1])
    def test_other_tenants_cannot_replay_the_stream(self):
        response = self.client.post(
            "/api/v1/auth/register",
            data=json.dumps({"username": "intruder", "email": "intruder@example.com", "password": "password123"}),
            content_type="application/json",
        )
        token = response.json()["data"]["token"]
        url = f"/api/v1/sessions/continue-stream/{self.session.id}?message_id={self.message.id}"

        self.assertEqual(self.client.get(url, HTTP_AUTHORIZATION=f"Bearer {token}").status_code, 404)
