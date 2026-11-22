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


class EmbedInvitationContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = new_client(admin()["token"], timeout=180)

    @classmethod
    def tearDownClass(cls):
        cls.client.close()

    def call(self, responses, label, method, path, expected_status=200, **kwargs):
        response = self.client.request(method, path, **kwargs)
        responses[label] = capture(self, label, response, expected_status)
        if "application/json" in response.headers.get("content-type", ""):
            return response.json()
        return None

    def test_embed_public_routes_contract(self):
        reset_requests()
        responses = {}
        agent = self.call(
            responses,
            "create_embed_agent",
            "POST",
            f"{API}/agents",
            201,
            json={"name": "Contract embed agent"},
        )
        channel = self.call(
            responses,
            "create_agent_embed_channel",
            "POST",
            f"{API}/embed-channels",
            201,
            json={
                "name": "Contract embed channel",
                "agent_id": agent["data"]["id"],
                "enabled": True,
            },
        )
        channel_id = channel["data"]["id"]

        self.call(
            responses,
            "embed_exchange",
            "POST",
            f"{API}/embed/{channel_id}/exchange",
            200,
            json={},
        )
        self.call(
            responses,
            "embed_config",
            "GET",
            f"{API}/embed/{channel_id}/config",
            200,
        )
        self.call(
            responses,
            "embed_suggested_questions",
            "GET",
            f"{API}/embed/{channel_id}/suggested-questions",
            200,
        )

        response_types = {}
        for kind, endpoint in (
            ("knowledge", "knowledge-chat"),
            ("agent", "agent-chat"),
        ):
            session = self.call(
                responses,
                f"create_embed_{kind}_session",
                "POST",
                f"{API}/embed/{channel_id}/sessions",
                201,
                json={},
            )
            session_id = session["data"]["id"]
            response = self.client.post(
                f"{API}/embed/{channel_id}/{endpoint}/{session_id}",
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
            self.assertTrue(frames, f"{kind} embed chat returned no SSE frames")
            ids = [frame["id"] for frame in frames if isinstance(frame["id"], int)]
            self.assertEqual(ids, sorted(ids))
            self.assertEqual(frames[-1]["event"], "done")
            types = list(
                dict.fromkeys(
                    frame["data"]["response_type"]
                    for frame in frames
                    if isinstance(frame["data"], dict)
                    and frame["data"].get("response_type")
                )
            )
            answer = "".join(
                frame["data"].get("content", "")
                for frame in frames
                if isinstance(frame["data"], dict)
                and frame["data"].get("response_type") == "answer"
            )
            self.assertIn(FIXED_ANSWER, answer)
            response_types[kind] = types
            responses[f"embed_{kind}_chat"] = {
                "status": response.status_code,
                "content_type": response.headers.get("content-type", ""),
                "frames": frames,
            }

            loaded = self.call(
                responses,
                f"embed_{kind}_messages_load",
                "GET",
                f"{API}/embed/{channel_id}/messages/{session_id}/load",
                200,
            )
            messages = loaded["data"].get("messages", [])
            self.assertTrue(
                any(message["role"] == "assistant" for message in messages),
                f"{kind} embed assistant message missing from message history",
            )
            self.call(
                responses,
                f"embed_{kind}_events",
                "GET",
                f"{API}/embed/{channel_id}/sessions/{session_id}/events",
                200,
            )
            self.call(
                responses,
                f"embed_{kind}_stop",
                "POST",
                f"{API}/embed/{channel_id}/sessions/{session_id}/stop",
                200,
                json={},
            )

        if not os.environ.get("CONTRACT_BASE_URL"):
            chat_requests = [
                request
                for request in received_requests()
                if request["path"].endswith("/chat/completions")
            ]
            self.assertGreaterEqual(len(chat_requests), 2)
            self.assertIn(
                True, [request["payload"].get("stream") for request in chat_requests]
            )

        assert_snapshot(self, "embed_public", responses)
        assert_value_snapshot(self, "embed_sse_response_types", response_types)

    def test_account_invitation_and_leave_routes_contract(self):
        responses = {}
        me = self.call(
            responses,
            "invitation_owner_me",
            "GET",
            f"{API}/auth/me",
            200,
        )
        tenant_id = me["data"]["tenant"]["id"]

        self.call(
            responses,
            "tenant_invitations_list",
            "GET",
            f"{API}/tenants/{tenant_id}/invitations",
            200,
        )
        first_invitation = self.call(
            responses,
            "tenant_invitation_create_accept",
            "POST",
            f"{API}/tenants/{tenant_id}/invitations",
            201,
            json={"email": "contract-accept@example.test", "role": "viewer"},
        )
        second_invitation = self.call(
            responses,
            "tenant_invitation_create_decline",
            "POST",
            f"{API}/tenants/{tenant_id}/invitations",
            201,
            json={"email": "contract-decline@example.test", "role": "viewer"},
        )
        self.call(
            responses,
            "tenant_invite_links",
            "GET",
            f"{API}/tenants/{tenant_id}/invite-links",
            200,
        )
        self.call(
            responses,
            "my_invitations_list",
            "GET",
            f"{API}/me/invitations",
            200,
        )
        self.call(
            responses,
            "invitation_accept",
            "POST",
            f"{API}/me/invitations/{first_invitation['data']['id']}/accept",
            200,
            json={},
        )
        self.call(
            responses,
            "invitation_decline",
            "POST",
            f"{API}/me/invitations/{second_invitation['data']['id']}/decline",
            200,
            json={},
        )

        member = self.call(
            responses,
            "register_member_for_leave",
            "POST",
            f"{API}/auth/register",
            201,
            json={
                "username": "contract-invitation-member",
                "email": "contract-invitation-member@example.test",
                "password": "Contract-member-password!",
            },
        )
        self.call(
            responses,
            "add_non_owner_member",
            "POST",
            f"{API}/tenants/{tenant_id}/members",
            200,
            json={"user_id": member["data"]["user"]["id"], "role": "viewer"},
        )
        with new_client(member["data"]["token"]) as member_client:
            leave = member_client.post(f"{API}/tenants/{tenant_id}/leave", json={})
            responses["non_owner_leave"] = capture(self, "non_owner_leave", leave, 200)

        assert_snapshot(self, "account_invitations", responses)
