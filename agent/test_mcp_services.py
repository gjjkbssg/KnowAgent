import json

from django.test import Client, RequestFactory, TestCase, override_settings

from personal_knowledge_base.models import GenericResource, Tenant
from personal_knowledge_base.test_mcp_client import FakeMCPServer


class MCPServiceManagementTests(TestCase):
    def setUp(self):
        self.client = Client()
        response = self.client.post("/api/v1/auth/auto-setup", content_type="application/json")
        self.assertIn(response.status_code, (200, 201))
        self.headers = {"HTTP_AUTHORIZATION": f"Bearer {response.json()['data']['token']}"}
        self.tenant = Tenant.objects.first()

    def create_service(self, tenant=None, **data):
        payload = {"url": "http://127.0.0.1:8000/mcp", "enabled": True, **data}
        return GenericResource.objects.create(
            tenant=tenant or self.tenant,
            resource_type="mcp_services",
            name=payload.get("name", "Local MCP"),
            data=payload,
        )

    def test_cross_tenant_resource_get_put_delete_are_not_found(self):
        other_tenant = Tenant.objects.create(name="other MCP tenant", api_key="other-mcp-tenant-key")
        resource = self.create_service(other_tenant, api_key="secret-key")

        get_response = self.client.get(f"/api/v1/mcp-services/{resource.id}", **self.headers)
        put_response = self.client.put(
            f"/api/v1/mcp-services/{resource.id}",
            data=json.dumps({"name": "changed"}),
            content_type="application/json",
            **self.headers,
        )
        delete_response = self.client.delete(f"/api/v1/mcp-services/{resource.id}", **self.headers)

        self.assertEqual([get_response.status_code, put_response.status_code, delete_response.status_code], [404, 404, 404])
        resource.refresh_from_db()
        self.assertEqual(resource.name, "Local MCP")
        self.assertIsNone(resource.deleted_at)

    def test_api_key_is_redacted_and_empty_update_preserves_it(self):
        resource = self.create_service(api_key="stored-secret")

        response = self.client.get(f"/api/v1/mcp-services/{resource.id}", **self.headers)
        payload = response.content.decode()
        self.assertNotIn("stored-secret", payload)
        self.assertNotIn('"api_key"', payload)
        self.assertTrue(response.json()["data"]["api_key_set"])
        listed = self.client.get("/api/v1/mcp-services", **self.headers)
        self.assertNotIn("stored-secret", listed.content.decode())

        response = self.client.put(
            f"/api/v1/mcp-services/{resource.id}",
            data=json.dumps({"name": "Updated MCP", "api_key": ""}),
            content_type="application/json",
            **self.headers,
        )
        self.assertEqual(response.status_code, 200)
        resource.refresh_from_db()
        self.assertEqual(resource.data["api_key"], "stored-secret")
        self.assertNotIn("stored-secret", response.content.decode())
        self.assertTrue(response.json()["data"]["api_key_set"])
        response = self.client.put(
            f"/api/v1/mcp-services/{resource.id}",
            data=json.dumps({"name": "Still updated"}),
            content_type="application/json",
            **self.headers,
        )
        resource.refresh_from_db()
        self.assertEqual(resource.data["api_key"], "stored-secret")
        self.assertNotIn("stored-secret", response.content.decode())

    def test_test_and_tools_actions_use_streamable_http_server(self):
        server = FakeMCPServer()
        self.addCleanup(server.close)
        resource = self.create_service(url=server.url, api_key="action-key")
        with override_settings(MCP_ALLOW_PRIVATE_NETWORK=True):
            test_response = self.client.post(f"/api/v1/mcp-services/{resource.id}/test", **self.headers)
            tools_response = self.client.get(f"/api/v1/mcp-services/{resource.id}/tools", **self.headers)

        self.assertEqual(test_response.status_code, 200)
        self.assertEqual(
            test_response.json()["data"],
            {"status": "ok", "available": True, "protocol_version": "2025-06-18", "tool_count": 1},
        )
        self.assertEqual(tools_response.status_code, 200)
        self.assertEqual(
            tools_response.json()["data"]["items"][0],
            {
                "name": "echo",
                "description": "Echo text",
                "input_schema": {"type": "object", "properties": {"text": {"type": "string"}}},
                "annotations": {"title": "Echo"},
                "requires_approval": True,
            },
        )
        self.assertNotIn("action-key", test_response.content.decode())

    def test_mcp_service_default_approval_mode_is_always(self):
        from .views import default_resource_payload

        self.assertEqual(default_resource_payload("mcp_services", {"name": "service"})["approval_mode"], "always")

    def test_resource_output_omits_only_the_default_approval_mode(self):
        resource = self.create_service(approval_mode="always")
        default_response = self.client.get(f"/api/v1/mcp-services/{resource.id}", **self.headers)
        self.assertNotIn("approval_mode", default_response.json()["data"])

        resource.data["approval_mode"] = "never"
        resource.save(update_fields=["data", "updated_at"])
        configured_response = self.client.get(f"/api/v1/mcp-services/{resource.id}", **self.headers)
        self.assertEqual(configured_response.json()["data"]["approval_mode"], "never")

    def test_test_action_reports_connection_failure_with_http_200(self):
        server = FakeMCPServer()
        self.addCleanup(server.close)
        server.state.protocol_version = "2099-01-01"
        resource = self.create_service(url=server.url, api_key="failure-key")
        with override_settings(MCP_ALLOW_PRIVATE_NETWORK=True):
            response = self.client.post(f"/api/v1/mcp-services/{resource.id}/test", **self.headers)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["data"]["status"], "error")
        self.assertFalse(response.json()["data"]["available"])
        self.assertIn("Unsupported MCP protocol version", response.json()["data"]["error"])

    def test_mcp_actions_are_tenant_scoped(self):
        other_tenant = Tenant.objects.create(name="other action tenant", api_key="other-action-key")
        resource = self.create_service(other_tenant)

        for action in ("test", "tools"):
            response = self.client.post(f"/api/v1/mcp-services/{resource.id}/{action}", **self.headers)
            self.assertEqual(response.status_code, 404)

    def test_system_settings_uses_tenant_fallback_for_item_lookup(self):
        from unittest.mock import patch

        from .views import generic_collection

        resource = GenericResource.objects.create(
            tenant=self.tenant,
            resource_type="system_settings",
            name="setting",
            data={"value": "kept"},
        )
        request = RequestFactory().get(f"/api/v1/system/admin/settings/{resource.id}")
        with patch("agent.views.auth_context", return_value=(None, None)):
            response = generic_collection(request, "system_settings", item_id=resource.id)

        self.assertEqual(response.status_code, 200)
