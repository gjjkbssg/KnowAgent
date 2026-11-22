import json

from django.test import Client, TestCase

from personal_knowledge_base.models import GenericResource, ModelConfig, Tenant, TenantMember


class ServiceLayerRestMethodDispatchTests(TestCase):
    def setUp(self):
        self.client = Client()
        response = self.client.post("/api/v1/auth/auto-setup", content_type="application/json")
        self.assertIn(response.status_code, (200, 201))
        self.headers = {"HTTP_AUTHORIZATION": f"Bearer {response.json()['data']['token']}"}
        self.tenant = Tenant.objects.get(id=1)

    def _request(self, method, path, data=None):
        kwargs = dict(self.headers)
        if data is not None:
            kwargs.update(data=json.dumps(data), content_type="application/json")
        return getattr(self.client, method.lower())(path, **kwargs)

    def test_tenant_method_fallthroughs(self):
        response = self._request("PUT", "/api/v1/tenants", {"name": "created by PUT"})
        self.assertEqual(response.status_code, 201)
        created = Tenant.objects.get(id=response.json()["data"]["id"])
        self.assertEqual(created.name, "created by PUT")

        response = self._request(
            "PATCH",
            f"/api/v1/tenants/{self.tenant.id}",
            {"name": "updated by PATCH"},
        )
        self.assertEqual(response.status_code, 200)
        self.tenant.refresh_from_db()
        self.assertEqual(self.tenant.name, "updated by PATCH")

        member = TenantMember.objects.get(tenant=self.tenant)
        response = self._request(
            "PATCH",
            f"/api/v1/tenants/{self.tenant.id}/members/{member.user_id}",
            {"role": "viewer"},
        )
        self.assertEqual(response.status_code, 200)
        member.refresh_from_db()
        self.assertEqual(member.role, "viewer")

        response = self._request("PATCH", "/api/v1/tenants/kv/agent-config", {"value": {"mode": "fallback"}})
        self.assertEqual(response.status_code, 200)
        self.tenant.refresh_from_db()
        self.assertEqual(self.tenant.agent_config, {"mode": "fallback"})

    def test_agent_resource_method_fallthroughs(self):
        response = self._request("PUT", "/api/v1/web-search-providers", {"name": "created by PUT"})
        self.assertEqual(response.status_code, 201)
        item_id = response.json()["data"]["id"]
        resource = GenericResource.objects.get(id=item_id)
        self.assertEqual(resource.name, "created by PUT")

        response = self._request(
            "PATCH",
            f"/api/v1/web-search-providers/{item_id}",
            {"name": "updated by PATCH"},
        )
        self.assertEqual(response.status_code, 200)
        resource.refresh_from_db()
        self.assertEqual(resource.name, "updated by PATCH")

    def test_model_method_fallthroughs(self):
        response = self._request(
            "PUT",
            "/api/v1/models",
            {"id": "method-dispatch-model", "type": "KnowledgeQA", "name": "created by PUT"},
        )
        self.assertEqual(response.status_code, 201)
        model = ModelConfig.objects.get(id="method-dispatch-model")

        response = self._request(
            "PATCH",
            f"/api/v1/models/{model.id}",
            {"name": "updated by PATCH"},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["data"]["name"], "updated by PATCH")

        response = self._request(
            "PATCH",
            f"/api/v1/models/{model.id}/credentials",
            {"api_key": "method-dispatch-secret"},
        )
        self.assertEqual(response.status_code, 200)
        model.refresh_from_db()
        self.assertEqual(model.parameters["api_key"], "method-dispatch-secret")
