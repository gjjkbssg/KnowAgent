import json
import uuid

from django.test import Client, TestCase

from personal_knowledge_base.authentication import hash_password, issue_tokens
from personal_knowledge_base.models import Tenant, TenantMember, User


class TenantAdminAuthorizationTests(TestCase):
    def setUp(self):
        self.client = Client()
        self.target = Tenant.objects.create(
            name="Target tenant",
            api_key=f"target-{uuid.uuid4().hex}",
        )
        self.other = Tenant.objects.create(
            name="Other tenant",
            api_key=f"other-{uuid.uuid4().hex}",
        )
        self.owner, self.owner_headers = self._create_user(self.target, "owner")
        self.other_owner, self.other_owner_headers = self._create_user(self.target, "owner")
        self.viewer, self.viewer_headers = self._create_user(self.target, "viewer")
        self.admin, self.admin_headers = self._create_user(self.target, "admin")
        self.outsider, self.outsider_headers = self._create_user(self.other, "owner")
        self.system_admin, self.system_admin_headers = self._create_user(
            self.other,
            system_admin=True,
        )
        self.unaffiliated, _ = self._create_user(self.other)

    def _create_user(self, tenant, role=None, *, system_admin=False):
        suffix = uuid.uuid4().hex
        user = User.objects.create(
            username=f"user-{suffix}",
            email=f"user-{suffix}@example.test",
            password_hash="unused",
            tenant=tenant,
            is_system_admin=system_admin,
        )
        if role:
            TenantMember.objects.create(user=user, tenant=tenant, role=role)
        access_token, _ = issue_tokens(user)
        return user, {"HTTP_AUTHORIZATION": f"Bearer {access_token}"}

    def _request(self, method, path, data=None, headers=None):
        kwargs = dict(headers or {})
        if data is not None:
            kwargs.update(data=json.dumps(data), content_type="application/json")
        return getattr(self.client, method.lower())(path, **kwargs)

    def _detail(self, tenant_id=None):
        return f"/api/v1/tenants/{tenant_id or self.target.id}"

    def _members(self, tenant_id=None):
        return f"{self._detail(tenant_id)}/members"

    def _member(self, user_id, tenant_id=None):
        return f"{self._members(tenant_id)}/{user_id}"

    def _api_key(self, tenant_id=None):
        return f"{self._detail(tenant_id)}/api-key"

    def _assert_error(self, response, status, message):
        self.assertEqual(response.status_code, status)
        self.assertEqual(response["Content-Type"], "application/json")
        self.assertEqual(response.json()["message"], message)
        self.assertEqual(response.json()["error"]["message"], message)

    def _leave_single_active_owner(self):
        TenantMember.objects.filter(
            tenant=self.target,
            role="owner",
        ).exclude(user=self.owner).delete()

    def test_unauthenticated_requests_are_rejected(self):
        requests = [
            ("GET", self._detail(), None),
            ("GET", self._members(), None),
            ("POST", self._api_key(), None),
        ]
        for method, path, data in requests:
            with self.subTest(method=method, path=path):
                self._assert_error(self._request(method, path, data), 401, "unauthorized")

    def test_nonmembers_get_json_not_found_for_existing_and_missing_tenants(self):
        requests = [
            ("GET", self._detail(), None),
            ("GET", self._members(), None),
            ("PATCH", self._detail(), {"name": "intrusion"}),
            ("GET", self._detail(self.target.id + 10000), None),
        ]
        for method, path, data in requests:
            with self.subTest(method=method, path=path):
                self._assert_error(
                    self._request(method, path, data, self.outsider_headers),
                    404,
                    "not found",
                )

    def test_viewer_can_read_but_cannot_mutate_tenant_or_members(self):
        detail = self._request("GET", self._detail(), headers=self.viewer_headers)
        self.assertEqual(detail.status_code, 200)
        self.assertEqual(detail.json()["data"]["api_key"], "")
        self.assertEqual(
            self._request("GET", self._members(), headers=self.viewer_headers).status_code,
            200,
        )
        tenant_list = self._request(
            "GET",
            "/api/v1/tenants",
            headers=self.viewer_headers,
        )
        self.assertEqual(tenant_list.status_code, 200)
        self.assertEqual(tenant_list.json()["data"]["items"][0]["api_key"], "")
        requests = [
            ("PATCH", self._detail(), {"name": "not allowed"}),
            ("DELETE", self._detail(), None),
            ("POST", self._api_key(), None),
            ("POST", self._members(), {"email": self.unaffiliated.email}),
            ("PATCH", self._member(self.owner.id), {"role": "viewer"}),
            ("DELETE", self._member(self.owner.id), None),
        ]
        for method, path, data in requests:
            with self.subTest(method=method, path=path):
                self._assert_error(
                    self._request(method, path, data, self.viewer_headers),
                    403,
                    "forbidden",
                )

    def test_admin_can_manage_but_cannot_escalate_or_change_owner_membership(self):
        self.assertEqual(
            self._request(
                "PATCH",
                self._detail(),
                {"name": "managed by admin"},
                self.admin_headers,
            ).status_code,
            200,
        )
        detail = self._request("GET", self._detail(), headers=self.admin_headers)
        self.assertEqual(detail.json()["data"]["api_key"], self.target.api_key)
        tenant_list = self._request(
            "GET",
            "/api/v1/tenants",
            headers=self.admin_headers,
        )
        self.assertEqual(
            tenant_list.json()["data"]["items"][0]["api_key"],
            self.target.api_key,
        )
        rotated = self._request("POST", self._api_key(), headers=self.admin_headers)
        self.assertEqual(rotated.status_code, 200)
        self.target.refresh_from_db()
        self.assertEqual(rotated.json()["data"]["api_key"], self.target.api_key)
        added = self._request(
            "POST",
            self._members(),
            {"email": self.unaffiliated.email, "role": "viewer"},
            self.admin_headers,
        )
        self.assertEqual(added.status_code, 200)
        self.assertEqual(added.json()["data"]["role"], "viewer")

        escalation = self._request(
            "POST",
            self._members(),
            {"email": self.unaffiliated.email, "role": "owner"},
            self.admin_headers,
        )
        self._assert_error(escalation, 403, "forbidden")
        update_owner = self._request(
            "PATCH",
            self._member(self.owner.id),
            {"role": "viewer"},
            self.admin_headers,
        )
        self._assert_error(update_owner, 403, "forbidden")
        remove_owner = self._request(
            "DELETE",
            self._member(self.owner.id),
            headers=self.admin_headers,
        )
        self._assert_error(remove_owner, 403, "forbidden")
        delete_tenant = self._request("DELETE", self._detail(), headers=self.admin_headers)
        self._assert_error(delete_tenant, 403, "forbidden")

    def test_owner_can_manage_members_and_delete_tenant(self):
        self.assertEqual(
            self._request(
                "PATCH",
                self._detail(),
                {"name": "managed by owner"},
                self.owner_headers,
            ).status_code,
            200,
        )
        detail = self._request("GET", self._detail(), headers=self.owner_headers)
        self.assertEqual(detail.json()["data"]["api_key"], self.target.api_key)
        self.assertEqual(
            self._request("POST", self._api_key(), headers=self.owner_headers).status_code,
            200,
        )
        added = self._request(
            "POST",
            self._members(),
            {"email": self.unaffiliated.email, "role": "viewer"},
            self.owner_headers,
        )
        self.assertEqual(added.status_code, 200)

        grant_owner = self._request(
            "PATCH",
            self._member(self.unaffiliated.id),
            {"role": "owner"},
            self.owner_headers,
        )
        self.assertEqual(grant_owner.status_code, 200)
        self.assertEqual(grant_owner.json()["data"]["role"], "owner")

        invalid_add = self._request(
            "POST",
            self._members(),
            {"email": self.unaffiliated.email, "role": "superuser"},
            self.owner_headers,
        )
        self._assert_error(invalid_add, 400, "invalid role")
        invalid_update = self._request(
            "PATCH",
            self._member(self.viewer.id),
            {"role": "superuser"},
            self.owner_headers,
        )
        self._assert_error(invalid_update, 400, "invalid role")

        update_owner = self._request(
            "PATCH",
            self._member(self.other_owner.id),
            {"role": "admin"},
            self.owner_headers,
        )
        self.assertEqual(update_owner.status_code, 200)
        remove_owner = self._request(
            "DELETE",
            self._member(self.other_owner.id),
            headers=self.owner_headers,
        )
        self.assertEqual(remove_owner.status_code, 200)
        delete_tenant = self._request("DELETE", self._detail(), headers=self.owner_headers)
        self.assertEqual(delete_tenant.status_code, 200)

    def test_api_key_can_access_its_tenant_but_not_another(self):
        own_headers = {"HTTP_X_API_KEY": self.target.api_key}
        detail = self._request("GET", self._detail(), headers=own_headers)
        self.assertEqual(detail.status_code, 200)
        self.assertEqual(detail.json()["data"]["api_key"], "")
        updated = self._request(
            "PATCH",
            self._detail(),
            {"name": "updated by api key"},
            own_headers,
        )
        self.assertEqual(updated.status_code, 200)
        self.assertEqual(updated.json()["data"]["api_key"], "")
        self._assert_error(
            self._request("GET", self._detail(self.other.id), headers=own_headers),
            404,
            "not found",
        )
        self._assert_error(
            self._request(
                "POST",
                self._api_key(self.other.id),
                headers=own_headers,
            ),
            404,
            "not found",
        )
        self._assert_error(
            self._request("DELETE", self._detail(), headers=own_headers),
            403,
            "forbidden",
        )
        self._assert_error(
            self._request(
                "POST",
                self._members(),
                {"email": self.unaffiliated.email, "role": "owner"},
                own_headers,
            ),
            403,
            "forbidden",
        )
        self._assert_error(
            self._request(
                "PATCH",
                self._member(self.owner.id),
                {"role": "viewer"},
                own_headers,
            ),
            403,
            "forbidden",
        )
        rotated = self._request("POST", self._api_key(), headers=own_headers)
        self.assertEqual(rotated.status_code, 200)
        self.assertTrue(rotated.json()["data"]["api_key"])

    def test_viewer_login_and_me_do_not_expose_tenant_api_key(self):
        self.viewer.password_hash = hash_password("viewer-password")
        self.viewer.save(update_fields=["password_hash"])
        login = self._request(
            "POST",
            "/api/v1/auth/login",
            {"email": self.viewer.email, "password": "viewer-password"},
        )
        self.assertEqual(login.status_code, 200)
        self.assertEqual(login.json()["data"]["tenant"]["api_key"], "")

        me = self._request("GET", "/api/v1/auth/me", headers=self.viewer_headers)
        self.assertEqual(me.status_code, 200)
        self.assertEqual(me.json()["data"]["tenant"]["api_key"], "")

    def test_switch_tenant_requires_membership_and_redacts_for_viewer(self):
        other_tenant = self._request(
            "POST",
            "/api/v1/auth/switch-tenant",
            {"tenant_id": self.other.id},
            self.viewer_headers,
        )
        self._assert_error(other_tenant, 404, "tenant not found")

        member_tenant = self._request(
            "POST",
            "/api/v1/auth/switch-tenant",
            {"tenant_id": self.target.id},
            self.viewer_headers,
        )
        self.assertEqual(member_tenant.status_code, 200)
        self.assertEqual(member_tenant.json()["data"]["tenant"]["api_key"], "")

    def test_system_admin_can_access_and_manage_any_tenant(self):
        detail = self._request("GET", self._detail(), headers=self.system_admin_headers)
        self.assertEqual(detail.status_code, 200)
        self.assertEqual(detail.json()["data"]["api_key"], self.target.api_key)
        self.assertEqual(
            self._request("GET", self._members(), headers=self.system_admin_headers).status_code,
            200,
        )
        self.assertEqual(
            self._request(
                "POST",
                self._members(),
                {"email": self.unaffiliated.email, "role": "owner"},
                self.system_admin_headers,
            ).status_code,
            200,
        )
        self.assertEqual(
            self._request(
                "PATCH",
                self._detail(),
                {"description": "admin edit"},
                self.system_admin_headers,
            ).status_code,
            200,
        )
        self.assertEqual(
            self._request("POST", self._api_key(), headers=self.system_admin_headers).status_code,
            200,
        )
        self.assertEqual(
            self._request("DELETE", self._detail(), headers=self.system_admin_headers).status_code,
            200,
        )
        missing = self._request(
            "GET",
            self._detail(self.target.id + 10000),
            headers=self.system_admin_headers,
        )
        self.assertEqual(missing.status_code, 404)
        self.assertTrue(missing["Content-Type"].startswith("text/html"))

    def test_last_active_owner_cannot_demote_self(self):
        self._leave_single_active_owner()
        response = self._request(
            "PATCH",
            self._member(self.owner.id),
            {"role": "viewer"},
            self.owner_headers,
        )
        self._assert_error(response, 409, "tenant must keep at least one owner")
        self.assertEqual(
            TenantMember.objects.get(user=self.owner, tenant=self.target).role,
            "owner",
        )

    def test_last_active_owner_cannot_remove_self(self):
        self._leave_single_active_owner()
        response = self._request(
            "DELETE",
            self._member(self.owner.id),
            headers=self.owner_headers,
        )
        self._assert_error(response, 409, "tenant must keep at least one owner")
        self.assertTrue(
            TenantMember.objects.filter(user=self.owner, tenant=self.target).exists()
        )

    def test_last_active_owner_cannot_be_upserted_as_viewer(self):
        self._leave_single_active_owner()
        response = self._request(
            "POST",
            self._members(),
            {"email": self.owner.email, "role": "viewer"},
            self.owner_headers,
        )
        self._assert_error(response, 409, "tenant must keep at least one owner")
        self.assertEqual(
            TenantMember.objects.get(user=self.owner, tenant=self.target).role,
            "owner",
        )

    def test_owner_can_be_demoted_when_another_active_owner_remains(self):
        response = self._request(
            "PATCH",
            self._member(self.other_owner.id),
            {"role": "admin"},
            self.owner_headers,
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            TenantMember.objects.get(user=self.other_owner, tenant=self.target).role,
            "admin",
        )
