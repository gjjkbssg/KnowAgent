import sys
import unittest

from contract_tests.assertions import assert_snapshot, capture
from contract_tests.harness import new_client, set_admin


class AuthContractTests(unittest.TestCase):
    def test_initial_setup_login_rotation_preferences_and_password_change(self):
        self.assertFalse(
            any(module == "django" or module.startswith("django.") for module in sys.modules),
            "The black-box test process must not import Django",
        )
        responses = {}
        with new_client() as client:
            first = client.post("/api/v1/auth/auto-setup")
            responses["auto_setup_first"] = capture(self, "auto_setup_first", first, 201)
            initial = first.json()["data"]
            self.assertTrue(initial["require_password_change"])
            self.assertTrue(initial["temp_password"])

            second = client.post("/api/v1/auth/auto-setup")
            responses["auto_setup_second"] = capture(self, "auto_setup_second", second, 403)

            login = client.post(
                "/api/v1/auth/login",
                json={"username": "admin", "password": initial["temp_password"]},
            )
            responses["login"] = capture(self, "login", login, 200)
            login_data = login.json()["data"]
            set_admin(
                login_data["token"],
                login_data["refresh_token"],
                initial["temp_password"],
            )

            refresh = client.post(
                "/api/v1/auth/refresh",
                json={"refresh_token": login_data["refresh_token"]},
            )
            responses["refresh"] = capture(self, "refresh", refresh, 200)
            refreshed = refresh.json()["data"]
            set_admin(
                refreshed["token"],
                refreshed["refresh_token"],
                initial["temp_password"],
            )

            old_refresh = client.post(
                "/api/v1/auth/refresh",
                json={"refresh_token": login_data["refresh_token"]},
            )
            responses["refresh_old_token"] = capture(
                self, "refresh_old_token", old_refresh, 401
            )

            client.headers["Authorization"] = f"Bearer {refreshed['token']}"
            me = client.get("/api/v1/auth/me")
            responses["me"] = capture(self, "me", me, 200)

            preferences = client.put(
                "/api/v1/auth/me/preferences",
                json={"theme": "dark", "contract_fixture": True},
            )
            responses["preferences"] = capture(self, "preferences", preferences, 200)

            changed = client.post(
                "/api/v1/auth/change-password",
                json={
                    "old_password": initial["temp_password"],
                    "new_password": "Contract-password-2026!",
                },
            )
            responses["change_password"] = capture(
                self, "change_password", changed, 200
            )

            old_login = client.post(
                "/api/v1/auth/login",
                json={"username": "admin", "password": initial["temp_password"]},
            )
            responses["login_old_password"] = capture(
                self, "login_old_password", old_login, 401
            )

            new_login = client.post(
                "/api/v1/auth/login",
                json={"username": "admin", "password": "Contract-password-2026!"},
            )
            responses["login_new_password"] = capture(
                self, "login_new_password", new_login, 200
            )
            new_data = new_login.json()["data"]
            set_admin(
                new_data["token"],
                new_data["refresh_token"],
                "Contract-password-2026!",
            )

        assert_snapshot(self, "auth", responses)
