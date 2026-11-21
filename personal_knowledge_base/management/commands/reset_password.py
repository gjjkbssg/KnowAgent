import secrets

from django.core.management.base import BaseCommand, CommandError
from django.db.models import Q

from personal_knowledge_base.authentication import hash_password
from personal_knowledge_base.models import AuthToken, User


class Command(BaseCommand):
    help = "Set a new temporary password for a user and revoke their tokens (recovery when the password is lost)."

    def add_arguments(self, parser):
        parser.add_argument("login", nargs="?", default="admin@knowledge.local", help="Email or username")

    def handle(self, *args, **options):
        login = options["login"]
        user = User.objects.filter(Q(email=login) | Q(username=login), deleted_at__isnull=True).first()
        if not user:
            raise CommandError(f"No user found for {login}")
        password = secrets.token_urlsafe(12)
        user.password_hash = hash_password(password)
        user.save(update_fields=["password_hash", "updated_at"])
        AuthToken.objects.filter(user=user, is_revoked=False).update(is_revoked=True)
        self.stdout.write(f"Temporary password for {user.email}: {password}")
