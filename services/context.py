from dataclasses import dataclass

from personal_knowledge_base.models import Tenant, User


@dataclass(frozen=True)
class ServiceContext:
    tenant: Tenant
    user: User | None
    request_id: str = ""
