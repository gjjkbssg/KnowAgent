from django.core.exceptions import ObjectDoesNotExist


class ServiceError(Exception):
    status = 400
    code = "error"

    def __init__(
        self,
        message: str,
        *,
        status: int | None = None,
        code: str | None = None,
        details: dict | None = None,
    ):
        self.message = message
        self.status = self.__class__.status if status is None else status
        self.code = self.__class__.code if code is None else code
        self.details = details
        super().__init__(message)


class InvalidInput(ServiceError):
    status = 400


class Unauthorized(ServiceError):
    status = 401


class Forbidden(ServiceError):
    status = 403


class NotFound(ServiceError):
    status = 404


class Conflict(ServiceError):
    status = 409


def get_or_404(model_or_qs, **filters):
    try:
        manager_or_qs = getattr(model_or_qs, "_default_manager", model_or_qs)
        return manager_or_qs.get(**filters)
    except ObjectDoesNotExist:
        raise NotFound("") from None
