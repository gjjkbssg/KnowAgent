from functools import wraps

from django.http import Http404, JsonResponse

from services.errors import NotFound, ServiceError


def ok(data=None, message: str = "success", status: int = 200, **extra):
    payload = {"success": True, "message": message}
    if data is not None:
        payload["data"] = data
    payload.update(extra)
    return JsonResponse(payload, status=status, json_dumps_params={"ensure_ascii": False})


def fail(message: str, status: int = 400, code: str = "error", details=None):
    return JsonResponse(
        {"success": False, "error": {"code": code, "message": message, "details": details or {}}, "message": message},
        status=status,
        json_dumps_params={"ensure_ascii": False},
    )


def service_errors(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        try:
            return view(*args, **kwargs)
        except NotFound as exc:
            if not exc.message:
                raise Http404 from exc
            return fail(exc.message, exc.status, exc.code, exc.details)
        except ServiceError as exc:
            return fail(exc.message, exc.status, exc.code, exc.details)

    return wrapped
