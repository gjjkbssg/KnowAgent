import json

from .authentication import require_auth


def parse_body(request):
    if request.content_type and request.content_type.startswith("multipart/"):
        return request.POST.dict()
    if not request.body:
        return {}
    try:
        return json.loads(request.body.decode("utf-8"))
    except Exception:
        return {}


def auth_context(request):
    try:
        return require_auth(request)
    except PermissionError:
        return None, None


def bounded_int(value, default, minimum=None, maximum=None):
    try:
        number = int(value if value not in (None, "") else default)
    except (TypeError, ValueError):
        number = default
    if minimum is not None:
        number = max(number, minimum)
    if maximum is not None:
        number = min(number, maximum)
    return number


def paginate(qs, request):
    page_size = bounded_int(request.GET.get("page_size", request.GET.get("limit", 20)), 20, 1, 200)
    if "offset" in request.GET and "page" not in request.GET:
        offset = bounded_int(request.GET.get("offset"), 0, 0)
        page = offset // page_size + 1
    else:
        page = bounded_int(request.GET.get("page"), 1, 1)
        offset = (page - 1) * page_size
    total = qs.count()
    return qs[offset : offset + page_size], {"page": page, "page_size": page_size, "total": total}


def list_response(items, meta=None, aliases=None):
    payload = {"items": items, "data": items}
    for alias in aliases or []:
        payload[alias] = items
    if meta:
        payload.update(meta)
    return payload


def csv_values(value):
    if not value:
        return []
    if isinstance(value, (list, tuple)):
        values = value
    else:
        values = str(value).split(",")
    return [str(item).strip() for item in values if str(item).strip()]


def bool_from_value(value, default=False):
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"1", "true", "yes", "on", "enabled"}
