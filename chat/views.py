import uuid

from django.db.models import Q
from django.http import StreamingHttpResponse
from django.shortcuts import get_object_or_404
from django.utils import timezone
from django.views.decorators.csrf import csrf_exempt

from personal_knowledge_base.http_helpers import auth_context, parse_body
from personal_knowledge_base.models import AgentToolApproval, Message, Session, TaskRecord
from personal_knowledge_base.responses import fail, ok, service_errors
from personal_knowledge_base.tasks import RESUME_APPROVAL_TASK
from services.chat_generation import ChatTurn, decide_tool_approval, start_chat_turn
from services.chat_sessions import (
    batch_delete_sessions,
    chat_history_stats as chat_history_stats_service,
    clear_session_messages,
    create_session,
    delete_message as delete_message_service,
    delete_session,
    generate_session_title,
    get_session,
    list_sessions,
    load_messages,
    pin_session,
    search_messages,
    stop_session,
    update_session,
)
from services.context import ServiceContext

from .streaming import (
    STREAM_REPLAY_IDLE_TIMEOUT,
    _approval_frames,
    _numbered_frames,
    _saved_message_frames,
    stream_message_events,
    stream_terminal_event,
)


def _context(request, user, tenant, request_id=None):
    return ServiceContext(
        tenant=tenant,
        user=user,
        request_id=request_id or request.headers.get("X-Request-ID", ""),
    )


@csrf_exempt
@service_errors
def sessions_collection(request, session_id=None):
    user, tenant = auth_context(request)
    if not tenant:
        return fail("unauthorized", 401)
    ctx = _context(request, user, tenant)
    if session_id:
        if request.method == "GET":
            return ok(get_session(ctx, session_id))
        if request.method == "DELETE":
            return ok(delete_session(ctx, session_id))
        return ok(update_session(ctx, session_id, parse_body(request)))
    if request.method == "GET":
        return ok(list_sessions(ctx, request.GET))
    if request.method == "DELETE":
        return ok(batch_delete_sessions(ctx, parse_body(request)))
    return ok(create_session(ctx, parse_body(request)), status=201)


@csrf_exempt
@service_errors
def session_messages_clear(request, session_id):
    user, tenant = auth_context(request)
    if not tenant:
        return fail("unauthorized", 401)
    return ok(clear_session_messages(_context(request, user, tenant), session_id))


@csrf_exempt
@service_errors
def session_pin(request, session_id):
    user, tenant = auth_context(request)
    if not tenant:
        return fail("unauthorized", 401)
    return ok(pin_session(_context(request, user, tenant), session_id, pinned=request.method == "POST"))


@csrf_exempt
@service_errors
def session_title(request, session_id):
    user, tenant = auth_context(request)
    if not tenant:
        return fail("unauthorized", 401)
    return ok(generate_session_title(_context(request, user, tenant), session_id, parse_body(request)))


@csrf_exempt
@service_errors
def session_stop(request, session_id):
    user, tenant = auth_context(request)
    if not tenant:
        return fail("unauthorized", 401)
    return ok(stop_session(_context(request, user, tenant), session_id, parse_body(request)))


@csrf_exempt
def continue_stream(request, session_id):
    user, tenant = auth_context(request)
    if not tenant:
        return fail("unauthorized", 401)

    session = get_object_or_404(Session, id=session_id, tenant=tenant)

    message_id = request.GET.get("message_id") or request.GET.get("query")
    if not message_id:
        data = parse_body(request) if request.method == "POST" else {}
        message_id = data.get("message_id") or data.get("query")

    if not message_id:
        return fail("message_id is required", 400)

    message = Message.objects.filter(
        Q(id=message_id) | Q(request_id=message_id),
        session_id=session_id,
    ).first()

    if not message:
        return fail("message not found", 404)

    now = timezone.now()
    expired_count = AgentToolApproval.objects.filter(
        message=message,
        status="pending",
        expires_at__lte=now,
    ).update(status="expired", decided_at=now, updated_at=now)
    if expired_count:
        Message.objects.filter(id=message.id).update(
            content="等待确认超时，未执行该操作。",
            rendered_content="等待确认超时，未执行该操作。",
            is_completed=True,
            updated_at=timezone.now(),
        )
        message.refresh_from_db()

    try:
        last_event_id = max(-1, int(request.headers.get("Last-Event-ID", "")))
    except ValueError:
        last_event_id = -1

    approval = AgentToolApproval.objects.filter(
        message=message,
        status="pending",
        expires_at__gt=timezone.now(),
    ).first()
    terminal_event = stream_terminal_event(message.id)
    replayable = False
    if terminal_event:
        if approval:
            replayable = (
                terminal_event.event_type == "approval_required"
                and terminal_event.data.get("approval", {}).get("id") == approval.id
            )
        else:
            replayable = message.is_completed and terminal_event.event_type in ("complete", "error")
    if replayable:
        events = stream_message_events(message, last_event_id=last_event_id)
        return StreamingHttpResponse(events, content_type="text/event-stream")

    if approval:
        return StreamingHttpResponse(
            _numbered_frames(_approval_frames(approval), last_event_id + 1),
            content_type="text/event-stream",
        )

    if message.is_completed:
        return StreamingHttpResponse(
            _numbered_frames(_saved_message_frames(message.id, message), last_event_id + 1),
            content_type="text/event-stream",
        )

    wait_for_resume = TaskRecord.objects.filter(
        task_type=RESUME_APPROVAL_TASK,
        id__in=message.tool_approvals.values("id"),
    ).exists()
    events = stream_message_events(
        message,
        last_event_id=last_event_id,
        idle_timeout=STREAM_REPLAY_IDLE_TIMEOUT,
        wait_for_resume=wait_for_resume,
    )
    return StreamingHttpResponse(events, content_type="text/event-stream")


@service_errors
def messages_load(request, session_id):
    user, tenant = auth_context(request)
    if not tenant:
        return fail("unauthorized", 401)
    return ok(load_messages(_context(request, user, tenant), session_id, request.GET))


@csrf_exempt
@service_errors
def messages_search(request):
    user, tenant = auth_context(request)
    if not tenant:
        return fail("unauthorized", 401)
    return ok(search_messages(_context(request, user, tenant), parse_body(request)))


@service_errors
def chat_history_stats(request):
    user, tenant = auth_context(request)
    if not tenant:
        return fail("unauthorized", 401)
    return ok(chat_history_stats_service(_context(request, user, tenant)))


@csrf_exempt
@service_errors
def message_delete(request, session_id, message_id):
    user, tenant = auth_context(request)
    if not tenant:
        return fail("unauthorized", 401)
    return ok(delete_message_service(_context(request, user, tenant), session_id, message_id))


@csrf_exempt
@service_errors
def tool_approval_decision(request, session_id, approval_id):
    if request.method != "POST":
        return fail("method not allowed", 405)
    user, tenant = auth_context(request)
    if not tenant:
        return fail("unauthorized", 401)
    ctx = _context(request, user, tenant)
    return ok(decide_tool_approval(ctx, session_id, approval_id, parse_body(request)))


@csrf_exempt
@service_errors
def chat_endpoint(request, session_id, agent=False):
    user, tenant = auth_context(request)
    if not tenant:
        return fail("unauthorized", 401)
    data = parse_body(request)
    request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
    ctx = _context(request, user, tenant, request_id=request_id)
    streaming = bool(
        request.headers.get("Accept", "").find("text/event-stream") >= 0
        or data.get("stream")
    )
    turn: ChatTurn = start_chat_turn(
        ctx,
        session_id,
        data,
        agent=agent,
        streaming=streaming,
    )
    if streaming:
        return StreamingHttpResponse(stream_message_events(turn.message), content_type="text/event-stream")
    return ok(turn.payload)
