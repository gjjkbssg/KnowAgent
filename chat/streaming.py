import json
import time

from django.http import StreamingHttpResponse
from django.utils import timezone

from personal_knowledge_base.models import AgentToolApproval, Message
from personal_knowledge_base.serializers import message_dict
from personal_knowledge_base.stream_manager import StreamEvent, stream_manager


STREAM_REPLAY_IDLE_TIMEOUT = 120  # 重连后这么多秒没有新事件就放弃等待

def sse_frame(payload: dict, event: str = "message", event_id: int | None = None) -> str:
    prefix = f"id: {event_id}\n" if event_id is not None else ""
    return f"{prefix}event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"

TERMINAL_STREAM_EVENTS = ("complete", "error", "approval_required")

def _stream_event_frames(msg_id: str, event) -> list[tuple[str, dict]]:
    """一个 StreamManager 事件对应的 SSE 帧（事件名, payload）。只依赖事件本身，回放时结果不变。"""
    data = event.data
    if event.event_type == "thinking":
        payload = {"response_type": "answer", "content": data.get("content", ""), "done": False}
    elif event.event_type == "tool_call":
        payload = {"response_type": "tool_call", "name": data.get("name", ""), "arguments": data.get("arguments", {}), "iteration": data.get("iteration", 0)}
    elif event.event_type == "tool_result":
        payload = {"response_type": "tool_result", "name": data.get("name", ""), "output": data.get("output", "")[:300], "duration_ms": data.get("duration_ms", 0)}
    elif event.event_type.startswith("actor_"):
        payload = data
    elif event.event_type == "complete":
        stream = stream_manager.get_stream(msg_id)
        payload = {
            "response_type": "answer",
            "content": data.get("content") or (stream.final_content if stream else ""),
            "done": True,
            "knowledge_references": stream.final_refs if stream else [],
        }
    elif event.event_type == "error":
        payload = {"response_type": "error", "content": data.get("content", "生成失败"), "done": True}
    elif event.event_type == "approval_required":
        payload = {"response_type": "tool_approval_required", "approval": data.get("approval", {})}
    else:
        return []
    frames = [("message", {**payload, "assistant_message_id": msg_id})]
    if event.event_type == "complete":
        frames.append(("message", {"response_type": "complete", "assistant_message_id": msg_id, "done": True}))
    if event.event_type in TERMINAL_STREAM_EVENTS:
        frames.append(("done", {"message_id": msg_id}))
    return frames

def _saved_message_frames(message_id: str, message: Message | None) -> list[tuple[str, dict]]:
    """流已不在内存里（进程重启或 TTL 过期）时，用数据库里的最终结果收尾。"""
    if not message or not message.is_completed:
        return [("done", {"message_id": message_id})]
    return [
        ("message", message_dict(message)),
        ("message", {"response_type": "complete", "assistant_message_id": message.id, "done": True}),
        ("done", {"message_id": message.id}),
    ]

def _approval_frames(approval: AgentToolApproval) -> list[tuple[str, dict]]:
    return [
        ("message", {
            "response_type": "tool_approval_required",
            "approval": {
                "id": approval.id,
                "status": approval.status,
                "tool_calls": approval.tool_calls,
                "expires_at": approval.expires_at.isoformat(),
            },
            "assistant_message_id": approval.message_id,
        }),
        ("done", {"message_id": approval.message_id}),
    ]

def _numbered_frames(frames, first_id: int):
    for offset, (event, payload) in enumerate(frames):
        yield sse_frame(payload, event, event_id=first_id + offset)

def stream_terminal_event(message_id: str) -> StreamEvent | None:
    """内存流里第一个终止事件；流不存在或还没结束时返回 None。"""
    stream = stream_manager.get_stream(message_id)
    if not stream or stream.first_event_frame_id is None:
        return None
    for event in stream.get_events():
        if event.event_type in TERMINAL_STREAM_EVENTS:
            return event
    return None

def stream_message_events(
    message: Message, last_event_id: int = -1, idle_timeout: float | None = None, *, wait_for_resume: bool = False,
):
    """
    把 StreamManager 中一条消息的事件转成 SSE 帧，首次推送、断线重连和已完成消息的回放共用。

    帧 id 是这条消息的第几帧（从 0 开始，包括开头的 message_start），每帧唯一且递增；
    同一个事件流每次回放得到的帧和 id 都一样。客户端重连时把最后收到的 id 放进
    Last-Event-ID，这里只发 id 更大的帧，不重复也不遗漏。
    """
    msg_id = message.id
    stream = stream_manager.get_stream(msg_id)
    next_id = 0

    def emit(frames):
        nonlocal next_id
        for event, payload in frames:
            if next_id > last_event_id:
                yield sse_frame(payload, event, event_id=next_id)
            next_id += 1

    yield from emit([
        ("message_start", {"id": msg_id, "request_id": message.request_id}),
        ("message", {"response_type": "agent_query", "assistant_message_id": msg_id, "session_id": message.session_id, "content": "", "done": False}),
    ])

    offset = 0
    idle = 0.0
    next_db_check = 0.0
    while idle_timeout is None or idle < idle_timeout:
        current_stream = stream_manager.get_stream(msg_id)
        if wait_for_resume and current_stream is not stream:
            offset = 0
            next_id = max(next_id, last_event_id + 1)
        stream = current_stream
        if stream and stream.first_event_frame_id is None:
            # After a restart the earlier frame positions are unknown. Use saved state.
            wait_for_resume = True
            events = []
        elif stream:
            next_id = max(next_id, stream.first_event_frame_id)
            events = stream.get_events(offset)
        else:
            events = []
        for event in events:
            frames = _stream_event_frames(msg_id, event)
            if wait_for_resume and event.event_type == "approval_required" and not AgentToolApproval.objects.filter(
                id=event.data.get("approval", {}).get("id"), status="pending", expires_at__gt=timezone.now(),
            ).exists():
                # Preserve the positions of approval frames sent before the decision.
                next_id += len(frames)
                continue
            yield from emit(frames)
            if event.event_type in TERMINAL_STREAM_EVENTS:
                return
        offset += len(events)

        if not events and wait_for_resume and time.monotonic() >= next_db_check:
            # A worker in another process may only be visible through saved state.
            next_db_check = time.monotonic() + 1
            saved = Message.objects.filter(id=msg_id).first()
            first_id = max(next_id, last_event_id + 1)
            if not saved or saved.is_completed:
                yield from _numbered_frames(_saved_message_frames(msg_id, saved), first_id)
                return
            pending = saved.tool_approvals.filter(status="pending", expires_at__gt=timezone.now()).first()
            if pending:
                yield from _numbered_frames(_approval_frames(pending), first_id)
                return
        elif not events and not wait_for_resume and stream_manager.is_complete(msg_id):
            saved = Message.objects.filter(id=msg_id).first()
            yield from _numbered_frames(_saved_message_frames(msg_id, saved), max(next_id, last_event_id + 1))
            return

        idle = 0.0 if events else idle + 0.1
        time.sleep(0.1)

    # 超时帧只属于这一次连接，不在消息的事件序列里：不带 id，客户端的 Last-Event-ID
    # 仍停在最后一个真实帧上，之后续传不会跳过还没生成的帧。
    yield sse_frame({"response_type": "error", "assistant_message_id": msg_id, "content": "等待超时", "done": True})
    yield sse_frame({"message_id": msg_id}, "done")

def stream_message(message: Message):
    def events():
        payload = message_dict(message)
        start = {"id": message.id, "request_id": message.request_id, "assistant_message_id": message.id, "session_id": message.session_id, "response_type": "agent_query", "content": "", "done": False}
        yield f"event: message_start\ndata: {json.dumps({'id': message.id, 'request_id': message.request_id}, ensure_ascii=False)}\n\n"
        yield f"event: message\ndata: {json.dumps(start, ensure_ascii=False)}\n\n"
        text = message.content or ""
        step = max(1, len(text) // 12)
        sent = ""
        for index in range(0, len(text), step):
            sent += text[index : index + step]
            chunk = {**payload, "content": sent, "is_completed": False}
            yield f"event: message\ndata: {json.dumps(chunk, ensure_ascii=False)}\n\n"
            compat = {"id": message.request_id, "assistant_message_id": message.id, "session_id": message.session_id, "response_type": "answer", "content": sent, "done": False, "knowledge_references": message.knowledge_references}
            yield f"event: message\ndata: {json.dumps(compat, ensure_ascii=False)}\n\n"
        refs = {"id": message.id, "knowledge_references": message.knowledge_references}
        yield f"event: references\ndata: {json.dumps(refs, ensure_ascii=False)}\n\n"
        yield f"event: message\ndata: {json.dumps({**refs, 'response_type': 'references', 'assistant_message_id': message.id, 'session_id': message.session_id}, ensure_ascii=False)}\n\n"
        yield f"event: message\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"
        yield f"event: message\ndata: {json.dumps({'id': message.request_id, 'assistant_message_id': message.id, 'session_id': message.session_id, 'response_type': 'answer', 'content': text, 'done': True, 'knowledge_references': message.knowledge_references}, ensure_ascii=False)}\n\n"
        yield f"event: message\ndata: {json.dumps({'id': message.request_id, 'assistant_message_id': message.id, 'session_id': message.session_id, 'response_type': 'complete', 'done': True}, ensure_ascii=False)}\n\n"
        yield f"event: done\ndata: {json.dumps({'message_id': message.id}, ensure_ascii=False)}\n\n"

    return StreamingHttpResponse(events(), content_type="text/event-stream")
