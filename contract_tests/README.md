# Black-box Django API contracts

These tests exercise the running HTTP API only. The `unittest` process imports
stdlib modules, `httpx`, and `contract_tests`; Django and application modules are
used only by the separate migration and development-server subprocesses.

Run from the repository root with:

```sh
/home/ubuntu/work/venv/bin/python -m unittest discover -s contract_tests -t .
```

By default, the shared harness starts a threaded fake OpenAI-compatible service,
migrates a fresh SQLite database under a temporary directory, starts Django, and
stops both services at process exit. `APP_DB_PATH` and `APP_MEDIA_ROOT` keep
contract runs away from the repository database and media directory. The
server's inherited LLM/API-key variables are removed before startup. Set
`CONTRACT_SERVER_CMD` to override the command template; `{python}` and `{port}`
are substituted. Set `CONTRACT_BASE_URL` to use an already-running API instead
of starting/migrating a local server. Use the dedicated unittest command above;
the package avoids having Django's in-process test runner import these
subprocess-isolated tests. With `CONTRACT_BASE_URL`, provide a clean test API
instance configured with the deterministic fake LLM; the harness cannot inspect
model requests made by that external server.

Snapshots in `snapshots/` describe response JSON structure, status, and SSE
frame structure. `sse_response_types.json` additionally preserves the ordered
distinct `response_type` values. To intentionally regenerate snapshots after an
API contract change, run with `CONTRACT_UPDATE=1`.

## Route coverage

All paths below are under `/api/v1`. All frontend API calls in
`frontend/src/api/index.ts` are covered except the two external-model evaluation
routes listed below.

| Surface | Covered methods and routes |
| --- | --- |
| Authentication | `POST /auth/auto-setup`, `/auth/login`, `/auth/refresh`, `/auth/change-password`; `GET /auth/me`; `PUT /auth/me/preferences` |
| Knowledge bases | `GET` and `POST /knowledge-bases`; `GET`, `PUT`, and `DELETE /knowledge-bases/{id}`; `PUT /knowledge-bases/{id}/pin`; `POST /knowledge-bases/copy`; `GET /knowledge-bases/{id}/move-targets` |
| Knowledge and chunks | `GET /knowledge-bases/{id}/knowledge`; `POST /knowledge-bases/{id}/knowledge/file`; `GET /knowledge/{id}/stages`; `PUT` and `DELETE /knowledge/{id}`; `POST /knowledge/{id}/reparse` and `/cancel-parse`; `POST /knowledge/move` and `/knowledge/batch-delete`; `GET /chunks/{knowledge_id}`; `PUT` and `DELETE /chunks/{knowledge_id}/{chunk_id}`; missing-image response at `GET /knowledge/{id}/images/{image_id}` |
| Tags and wiki | `GET` and `POST /knowledge-bases/{id}/tags`; `GET` and `POST /knowledge-bases/{id}/wiki/pages`; `GET /knowledge-bases/{id}/wiki/pages/{slug}`; `GET /knowledge-bases/{id}/wiki/search` and `/wiki/graph` |
| Sessions, messages, chat | `POST /sessions`; `GET /sessions` and `/sessions/{id}`; `DELETE /sessions/{id}`, `/sessions/batch` (selected and all), and `/sessions/{id}/messages`; `POST` and `DELETE /sessions/{id}/pin`; `POST /sessions/{id}/stop`; `GET /messages/{session_id}/load`; streamed `POST /knowledge-chat/{session_id}` and `/agent-chat/{session_id}`; `GET /sessions/continue-stream/{session_id}` with `Last-Event-ID` |
| Models and configuration | `GET /models` and `/models/usage`; `POST /models`; `PUT` and `DELETE /models/{id}`; `PUT /models/{id}/credentials`; `DELETE /models/{id}/credentials/{field}`; `GET /system/info`, `/system/parser-engines`, `/system/storage-engine-status`, `/vector-stores/types`, and `/web-search-providers/types`; `POST /system/parser-engines/check` and `/system/storage-engine-check`; `GET` and `PUT /tenants/kv/{key}` |
| MCP and RAG eval | `GET`, `POST`, `PUT`, and `DELETE /mcp-services[/{id}]`; `GET` and `POST /rag-eval/questions`; `GET /rag-eval/history` |
| Embed and account invitations | Agent/embed-channel fixture creation; `POST /embed/{channel}/exchange`; `GET /embed/{channel}/config` and `/suggested-questions`; `POST /embed/{channel}/sessions`; streamed `/embed/{channel}/knowledge-chat/{session}` and `/agent-chat/{session}`; `GET /embed/{channel}/messages/{session}/load` and `/sessions/{session}/events`; `POST /embed/{channel}/sessions/{session}/stop`; tenant invitations list/create, invite links, my-invitations list, accept/decline, and non-owner tenant leave |
| Auth boundaries | Unauthenticated 401 contracts for accounts, knowledge, chat, wiki, agent, models-config, and personal-knowledge-base routes; cross-tenant KB detail lookup |

## Exclusions

| Route | Reason |
| --- | --- |
| `POST /rag-eval/run` | Runs a full RAG evaluation, which requires model-backed evaluation and retrieval rather than only exercising the API contract. |
| `POST /rag-eval/generate` | Generates questions from indexed knowledge using the model pipeline; this is an integration/evaluation operation rather than a deterministic contract fixture. |

Parser-engine and storage-engine check routes are included: the current
implementations return local capability/status results and do not need a real
external parser or storage service for these fixtures.

## Contract findings

The current Django API uses `{"success": true, "message": ..., "data": ...}`
for success and `{"success": false, "error": {"code", "message", "details"},
"message": ...}` for errors; it does not use the proposed top-level `code` /
`message` envelope. The snapshots lock this observed behavior.

Knowledge-chat and agent-chat return ordered `response_type` values
`agent_query`, `answer`, `complete`, and include the fake LLM answer. The
knowledge-chat stream currently emits event IDs `[0, 1, 2, 3, 3, 3]` and
agent-chat emits `[0, 1, 1, 1]` (the last three frames in each share an ID).
Therefore IDs are non-decreasing but not strictly increasing. Completed-stream
continuation omits `id:` fields, so the `Last-Event-ID` greater-than check has no
IDs to evaluate on completed replay. The snapshots preserve this behavior
rather than claiming that ID-based replay was verified.
