import client from './client'
import { streamWithResume } from './sse.mjs'

function authHeaders(extra: Record<string, string> = {}) {
  const token = localStorage.getItem('personal_kb_token')
  const tenant = localStorage.getItem('personal_kb_selected_tenant_id')
  return {
    'Content-Type': 'application/json',
    Accept: 'text/event-stream',
    ...(token ? { Authorization: `Bearer ${token}` } : {}),
    ...(tenant ? { 'X-Tenant-ID': tenant } : {}),
    'X-Request-ID': Math.random().toString(36).slice(2),
    ...extra,
  }
}

async function openStream(url: string, init: RequestInit) {
  const response = await fetch(url, init)
  if (!response.ok || !response.body) {
    throw new Error(`stream request failed: ${response.status}`)
  }
  return response
}

type StreamHandler = (event: string, payload: any) => void

// 续传：带上最后收到的事件 id，服务端只补发它之后的事件
function resumeStream(sessionId: string, signal?: AbortSignal) {
  return (state: { messageId: string; lastEventId: string }) =>
    openStream(`/api/v1/sessions/continue-stream/${sessionId}?message_id=${encodeURIComponent(state.messageId)}`, {
      method: 'GET',
      headers: authHeaders(state.lastEventId ? { 'Last-Event-ID': state.lastEventId } : {}),
      signal,
    })
}

export async function streamChat(sessionId: string, data: any, agent = false, onEvent: StreamHandler, signal?: AbortSignal) {
  const url = `${agent ? '/api/v1/agent-chat' : '/api/v1/knowledge-chat'}/${sessionId}`
  const open = () =>
    openStream(url, {
      method: 'POST',
      headers: authHeaders(),
      body: JSON.stringify({ ...data, stream: true, channel: 'web' }),
      signal,
    })
  await streamWithResume(open, resumeStream(sessionId, signal), onEvent)
}

/**
 * 页面刷新或重新打开有未完成消息的会话时，从头回放这条消息的事件并继续接收。
 */
export async function continueStream(sessionId: string, messageId: string, onEvent: StreamHandler, signal?: AbortSignal) {
  const resume = resumeStream(sessionId, signal)
  await streamWithResume(() => resume({ messageId, lastEventId: '' }), resume, onEvent)
}

export const api = {
  autoSetup: () => client.post('/api/v1/auth/auto-setup'),
  login: (data: any) => client.post('/api/v1/auth/login', data),
  me: () => client.get('/api/v1/auth/me'),
  updatePreferences: (data: any) => client.put('/api/v1/auth/me/preferences', data),
  changePassword: (data: any) => client.post('/api/v1/auth/change-password', data),
  listKbs: () => client.get('/api/v1/knowledge-bases'),
  searchKbs: (params: any = {}) => client.get('/api/v1/knowledge-bases', { params }),
  createKb: (data: any) => client.post('/api/v1/knowledge-bases', data),
  getKb: (id: string) => client.get(`/api/v1/knowledge-bases/${id}`),
  updateKb: (id: string, data: any) => client.put(`/api/v1/knowledge-bases/${id}`, data),
  deleteKb: (id: string) => client.delete(`/api/v1/knowledge-bases/${id}`),
  pinKb: (id: string, isPinned?: boolean) => client.put(`/api/v1/knowledge-bases/${id}/pin`, { is_pinned: isPinned }),
  copyKb: (sourceId: string) => client.post('/api/v1/knowledge-bases/copy', { source_id: sourceId }),
  moveTargets: (kbId: string) => client.get(`/api/v1/knowledge-bases/${kbId}/move-targets`),
  listKnowledge: (kbId: string, params: any = {}) => client.get(`/api/v1/knowledge-bases/${kbId}/knowledge`, { params }),
  getKnowledgeSpans: (knowledgeId: string) => client.get(`/api/v1/knowledge/${knowledgeId}/stages`),
  uploadFile: (kbId: string, file: File, options: { tag_id?: string; process_config?: any } = {}) => {
    const fd = new FormData()
    fd.append('file', file)
    if (options.tag_id) fd.append('tag_id', options.tag_id)
    if (options.process_config) fd.append('process_config', JSON.stringify(options.process_config))
    return client.post(`/api/v1/knowledge-bases/${kbId}/knowledge/file`, fd, { headers: { 'Content-Type': 'multipart/form-data' } })
  },
  deleteKnowledge: (id: string) => client.delete(`/api/v1/knowledge/${id}`),
  reparseKnowledge: (id: string, data: any = {}) => client.post(`/api/v1/knowledge/${id}/reparse`, data),
  cancelKnowledge: (id: string) => client.post(`/api/v1/knowledge/${id}/cancel-parse`),
  batchDeleteKnowledge: (ids: string[], kbId = '') => client.post('/api/v1/knowledge/batch-delete', { ids, kb_id: kbId }),
  moveKnowledge: (ids: string[], targetKbId: string, sourceKbId = '') => client.post('/api/v1/knowledge/move', { ids, source_kb_id: sourceKbId, target_knowledge_base_id: targetKbId }),
  listChunks: (knowledgeId: string, params: any = {}) => client.get(`/api/v1/chunks/${knowledgeId}`, { params }),
  knowledgeImage: (knowledgeId: string, imageId: string) => client.get(`/api/v1/knowledge/${knowledgeId}/images/${imageId}`, { responseType: 'blob' }),
  updateChunk: (knowledgeId: string, chunkId: string, data: any) => client.put(`/api/v1/chunks/${knowledgeId}/${chunkId}`, data),
  deleteChunk: (knowledgeId: string, chunkId: string) => client.delete(`/api/v1/chunks/${knowledgeId}/${chunkId}`),
  listTags: (kbId: string) => client.get(`/api/v1/knowledge-bases/${kbId}/tags`),
  createTag: (kbId: string, data: any) => client.post(`/api/v1/knowledge-bases/${kbId}/tags`, data),
  createSession: (data: any) => client.post('/api/v1/sessions', data),
  listSessions: (params: any = {}) => client.get('/api/v1/sessions', { params }),
  getSession: (sessionId: string) => client.get(`/api/v1/sessions/${sessionId}`),
  deleteSession: (sessionId: string) => client.delete(`/api/v1/sessions/${sessionId}`),
  deleteSessions: (ids: string[]) => client.delete('/api/v1/sessions/batch', { data: { ids } }),
  deleteAllSessions: () => client.delete('/api/v1/sessions/batch', { data: { delete_all: true } }),
  pinSession: (sessionId: string) => client.post(`/api/v1/sessions/${sessionId}/pin`),
  unpinSession: (sessionId: string) => client.delete(`/api/v1/sessions/${sessionId}/pin`),
  clearSessionMessages: (sessionId: string) => client.delete(`/api/v1/sessions/${sessionId}/messages`),
  stopSession: (sessionId: string, messageId = '') => client.post(`/api/v1/sessions/${sessionId}/stop`, { message_id: messageId }),
  loadMessages: (sessionId: string, params: any = {}) => client.get(`/api/v1/messages/${sessionId}/load`, { params: { limit: 20, ...params } }),
  chat: (sessionId: string, data: any) => client.post(`/api/v1/knowledge-chat/${sessionId}`, data),
  agentChat: (sessionId: string, data: any) => client.post(`/api/v1/agent-chat/${sessionId}`, data),
  listModels: () => client.get('/api/v1/models'),
  modelUsage: (params: any = {}) => client.get('/api/v1/models/usage', { params }),
  createModel: (data: any) => client.post('/api/v1/models', data),
  updateModel: (id: string, data: any) => client.put(`/api/v1/models/${id}`, data),
  deleteModel: (id: string) => client.delete(`/api/v1/models/${id}`),
  updateModelCredentials: (id: string, data: any) => client.put(`/api/v1/models/${id}/credentials`, data),
  deleteModelCredential: (id: string, field: string) => client.delete(`/api/v1/models/${id}/credentials/${field}`),
  systemInfo: () => client.get('/api/v1/system/info'),
  parserEngines: () => client.get('/api/v1/system/parser-engines'),
  storageStatus: () => client.get('/api/v1/system/storage-engine-status'),
  vectorStoreTypes: () => client.get('/api/v1/vector-stores/types'),
  webSearchProviderTypes: () => client.get('/api/v1/web-search-providers/types'),
  checkParserEngine: (data: any = {}) => client.post('/api/v1/system/parser-engines/check', data),
  checkStorageEngine: (data: any = {}) => client.post('/api/v1/system/storage-engine-check', data),
  getTenantKv: (key: string) => client.get(`/api/v1/tenants/kv/${key}`),
  updateTenantKv: (key: string, value: any) => client.put(`/api/v1/tenants/kv/${key}`, { value }),
  listMcpServices: () => client.get('/api/v1/mcp-services'),
  createMcpService: (data: any) => client.post('/api/v1/mcp-services', data),
  updateMcpService: (id: string, data: any) => client.put(`/api/v1/mcp-services/${id}`, data),
  deleteMcpService: (id: string) => client.delete(`/api/v1/mcp-services/${id}`),
  wikiPages: (kbId: string) => client.get(`/api/v1/knowledge-bases/${kbId}/wiki/pages`),
  getWikiPage: (kbId: string, slug: string) => client.get(`/api/v1/knowledge-bases/${kbId}/wiki/pages/${slug.split('/').map(encodeURIComponent).join('/')}`),
  createWikiPage: (kbId: string, data: any) => client.post(`/api/v1/knowledge-bases/${kbId}/wiki/pages`, data),
  wikiSearch: (kbId: string, params: any = {}) => client.get(`/api/v1/knowledge-bases/${kbId}/wiki/search`, { params }),
  wikiGraph: (kbId: string, params: any = {}) => client.get(`/api/v1/knowledge-bases/${kbId}/wiki/graph`, { params }),

  // RAG 评估
  ragEvalRun: (data: any = {}) => client.post('/api/v1/rag-eval/run', data),
  ragEvalQuestions: () => client.get('/api/v1/rag-eval/questions'),
  ragEvalAddQuestion: (data: any) => client.post('/api/v1/rag-eval/questions', data),
  ragEvalGenerate: (data: any = {}) => client.post('/api/v1/rag-eval/generate', data),
  ragEvalHistory: () => client.get('/api/v1/rag-eval/history'),
}
