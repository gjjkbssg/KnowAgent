// SSE 帧解析与断线续传。不依赖浏览器 API（fetch 由调用方传入），可以直接用 node --test 测。

export function newStreamState() {
  return { messageId: '', lastEventId: '', done: false }
}

export function parseSseFrame(frame, state) {
  let event = 'message'
  let data = ''
  for (const line of frame.split('\n')) {
    if (line.startsWith('event:')) event = line.slice(6).trim()
    else if (line.startsWith('id:')) state.lastEventId = line.slice(3).trim()
    // SSE 规范：多行 data 字段用换行符连接
    else if (line.startsWith('data:')) data += (data ? '\n' : '') + line.slice(5).trim()
  }
  if (!data) return null
  let payload = data
  try {
    payload = JSON.parse(data)
  } catch {
    // 非 JSON 的 data 原样交给调用方
  }
  if (event === 'message_start' && payload?.id) state.messageId = payload.id
  if (event === 'done') state.done = true
  return { event, payload }
}

export async function readSse(response, onEvent, state) {
  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''
  try {
    while (true) {
      const { value, done } = await reader.read()
      if (done) break
      buffer += decoder.decode(value, { stream: true })
      const frames = buffer.split('\n\n')
      buffer = frames.pop() || ''
      for (const frame of frames) {
        const parsed = parseSseFrame(frame, state)
        if (parsed) onEvent(parsed.event, parsed.payload)
      }
    }
  } finally {
    reader.cancel()
  }
}

/**
 * 读取一条回答的事件流；连接在收到 done 之前断开时，用最后收到的事件 id 续传，
 * 服务端只补发之后的事件。回答在服务端独立生成，所以续传不会重新提问。
 */
export async function streamWithResume(open, resume, onEvent, { attempts = 3, delayMs = 1000 } = {}) {
  const state = newStreamState()
  try {
    await readSse(await open(), onEvent, state)
  } catch (err) {
    if (err?.name === 'AbortError' || !state.messageId) throw err
  }
  for (let attempt = 1; !state.done && state.messageId && attempt <= attempts; attempt++) {
    await new Promise((resolve) => setTimeout(resolve, delayMs * attempt))
    try {
      await readSse(await resume(state), onEvent, state)
    } catch (err) {
      if (err?.name === 'AbortError') throw err
    }
  }
  if (!state.done && state.messageId) throw new Error('stream interrupted')
}
