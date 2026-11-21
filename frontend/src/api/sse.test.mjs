import assert from 'node:assert/strict'
import { test } from 'node:test'
import { streamWithResume } from './sse.mjs'

const frame = (event, payload, id) => `${id === undefined ? '' : `id: ${id}\n`}event: ${event}\ndata: ${JSON.stringify(payload)}\n\n`

function response(text, { failAfter = false } = {}) {
  const chunks = [new TextEncoder().encode(text)]
  return {
    body: {
      getReader: () => ({
        read: async () => {
          if (chunks.length) return { value: chunks.shift(), done: false }
          if (failAfter) throw new TypeError('network error')
          return { done: true }
        },
        cancel: () => {},
      }),
    },
  }
}

const start = frame('message_start', { id: 'm1' }) + frame('message', { response_type: 'tool_call', name: 'search' }, 0)
const rest = frame('message', { response_type: 'answer', content: 'final', done: true }, 1) + frame('done', { message_id: 'm1' }, 1)

test('a dropped connection resumes after the last event id instead of starting over', async () => {
  const events = []
  const resumedFrom = []
  await streamWithResume(
    async () => response(start, { failAfter: true }),
    async (state) => {
      resumedFrom.push([state.messageId, state.lastEventId])
      return response(rest)
    },
    (event, payload) => events.push(payload.response_type || event),
    { delayMs: 0 },
  )

  assert.deepEqual(resumedFrom, [['m1', '0']])
  assert.deepEqual(events, ['message_start', 'tool_call', 'answer', 'done'])
})

test('a stream that closes without a done event is resumed too', async () => {
  let resumes = 0
  await streamWithResume(async () => response(start), async () => (++resumes, response(rest)), () => {}, { delayMs: 0 })

  assert.equal(resumes, 1)
})

test('a complete stream is not resumed', async () => {
  let resumes = 0
  await streamWithResume(async () => response(start + rest), async () => (++resumes, response(rest)), () => {}, { delayMs: 0 })

  assert.equal(resumes, 0)
})

test('gives up after the configured attempts and reports the interruption', async () => {
  let resumes = 0
  const failing = async () => {
    resumes++
    throw new TypeError('offline')
  }

  await assert.rejects(streamWithResume(async () => response(start, { failAfter: true }), failing, () => {}, { attempts: 2, delayMs: 0 }), /stream interrupted/)
  assert.equal(resumes, 2)
})

test('a request that fails before any message exists is surfaced, not resumed', async () => {
  let resumes = 0
  const open = async () => {
    throw new Error('stream request failed: 500')
  }

  await assert.rejects(streamWithResume(open, async () => (++resumes, response(rest)), () => {}, { delayMs: 0 }), /500/)
  assert.equal(resumes, 0)
})

test('aborting stops without resuming', async () => {
  let resumes = 0
  const open = async () => {
    throw Object.assign(new Error('aborted'), { name: 'AbortError' })
  }

  await assert.rejects(streamWithResume(open, async () => (++resumes, response(rest)), () => {}, { delayMs: 0 }), { name: 'AbortError' })
  assert.equal(resumes, 0)
})
