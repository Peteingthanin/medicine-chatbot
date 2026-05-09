import { describe, it, expect, beforeEach, vi } from 'vitest'
import { useChat } from './useChat'

function mockFetch(response: unknown, ok = true) {
  return vi.fn().mockResolvedValue({
    ok,
    json: () => Promise.resolve(response),
    text: () => Promise.resolve(JSON.stringify(response)),
  })
}

describe('useChat', () => {
  beforeEach(() => {
    vi.restoreAllMocks()
  })

  it('starts with a welcome message', () => {
    const { messages, isStreaming } = useChat()

    expect(messages.value).toHaveLength(1)
    expect(messages.value[0].role).toBe('bot')
    expect(messages.value[0].content).toContain('ผู้ช่วยเภสัชกร')
    expect(isStreaming.value).toBe(false)
  })

  it('starts with null session_id', () => {
    const { sessionId } = useChat()
    expect(sessionId.value).toBeNull()
  })

  it('appends user message and bot response on send', async () => {
    globalThis.fetch = mockFetch({
      answer: 'ลองใช้ยาพาราเซตามอล',
      session_id: 'sess-1',
      is_final: false,
    })

    const { messages, send } = useChat()
    await send('ปวดหัว')

    expect(messages.value).toHaveLength(3)
    expect(messages.value[1].role).toBe('user')
    expect(messages.value[1].content).toBe('ปวดหัว')
    expect(messages.value[2].role).toBe('bot')
    expect(messages.value[2].content).toBe('ลองใช้ยาพาราเซตามอล')
  })

  it('saves session_id from response', async () => {
    globalThis.fetch = mockFetch({
      answer: 'ตอบกลับ',
      session_id: 'sess-abc',
      is_final: false,
    })

    const { sessionId, send } = useChat()
    await send('คำถาม')

    expect(sessionId.value).toBe('sess-abc')
  })

  it('sends session_id in request after first turn', async () => {
    globalThis.fetch = mockFetch({
      answer: 'รอบแรก',
      session_id: 'sess-1',
      is_final: false,
    })

    const { send, sessionId } = useChat()
    await send('คำถามแรก')

    expect(sessionId.value).toBe('sess-1')

    const mock2 = mockFetch({
      answer: 'รอบสอง',
      session_id: 'sess-1',
      is_final: false,
    })
    globalThis.fetch = mock2

    await send('คำถามต่อ')

    const body = JSON.parse((mock2.mock.calls[0]?.[1] as RequestInit)?.body as string)
    expect(body.session_id).toBe('sess-1')
  })

  it('resets session when is_final is true', async () => {
    globalThis.fetch = mockFetch({
      answer: 'วิเคราะห์เสร็จสิ้น',
      session_id: 'sess-final',
      is_final: true,
    })

    const { sessionId, send } = useChat()
    await send('ตรวจสอบ')

    expect(sessionId.value).toBeNull()
  })

  it('includes candidates in bot message when present', async () => {
    globalThis.fetch = mockFetch({
      answer: 'ยาที่ตรงกับอาการ',
      session_id: 'sess-2',
      is_final: false,
      candidates: ['Paracetamol', 'Ibuprofen'],
    })

    const { messages, send } = useChat()
    await send('ปวด')

    expect(messages.value[2].candidates).toEqual(['Paracetamol', 'Ibuprofen'])
  })

  it('sets isStreaming during send', async () => {
    let resolve: (v: unknown) => void = () => {}
    const promise = new Promise((r) => { resolve = r })

    globalThis.fetch = vi.fn().mockReturnValue(promise)

    const { isStreaming, send } = useChat()
    const sendPromise = send('test')

    expect(isStreaming.value).toBe(true)

    resolve({ ok: true, json: () => Promise.resolve({ answer: 'ok', session_id: 's', is_final: false }), text: () => Promise.resolve('') })
    await sendPromise

    expect(isStreaming.value).toBe(false)
  })

  it('handles fetch error gracefully', async () => {
    const consoleSpy = vi.spyOn(console, 'error').mockImplementation(() => {})
    globalThis.fetch = vi.fn().mockRejectedValue(new Error('Network error'))

    const { messages, send } = useChat()
    await send('test')

    expect(messages.value).toHaveLength(3)
    expect(messages.value[2].content).toContain('เกิดข้อผิดพลาด')
    consoleSpy.mockRestore()
  })

  it('checkHealth stores health data', async () => {
    globalThis.fetch = mockFetch({
      vector_store: { points: 100 },
      graph_store: { nodes: 50 },
    })

    const { health, checkHealth } = useChat()
    await checkHealth()

    expect(health.value).toEqual({
      vector_store: { points: 100 },
      graph_store: { nodes: 50 },
    })
  })

  it('reset clears all state', async () => {
    globalThis.fetch = mockFetch({
      answer: 'ทดสอบ',
      session_id: 'sess-x',
      is_final: false,
    })

    const { messages, sessionId, send, reset } = useChat()
    await send('hello')

    reset()

    expect(messages.value).toHaveLength(1)
    expect(sessionId.value).toBeNull()
  })
})
