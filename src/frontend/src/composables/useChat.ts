import { ref } from 'vue'
import type { Message, HealthResponse } from '../types/chat'
import { sendChat, checkHealth as apiCheckHealth } from '../api/chat'

const WELCOME_MESSAGE: Message = {
  role: 'bot',
  content: 'สวัสดีครับ ผมคือผู้ช่วยเภสัชกร AI 💊\nบอกอาการหรือโรคของคุณมาได้เลยครับ (เช่น "ผมปวดหัว" หรือ "มีไข้สูง")',
}

export function useChat() {
  const messages = ref<Message[]>([{ ...WELCOME_MESSAGE }])
  const sessionId = ref<string | null>(null)
  const isStreaming = ref(false)
  const health = ref<HealthResponse | null>(null)

  async function send(query: string) {
    messages.value = [...messages.value, { role: 'user', content: query }]
    isStreaming.value = true

    try {
      const data = await sendChat({
        query,
        session_id: sessionId.value,
        model: 'deepseek',
      })

      sessionId.value = data.is_final ? null : data.session_id

      messages.value = [
        ...messages.value,
        {
          role: 'bot',
          content: data.answer,
          candidates: data.candidates,
          isFinal: data.is_final,
        },
      ]
    } catch (err) {
      console.error(err)
      messages.value = [
        ...messages.value,
        {
          role: 'bot',
          content: `❌ เกิดข้อผิดพลาด: ${err instanceof Error ? err.message : 'Unknown error'}`,
        },
      ]
    } finally {
      isStreaming.value = false
    }
  }

  async function checkHealth() {
    try {
      health.value = await apiCheckHealth()
    } catch {
      health.value = null
    }
  }

  function reset() {
    messages.value = [{ ...WELCOME_MESSAGE }]
    sessionId.value = null
    isStreaming.value = false
  }

  return {
    messages,
    sessionId,
    isStreaming,
    health,
    send,
    checkHealth,
    reset,
  }
}
