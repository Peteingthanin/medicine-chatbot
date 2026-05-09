<script setup lang="ts">
import { onMounted, nextTick, watch } from 'vue'
import { useChat } from '../composables/useChat'
import ChatHeader from './ChatHeader.vue'
import ChatMessage from './ChatMessage.vue'
import ChatInput from './ChatInput.vue'

const {
  messages,
  isStreaming,
  health,
  checkHealth,
  send,
} = useChat()

onMounted(() => {
  checkHealth()
})

watch(messages, async () => {
  await nextTick()
  const container = document.querySelector('.chat-scroll')
  if (container) {
    container.scrollTo({ top: container.scrollHeight, behavior: 'smooth' })
  }
}, { deep: true })
</script>

<template>
  <div class="flex flex-col h-screen bg-glow">
    <ChatHeader :health="health" :loading="isStreaming" />
    <div class="flex-1 overflow-y-auto chat-scroll bg-dots px-4 py-4 flex flex-col gap-3" id="chatHistory">
      <template v-if="messages.length <= 1 && !isStreaming">
        <div class="flex-1 flex flex-col items-center justify-center text-center px-8 mt-12">
          <div class="w-20 h-20 rounded-3xl bg-gradient-to-br from-[var(--color-accent)] to-[var(--color-accent-2)] flex items-center justify-center text-white text-3xl font-bold shadow-2xl shadow-[var(--color-glow)] mb-6">
            AI
          </div>
          <h2 class="text-lg font-semibold mb-2">AI Pharmacist</h2>
          <p class="text-sm text-[var(--color-muted)] max-w-sm">
            บอกอาการหรือปัญหาสุขภาพของคุณ แล้วผมจะช่วยวิเคราะห์และแนะนำยาเบื้องต้นให้ครับ
          </p>
          <div class="flex gap-2 mt-6 flex-wrap justify-center">
            <span class="text-xs px-3 py-1.5 rounded-full bg-[var(--color-surface2)] border border-[var(--color-border)] text-[var(--color-muted)]">💊 วิเคราะห์อาการ</span>
            <span class="text-xs px-3 py-1.5 rounded-full bg-[var(--color-surface2)] border border-[var(--color-border)] text-[var(--color-muted)]">🔍 แนะนำยา</span>
            <span class="text-xs px-3 py-1.5 rounded-full bg-[var(--color-surface2)] border border-[var(--color-border)] text-[var(--color-muted)]">⚠️ ตรวจสอบข้อห้าม</span>
          </div>
        </div>
      </template>
      <template v-else>
        <ChatMessage
          v-for="(msg, i) in messages"
          :key="i"
          :role="msg.role"
          :content="msg.content"
          :candidates="msg.candidates"
          :is-final="msg.isFinal"
        />
      </template>
    </div>
    <ChatInput
      :is-streaming="isStreaming"
      @send="send"
    />
  </div>
</template>
