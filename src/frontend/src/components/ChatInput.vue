<script setup lang="ts">
import { ref } from 'vue'

const props = defineProps<{
  isStreaming: boolean
}>()

const emit = defineEmits<{
  send: [query: string]
}>()

const input = ref('')

const staticChips = [
  { icon: '🤰', label: 'ตั้งครรภ์' },
  { icon: '🫘', label: 'โรคไต' },
  { icon: '🫁', label: 'โรคตับ' },
  { icon: '🍺', label: 'ดื่มแอลกอฮอล์' },
  { icon: '⚠️', label: 'แพ้เซฟาโลสปอริน' },
]

function handleSend() {
  const q = input.value.trim()
  if (!q || props.isStreaming) return
  emit('send', q)
  input.value = ''
}
</script>

<template>
  <div class="border-t border-[var(--color-border)] glass p-4">
    <div class="flex items-center gap-2 mb-3 flex-wrap">
      <span class="text-xs text-[var(--color-muted)]">ข้อควรระวัง:</span>
      <span
        v-for="chip in staticChips"
        :key="chip.label"
        class="text-xs px-3 py-1.5 rounded-full border border-[var(--color-border)] bg-[var(--color-surface2)] text-[var(--color-muted)] flex items-center gap-1"
      >
        <span class="text-xs">{{ chip.icon }}</span>
        {{ chip.label }}
      </span>
    </div>
    <div class="flex gap-2.5 items-end">
      <div class="flex-1 relative">
        <textarea
          v-model="input"
          :disabled="isStreaming"
          class="w-full bg-[var(--color-bg)]/80 border border-[var(--color-border)] rounded-2xl px-4 py-3 text-sm text-[var(--color-text)] font-sans resize-none outline-none h-11 max-h-32 overflow-y-auto transition-all duration-200 focus:border-[var(--color-accent)] input-glow disabled:opacity-40 placeholder:text-[var(--color-muted)]/60"
          placeholder="พิมพ์ข้อความที่นี่..."
          rows="1"
          @keydown.enter.exact.prevent="handleSend"
        />
      </div>
      <button
        :disabled="isStreaming || !input.trim()"
        class="h-11 px-5 rounded-2xl font-semibold text-sm cursor-pointer transition-all duration-200 disabled:opacity-40 disabled:cursor-not-allowed flex items-center gap-1.5 text-white shadow-lg"
        :style="{ background: isStreaming || !input.trim() ? 'var(--color-surface2)' : 'linear-gradient(135deg, #4f6ef6, #7c5ce7)' }"
        @click="handleSend"
      >
        <span v-if="isStreaming" class="flex gap-1">
          <span class="w-1.5 h-1.5 rounded-full bg-white/60 animate-bounce" style="animation-delay: 0s" />
          <span class="w-1.5 h-1.5 rounded-full bg-white/60 animate-bounce" style="animation-delay: 0.1s" />
          <span class="w-1.5 h-1.5 rounded-full bg-white/60 animate-bounce" style="animation-delay: 0.2s" />
        </span>
        <template v-else>
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><line x1="22" y1="2" x2="11" y2="13"/><polygon points="22 2 15 22 11 13 2 9 22 2"/></svg>
          ส่ง
        </template>
      </button>
    </div>
  </div>
</template>
