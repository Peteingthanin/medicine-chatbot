<script setup lang="ts">
defineProps<{
  role: 'user' | 'bot'
  content: string
  candidates?: string[]
  isFinal?: boolean
}>()
</script>

<template>
  <div
    :class="[
      'flex gap-3 msg-enter',
      role === 'user' ? 'justify-end' : 'justify-start',
    ]"
  >
    <div
      v-if="role === 'bot'"
      data-testid="bot-avatar"
      class="w-8 h-8 rounded-full bg-gradient-to-br from-[var(--color-accent)] to-[var(--color-accent-2)] flex items-center justify-center text-white text-xs font-semibold shrink-0 shadow-md"
    >
      AI
    </div>

    <div
      data-testid="message-bubble"
      :class="[
        'max-w-[75%] leading-relaxed text-sm px-4 py-3 rounded-2xl whitespace-pre-wrap shadow-lg',
        role === 'user'
          ? 'self-end rounded-br-md text-white'
          : 'self-start rounded-bl-md bg-[var(--color-surface2)] border border-[var(--color-border)] text-[var(--color-text)]',
      ]"
      :style="role === 'user' ? { background: 'linear-gradient(135deg, #4f6ef6, #7c5ce7)' } : {}"
    >
      {{ content }}
      <div
        v-if="candidates && candidates.length > 0"
        class="mt-3 pt-2.5 border-t border-[var(--color-border)] flex flex-wrap gap-1.5"
      >
        <span
          v-for="c in candidates"
          :key="c"
          class="text-xs px-2.5 py-1 rounded-full bg-[var(--color-purple)]/10 border border-[var(--color-purple)]/30 text-[var(--color-purple)]"
        >
          💊 {{ c }}
        </span>
      </div>
      <div
        v-if="isFinal"
        class="mt-2.5 text-xs flex items-center gap-1 text-[var(--color-green)]"
      >
        <span>✓</span> เสร็จสิ้นการวิเคราะห์
      </div>
    </div>

    <div
      v-if="role === 'user'"
      class="w-8 h-8 rounded-full bg-gradient-to-br from-[#4f6ef6] to-[#7c5ce7] flex items-center justify-center text-white text-xs font-semibold shrink-0 shadow-md"
    >
      U
    </div>
  </div>
</template>
