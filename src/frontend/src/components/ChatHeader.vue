<script setup lang="ts">
import type { HealthResponse } from '../types/chat'

defineProps<{
  health: HealthResponse | null
  loading: boolean
}>()
</script>

<template>
  <header class="glass border-b px-6 py-3 flex items-center gap-3 backdrop-blur-xl">
    <div class="w-9 h-9 rounded-xl bg-gradient-to-br from-[var(--color-accent)] to-[var(--color-accent-2)] flex items-center justify-center text-white font-bold text-sm shadow-lg shadow-[var(--color-glow)]">
      AI
    </div>
    <div>
      <h1 class="text-sm font-semibold tracking-tight">Medication RAG</h1>
      <p class="text-xs text-[var(--color-muted)]">AI Pharmacist</p>
    </div>
    <div class="ml-auto flex items-center gap-2">
      <template v-if="loading">
        <span class="w-1.5 h-1.5 rounded-full bg-[var(--color-orange)] animate-pulse" />
        <span class="text-xs text-[var(--color-muted)]">กำลังตรวจสอบ</span>
      </template>
      <template v-else-if="health">
        <span class="pulse-dot" />
        <span class="text-xs text-[var(--color-muted)]">
          {{ health.vector_store.points }} vectors · {{ health.graph_store.nodes }} nodes
        </span>
      </template>
      <template v-else>
        <span class="w-1.5 h-1.5 rounded-full bg-[var(--color-red)]" />
        <span class="text-xs text-[var(--color-red)]">offline</span>
      </template>
    </div>
  </header>
</template>
