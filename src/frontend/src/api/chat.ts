import type { ChatRequest, ChatResponse, HealthResponse } from '../types/chat'

const API_BASE = import.meta.env.PROD ? '/medicine' : ''

async function request<T>(url: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${url}`, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  })
  if (!res.ok) {
    throw new Error(await res.text())
  }
  return res.json()
}

export function checkHealth(): Promise<HealthResponse> {
  return request<HealthResponse>('/health')
}

export function sendChat(body: ChatRequest): Promise<ChatResponse> {
  return request<ChatResponse>('/chat/converse', {
    method: 'POST',
    body: JSON.stringify(body),
  })
}
