export interface ChatRequest {
  query: string
  session_id: string | null
  model: string
}

export interface ChatResponse {
  answer: string
  session_id: string
  is_final: boolean
  candidates?: string[]
}

export interface HealthResponse {
  vector_store: {
    points: number
  }
  graph_store: {
    nodes: number
  }
}

export interface Message {
  role: 'user' | 'bot'
  content: string
  candidates?: string[]
  isFinal?: boolean
}
