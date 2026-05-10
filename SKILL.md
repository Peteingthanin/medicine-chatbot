---
name: medicine-chatbot-patterns
description: Coding patterns extracted from medicine-chatbot — a Thai medication RAG chatbot with FastAPI + Vue 3 + Neo4j + Qdrant
version: 1.0.0
source: local-git-analysis
analyzed_commits: 6
---

# Medicine Chatbot Patterns

## Commit Conventions

Commits use **imperative lowercase descriptions** without conventional commit prefixes:

```
starter pack
add neo4j container retry
Update index.html
Implement Vue as Frontend
Create neo4j.dump
```

**Guideline:** Write commit messages as short imperative phrases describing what the change does.

## Code Architecture

```
src/
├── ai/                           # Python backend (FastAPI)
│   ├── api/                      # FastAPI app + router
│   │   ├── main.py               # App factory, lifespan, static file serving
│   │   └── router.py             # API endpoints (/health, /eval, /chat/converse)
│   ├── config/                   # Environment-based configuration
│   │   ├── app.py                # General settings
│   │   ├── db.py                 # Neo4j + Qdrant connection config
│   │   └── llm.py                # LLM model paths + API keys
│   ├── models/                   # Pydantic schemas
│   │   └── schemas.py
│   ├── retrieval/                # RAG retrieval pipeline
│   │   ├── retriever_vector.py   # Phase 1: Vector-only (Qdrant + embedding)
│   │   └── retriever_hybrid.py   # Phase 2: Graph + Vector (Neo4j Cypher + Qdrant)
│   ├── llm/                      # Local LLM generation
│   │   └── generate.py
│   ├── session/                  # Session store
│   │   └── store.py
│   └── utils/                    # Utilities
│       └── text.py
├── frontend/                     # Vue 3 SPA frontend
│   ├── package.json              # Vue 3.5, Vite 6, Tailwind 4, Vitest
│   ├── vite.config.ts            # Build + dev proxy to :8000
│   ├── vitest.config.ts          # jsdom test environment
│   └── src/
│       ├── main.ts               # Vue app bootstrap
│       ├── App.vue               # Root component
│       ├── style.css             # Tailwind imports + custom theme vars
│       ├── types/chat.ts         # TypeScript interfaces
│       ├── api/chat.ts           # Typed fetch client
│       ├── composables/useChat.ts # Reactive chat state
│       └── components/
│           ├── ChatContainer.vue # Main layout
│           ├── ChatHeader.vue    # Logo, title, health badge
│           ├── ChatMessage.vue   # Message bubble
│           └── ChatInput.vue     # Textarea, send button, info chips
```

## Workflows

### Adding a new API endpoint
1. Define Pydantic models in `src/ai/models/schemas.py`
2. Add route handler in `src/ai/api/router.py`
3. Attach to FastAPI app via `app.include_router(router)` in `main.py`
4. If using external services, configure in `src/ai/config/`

### Adding a new Vue component
1. Create `src/frontend/src/components/ComponentName.vue` with `<script setup lang="ts">`
2. Define props with `defineProps<{ ... }>()`, emits with `defineEmits<{ ... }>()`
3. Create co-located test: `ComponentName.test.ts`
4. Import and use in parent component

### Adding a new Vue composable
1. Create `src/frontend/src/composables/useFeature.ts`
2. Write test first: `useFeature.test.ts` (TDD workflow)
3. Export reactive state (`ref`) + functions from composable
4. Return plain object with all return values

### Infrastructure change
1. Update `docker-compose.yml` for service config
2. Update `Dockerfile` if build process changes
3. These two files often change together — test both locally
4. For startup ordering: use `depends_on` with `condition: service_healthy` + healthchecks
5. For external service retries: add retry logic in Python as a second line of defense

### Deployment
1. `docker compose build` — build images
2. `docker compose up -d --build` — deploy
3. `docker compose logs api` — verify
4. Frontend is baked into the Docker image via multi-stage build — no volume mount needed

## Testing Patterns

- **Framework:** Vitest + @vue/test-utils + jsdom
- **Location:** Co-located: `ComponentName.test.ts` next to `ComponentName.vue`
- **Run:** `npm test` from `src/frontend/`
- **Type check:** `vue-tsc --noEmit` before build

### Test structure (AAA pattern)
```typescript
describe('ComponentName', () => {
  it('describes expected behavior', () => {
    const wrapper = mount(Component, { props: { ... } })
    expect(wrapper.text()).toContain('expected text')
  })
})
```

### Mock pattern (API client testing)
```typescript
function mockFetch(response: unknown, ok = true) {
  return vi.fn().mockResolvedValue({
    ok,
    json: () => Promise.resolve(response),
    text: () => Promise.resolve(JSON.stringify(response)),
  })
}
globalThis.fetch = mockFetch({ answer: '...', session_id: 'sess-1', is_final: false })
```

## Configuration Patterns

### Environment variables
- All config loaded from environment via `os.getenv()` in `src/ai/config/`
- API keys: `DEEPSEEK_API_KEY`, `OPENROUTER_API_KEY`
- DB connections: `NEO4J_URI=bolt://neo4j:7687`, `QDRANT_HOST=qdrant`
- Feature flags: `USE_LOCAL_CHAT_MODEL` (true/false)
- `.env` excluded from git and Docker build via `.dockerignore`

### Docker service dependencies
- API depends on Neo4j with `condition: service_healthy`
- API depends on Qdrant with `condition: service_started`
- Neo4j healthcheck uses `cypher-shell` (not `curl` — not in Neo4j image)
- Python retry logic as defense-in-depth against startup races

## Common Patterns

### Immutable state updates (Vue)
```typescript
// WRONG: mutation
messages.value.push(newMsg)
// CORRECT: new array reference
messages.value = [...messages.value, newMsg]
```

### Graceful degradation (Python)
```python
try:
    self.neo4j_driver.verify_connectivity()
except (ServiceUnavailable, OSError) as e:
    print(f"Warning: Falling back to vector-only.")
    self.neo4j_driver = None
```

### Relative API URLs
```typescript
// Always use relative URLs — never hardcode localhost
const API_BASE = ''  // resolves to current origin
```
