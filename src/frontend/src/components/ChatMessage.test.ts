import { describe, it, expect } from 'vitest'
import { mount } from '@vue/test-utils'
import ChatMessage from './ChatMessage.vue'

describe('ChatMessage', () => {
  it('renders user message with right alignment', () => {
    const wrapper = mount(ChatMessage, {
      props: { role: 'user', content: 'Hello' },
    })
    expect(wrapper.text()).toContain('Hello')
    expect(wrapper.find('[data-testid="message-bubble"]').classes()).toContain('self-end')
  })

  it('renders bot message with left alignment and avatar', () => {
    const wrapper = mount(ChatMessage, {
      props: { role: 'bot', content: 'Response' },
    })
    expect(wrapper.text()).toContain('Response')
    expect(wrapper.find('[data-testid="message-bubble"]').classes()).toContain('self-start')
    expect(wrapper.find('[data-testid="bot-avatar"]').exists()).toBe(true)
  })

  it('user message has no bot avatar', () => {
    const wrapper = mount(ChatMessage, {
      props: { role: 'user', content: 'Hello' },
    })
    expect(wrapper.find('[data-testid="bot-avatar"]').exists()).toBe(false)
  })

  it('displays candidate chips when present', () => {
    const wrapper = mount(ChatMessage, {
      props: {
        role: 'bot',
        content: 'Response',
        candidates: ['Paracetamol', 'Ibuprofen'],
      },
    })
    expect(wrapper.text()).toContain('Paracetamol')
    expect(wrapper.text()).toContain('Ibuprofen')
  })

  it('has shadow on message bubble', () => {
    const wrapper = mount(ChatMessage, {
      props: { role: 'bot', content: 'Test' },
    })
    expect(wrapper.find('[data-testid="message-bubble"]').classes()).toContain('shadow-lg')
  })

  it('shows final badge when isFinal is true', () => {
    const wrapper = mount(ChatMessage, {
      props: { role: 'bot', content: 'Done', isFinal: true },
    })
    expect(wrapper.text()).toContain('เสร็จสิ้นการวิเคราะห์')
  })

  it('does not show final badge when isFinal is false', () => {
    const wrapper = mount(ChatMessage, {
      props: { role: 'bot', content: 'Thinking', isFinal: false },
    })
    expect(wrapper.text()).not.toContain('เสร็จสิ้นการวิเคราะห์')
  })
})
