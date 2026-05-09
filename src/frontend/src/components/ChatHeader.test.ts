import { describe, it, expect } from 'vitest'
import { mount } from '@vue/test-utils'
import ChatHeader from './ChatHeader.vue'

describe('ChatHeader', () => {
  it('shows logo and title', () => {
    const wrapper = mount(ChatHeader, {
      props: { health: null, loading: false },
    })
    expect(wrapper.text()).toContain('Medication RAG')
  })

  it('has glass header with backdrop blur', () => {
    const wrapper = mount(ChatHeader, {
      props: { health: null, loading: false },
    })
    expect(wrapper.find('header').classes()).toContain('backdrop-blur-xl')
  })

  it('displays health stats when available', () => {
    const wrapper = mount(ChatHeader, {
      props: {
        health: { vector_store: { points: 150 }, graph_store: { nodes: 75 } },
        loading: false,
      },
    })
    expect(wrapper.text()).toContain('150')
    expect(wrapper.text()).toContain('75')
  })

  it('shows animated pulse dot when healthy', () => {
    const wrapper = mount(ChatHeader, {
      props: {
        health: { vector_store: { points: 10 }, graph_store: { nodes: 5 } },
        loading: false,
      },
    })
    expect(wrapper.find('.pulse-dot').exists()).toBe(true)
  })

  it('shows offline state without pulse', () => {
    const wrapper = mount(ChatHeader, {
      props: { health: null, loading: false },
    })
    expect(wrapper.text()).toContain('offline')
    expect(wrapper.find('.pulse-dot').exists()).toBe(false)
  })

  it('shows checking when loading', () => {
    const wrapper = mount(ChatHeader, {
      props: { health: null, loading: true },
    })
    expect(wrapper.text()).toContain('กำลังตรวจสอบ')
  })
})
