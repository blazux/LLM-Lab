import { afterEach } from 'vitest'
import { cleanup } from '@testing-library/react'
class ResizeObserverStub {
  observe() {}
  unobserve() {}
  disconnect() {}
}
Object.defineProperty(globalThis, 'ResizeObserver', { value: ResizeObserverStub })
Object.defineProperty(HTMLElement.prototype, 'offsetWidth', { configurable: true, get: () => 1000 })
Object.defineProperty(HTMLElement.prototype, 'offsetHeight', { configurable: true, get: () => 700 })
window.matchMedia = () => ({ matches: false, addListener() {}, removeListener() {}, addEventListener() {}, removeEventListener() {}, dispatchEvent: () => false, media: '', onchange: null })
afterEach(() => cleanup())
