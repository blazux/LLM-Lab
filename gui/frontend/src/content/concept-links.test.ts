import { describe, expect, it } from 'vitest'
import { conceptsForLayer } from './concept-links'
import type { AssemblyNode } from '../types'

describe('Understanding AI links', () => {
  it('follows the selected attention mechanism and position method', () => {
    const node: AssemblyNode = { id: 'a', kind: 'attention', params: { mechanism: 'gqa', position: 'rope' } }
    expect(conceptsForLayer(node).map(link => link.id)).toEqual(['gqa', 'rope'])
    node.params = { mechanism: 'mqa', position: 'alibi' }
    expect(conceptsForLayer(node).map(link => link.id)).toEqual(['mqa', 'alibi'])
    node.params = { mechanism: 'mha', position: 'none' }
    expect(conceptsForLayer(node).map(link => link.id)).toEqual(['transformer'])
    node.params = {}
    expect(conceptsForLayer(node).map(link => link.id)).toEqual(['gqa', 'rope'])
  })

  it('links tokens and normalization to their own explanations', () => {
    for (const kind of ['tokens', 'rmsnorm', 'layernorm'] as const) {
      const link = conceptsForLayer({ id: kind, kind })[0]
      expect(link.url).toBe(`https://www.thelastserver.com/understanding-ai/#tech-${kind === 'tokens' ? 'bpe' : kind}`)
    }
  })

  it('does not describe a plain ReLU network as a gated FFN', () => {
    expect(conceptsForLayer({ id: 'f', kind: 'swiglu', params: { activation: 'relu' } })).toEqual([])
    expect(conceptsForLayer({ id: 'f', kind: 'swiglu', params: { activation: 'geglu' } })[0].id).toBe('swiglu')
  })
})
