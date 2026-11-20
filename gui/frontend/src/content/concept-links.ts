import type { AssemblyNode } from '../types'

export interface ConceptLink { id: string; title: string; url: string }
const TITLES: Record<string, string> = {
  bpe: 'Text and tokenization', transformer: 'The Transformer',
  rmsnorm: 'RMSNorm', layernorm: 'LayerNorm', mqa: 'Multi-query attention',
  gqa: 'Grouped-query attention', mla: 'Latent attention',
  rope: 'Rotary positions (RoPE)', alibi: 'Distance bias (ALiBi)',
  swiglu: 'Gated feed-forward networks',
}

/** Link the operation actually configured, including attention variants. */
export function conceptsForLayer(node: AssemblyNode): ConceptLink[] {
  const params = node.params ?? {}
  let ids: string[] = []
  switch (node.kind) {
    case 'tokens': ids = ['bpe']; break
    case 'embedding': case 'head': case 'repeat': case 'add': ids = ['transformer']; break
    case 'rmsnorm': case 'layernorm': ids = [node.kind]; break
    case 'attention': {
      const mechanism = params.mechanism ?? 'gqa'
      if (mechanism === 'mha') ids = ['transformer']
      else if (['mqa', 'gqa', 'mla'].includes(String(mechanism))) ids = [String(mechanism)]
      const position = params.position ?? 'rope'
      if (position === 'rope' || position === 'alibi') ids.push(position)
      break
    }
    case 'swiglu':
      if (['swiglu', 'geglu', 'geglu_tanh', 'reglu'].includes(String(params.activation ?? 'swiglu'))) ids = ['swiglu']
      break
  }
  return ids.map(id => ({ id, title: TITLES[id], url: `https://www.thelastserver.com/understanding-ai/#tech-${id}` }))
}
