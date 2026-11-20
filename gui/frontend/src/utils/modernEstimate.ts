import type { AssemblyNode } from '../types'

function nextPrime(n: number): number {
  for (let p = Math.max(2, n + 1); ; p++) {
    let prime = true
    for (let d = 2; d * d <= p; d++) if (p % d === 0) { prime = false; break }
    if (prime) return p
  }
}

/** Matches the reference-backed modules, including fused expert matrices. */
export function modernParameters(node: AssemblyNode, width: number, vocab: number): number | null {
  const p = node.params ?? {}, w = Number(p.width ?? width)
  const n = (key: string, fallback: number) => Number(p[key] ?? fallback)
  const family = String(p.family ?? 'qwen35')
  switch (node.kind) {
    case 'modern_attention': {
      const h = n('heads',4), kv = n('kv_heads',1), d = n('head_dim',16)
      if (family === 'llama4') return 2*w*h*d + 2*w*kv*d
      if (family === 'gemma4') return 2*w*h*d + (p.qk_norm === false ? 0 : d) + (p.kv_shared ? 0 : w*kv*d*(p.k_eq_v && !p.window ? 1 : 2) + (p.qk_norm === false ? 0 : d))
      const base = 3*w*h*d + 2*w*kv*d + (p.qk_norm === false ? 0 : 2*d)
      return base + (family === 'qwen4' ? w*(n('index_heads',2)+1)*n('index_dim',16)+2*n('index_dim',16) : 0)
    }
    case 'gated_deltanet': {
      const kh=n('key_heads',2), vh=n('value_heads',4), kd=n('key_dim',16), vd=n('value_dim',16)
      return w*(2*kh*kd+3*vh*vd+2*vh)+(2*kh*kd+vh*vd)*n('conv_kernel',4)+2*vh+vd
    }
    case 'moe': {
      const e=n('experts',8), f=n('hidden',32), shared=n('shared_hidden',32)
      return 3*e*w*f+w*e+(family==='gemma4' ? w+e : 3*w*shared+(family==='llama4' ? 0 : w))
    }
    case 'gemma4_norm': return p.learned_scale === false ? 0 : w
    case 'qwen_norm': return w
    case 'ple_embedding': return (vocab+w)*n('layers',5)*n('dim',16)+n('dim',16)
    case 'ple_apply': return 2*w*n('dim',16)+w
    case 'gr_read': return n('streams',4)*w*(1+2*n('rank',16)+n('streams',4))
    case 'gr_collapse': return n('streams',4)*w*(1+2*n('rank',16))
    case 'ngram': {
      const heads=(n('ngram_size',3)-1)*n('heads_per_ngram',2), dim=n('dim',64), streams=n('streams',4)
      let sum=0, prime=n('buckets',257)-1
      for(let i=0;i<heads;i++) { prime=nextPrime(prime); sum+=prime }
      return Math.ceil(sum/8)*8*(dim/heads)+dim*w*(streams+1)+streams*w*(3+n('conv_kernel',4))
    }
    case 'scale': case 'gr_write': case 'residual_expand': return 0
    default: return null
  }
}
