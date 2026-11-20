import { useState } from 'react'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import AssemblyWorkshop, { arrangeBottomUp } from '../AssemblyWorkshop'
import { defaultTransformerConfig, defaultTrainingConfig, type AssemblyGraph, type ModelConfig } from '../../types'
import { buildPretrainPayload } from '../../services/api'
import { estimateParams } from '../../utils/estimate'

const body: AssemblyGraph = { version: 1, nodes: [
  { id: 'entry', kind: 'input_hidden' }, { id: 'norm', kind: 'rmsnorm' },
  { id: 'attn', kind: 'attention', params: { heads: 4, kv_heads: 2 } }, { id: 'add', kind: 'add' }, { id: 'exit', kind: 'output_hidden' },
], edges: [{ id: 'a', source: 'entry', target: 'norm', targetHandle: 'in' },
  { id: 'b', source: 'norm', target: 'attn', targetHandle: 'in' }, { id: 'c', source: 'attn', target: 'add', targetHandle: 'a' }, { id: 'r', source: 'entry', target: 'add', targetHandle: 'b' }, { id: 'd', source: 'add', target: 'exit', targetHandle: 'in' }] }
const template = { d_model: 64, n_heads: 4, n_kv_heads: 2, d_ff: 128, n_layers: 2, max_seq_len: 256, dropout: 0,
  assembly: { version: 1, nodes: [{ id: 'tokens', kind: 'tokens' }, { id: 'embedding', kind: 'embedding' },
    { id: 'block', kind: 'repeat', params: { count: 2 }, body }, { id: 'head', kind: 'head' }],
    edges: [{ id: 'e1', source: 'tokens', target: 'embedding', targetHandle: 'in' },
      { id: 'e2', source: 'embedding', target: 'block', targetHandle: 'in' }, { id: 'e3', source: 'block', target: 'head', targetHandle: 'in' }] } as AssemblyGraph }
const catalog = ['llama3', 'gemma1', 'qwen25'].map((id, i) => ({ id, name: ['Llama 3', 'Gemma 1', 'Qwen 2.5'][i], description: 'Editable decoder', source: 'https://example.com/reference', sizes: { nano: { d_model:64,n_layers:2,n_heads:4,n_kv_heads:1,d_ff:224,max_seq_len:256 }, tiny: { d_model:128,n_layers:4,n_heads:4,n_kv_heads:1,d_ff:448,max_seq_len:512 } } }))
let latest: ModelConfig
let inspection: Record<string, unknown>
let requests: { url: string; body?: unknown }[]
function Host() {
  const [cfg, setCfg] = useState(defaultTransformerConfig())
  latest = cfg
  return <AssemblyWorkshop config={cfg} onChange={setCfg} />
}
beforeEach(() => {
  requests = []
  inspection = { valid: true, parameters: 100000, fingerprint: 'example', expanded_bricks: 12 }
  vi.stubGlobal('fetch', vi.fn(async (url: string, options?: RequestInit) => {
    requests.push({ url, body: options?.body ? JSON.parse(String(options.body)) : undefined })
    if (url.endsWith('/templates')) return { ok: true, json: async () => ({ templates: catalog }) }
    if (url.endsWith('/templates/build')) {
      const payload = JSON.parse(String(options!.body))
      return { ok: true, json: async () => ({ ...structuredClone(template), ...catalog.find(f => f.id === payload.family)!.sizes[payload.size as 'nano' | 'tiny'], assembly_template: payload.family }) }
    }
    if (url.endsWith('/resize')) {
      const payload = JSON.parse(String(options!.body)), cfg = payload.model_cfg
      cfg.d_model = payload.width; cfg.n_layers = payload.layers; cfg.max_seq_len = payload.context
      cfg.assembly.nodes.find((n: { kind: string }) => n.kind === 'repeat').params.count = payload.layers
      return { ok: true, json: async () => ({ model_cfg: cfg }) }
    }
    return { ok: true, json: async () => url.endsWith('/template') ? template : url.endsWith('/tokenizer') ? { vocab_size: 321 } : url.endsWith('/probe') ? { message: 'Gradients verified.', device: 'cuda' } : inspection }
  }))
})
async function guided() {
  render(<Host />)
  const button = await screen.findByRole('button', { name: /Explore a small Transformer/ })
  await waitFor(() => expect((button as HTMLButtonElement).disabled).toBe(false))
  fireEvent.click(button)
  await screen.findByText('Model Assembly Workshop')
}

describe('Executable model workshop', () => {
  it('creates a real graph, opens a repeat block and persists internal changes to the training payload', async () => {
    await guided()
    const block = document.querySelector('.react-flow__node[data-id="block"]')!
    fireEvent.doubleClick(block)
    await screen.findByRole('button', { name: 'Block 1' })
    fireEvent.click(screen.getByRole('button', { name: /＋ Feed-Forward Network/ }))
    const changed = latest.assembly!.nodes.find(n => n.id === 'block')!.body!
    expect(changed.nodes.some(n => n.kind === 'swiglu')).toBe(true)
    const payload = buildPretrainPayload(latest as unknown as Record<string, unknown>, defaultTrainingConfig())
    expect((payload.model_cfg as ModelConfig).assembly).toEqual(latest.assembly)
    fireEvent.click(screen.getByRole('button', { name: /Undo/ }))
    expect(latest.assembly!.nodes.find(n => n.id === 'block')!.body!.nodes.some(n => n.kind === 'swiglu')).toBe(false)
  })
  it('can start blank, add an embedding and explain its role', async () => {
    render(<Host />)
    fireEvent.click(screen.getByRole('button', { name: /Start from a blank canvas/ }))
    expect(latest.assembly!.edges).toHaveLength(0)
    fireEvent.click(screen.getByRole('button', { name: /＋ Token Embedding/ }))
    expect(latest.assembly!.nodes.filter(n => n.kind === 'embedding')).toHaveLength(1)
    expect(screen.getByText(/A learned table maps/)).toBeTruthy()
  })
  it('shows a pedagogical backend error and focuses the brick inside its block', async () => {
    await guided()
    inspection = { valid: false, errors: [{ node_id: 'attn', path: ['block'], message: 'Attention: expected dimension 64, received 32.', hint: 'Match the dimensions.' }] }
    fireEvent.change(screen.getByLabelText('Model dimension'), { target: { value: '32' } })
    await screen.findByText('Attention: expected dimension 64, received 32.')
    expect((screen.getByRole('button', { name: /Test forward & backward/ }) as HTMLButtonElement).disabled).toBe(true)
    fireEvent.click(screen.getByRole('button', { name: 'Show affected layer' }))
    await screen.findByRole('button', { name: 'Block 1' })
    expect(screen.getByText(/Each token attends/)).toBeTruthy()
  })
  it('runs the real probe API with the complete assembly and labels the result', async () => {
    await guided()
    const tryButton = screen.getByRole('button', { name: /Test forward & backward/ })
    await waitFor(() => expect((tryButton as HTMLButtonElement).disabled).toBe(false))
    fireEvent.click(tryButton)
    await screen.findByText(/Gradients verified.*cuda/)
    const request = requests.find(r => r.url.endsWith('/probe'))!
    expect((request.body as { model_cfg: ModelConfig }).model_cfg.assembly).toEqual(latest.assembly)
  })
  it('uses the actual tokenizer vocabulary in the model and training payload', async () => {
    await guided()
    fireEvent.click(document.querySelector('.react-flow__node[data-id="tokens"]')!)
    fireEvent.change(screen.getByLabelText('Local tokenizer or Hugging Face ID'), { target: { value: '/tmp/my-tokenizer' } })
    fireEvent.click(screen.getByRole('button', { name: 'Read tokenizer' }))
    await screen.findByText('321 tokens read from tokenizer.')
    expect(latest.vocab_size).toBe(321)
    expect((buildPretrainPayload(latest as unknown as Record<string, unknown>, defaultTrainingConfig()).model_cfg as ModelConfig).vocab_size).toBe(321)
    expect(requests.find(r => r.url.endsWith('/tokenizer'))!.body).toEqual({ name: '/tmp/my-tokenizer' })
  })
  it('places inputs below outputs at every graph level and preserves the computation', async () => {
    await guided()
    const graph = latest.assembly!
    const position = (id: string) => graph.nodes.find(n => n.id === id)!.position!.y
    expect(position('tokens')).toBeGreaterThan(position('embedding'))
    expect(position('embedding')).toBeGreaterThan(position('block'))
    expect(position('block')).toBeGreaterThan(position('head'))
    const inner = graph.nodes.find(n => n.id === 'block')!.body!
    expect(inner.nodes.find(n => n.id === 'entry')!.position!.y).toBeGreaterThan(inner.nodes.find(n => n.id === 'exit')!.position!.y)
    const embedding = document.querySelector('.react-flow__node[data-id="embedding"]')!
    expect(embedding.querySelector('.react-flow__handle-bottom.target')).toBeTruthy()
    expect(embedding.querySelector('.react-flow__handle-top.source')).toBeTruthy()
    const old = structuredClone(graph)
    old.nodes.forEach((n, i) => { n.position = { x: 0, y: i * 145 } })
    const arranged = arrangeBottomUp(old)
    expect(arranged.edges).toEqual(graph.edges)
    expect(arranged.nodes.find(n => n.id === 'tokens')!.position!.y).toBeGreaterThan(arranged.nodes.find(n => n.id === 'head')!.position!.y)
    fireEvent.doubleClick(document.querySelector('.react-flow__node[data-id="block"]')!)
    const add = document.querySelector('.react-flow__node[data-id="add"]')!
    expect(add.querySelector('.react-flow__handle-bottom[data-handleid="a"]')).toBeTruthy()
    expect(add.querySelector('.react-flow__handle-left[data-handleid="b"]')).toBeTruthy()
    const blank = arrangeBottomUp({ version: 1, nodes: [{ id: 'tokens', kind: 'tokens' }, { id: 'head', kind: 'head' }], edges: [] })
    expect(blank.nodes[0].position!.y).toBeGreaterThan(blank.nodes[1].position!.y)

  })
  it('persists attention mechanisms and positional methods without keeping irrelevant parameters', async () => {
    await guided()
    fireEvent.doubleClick(document.querySelector('.react-flow__node[data-id="block"]')!)
    fireEvent.click(document.querySelector('.react-flow__node[data-id="attn"]')!)
    expect((screen.getByLabelText('Attention positional method') as HTMLSelectElement).value).toBe('rope')
    fireEvent.change(screen.getByLabelText('Attention mechanism'), { target: { value: 'mla' } })
    fireEvent.change(screen.getByLabelText('Latent KV dimension'), { target: { value: '12' } })
    fireEvent.change(screen.getByLabelText('Attention positional method'), { target: { value: 'alibi' } })
    expect(screen.queryByLabelText('RoPE fraction')).toBeNull()
    fireEvent.click(screen.getByLabelText('Sigmoid output gate'))
    const params = () => latest.assembly!.nodes.find(n => n.id === 'block')!.body!.nodes.find(n => n.id === 'attn')!.params!
    expect(params()).toMatchObject({ mechanism: 'mla', latent: 12, position: 'alibi', gate: true })
    expect(params().kv_heads).toBeUndefined()
    fireEvent.change(screen.getByLabelText('Attention mechanism'), { target: { value: 'mha' } })
    expect(params().latent).toBeUndefined()
    expect(screen.queryByLabelText('Latent KV dimension')).toBeNull()
    expect(buildPretrainPayload(latest as unknown as Record<string, unknown>, defaultTrainingConfig()).model_cfg).toMatchObject({ assembly: latest.assembly })
  })
  it('configures an actual FFN variant and adds a separate absolute-position layer', async () => {
    await guided()
    fireEvent.click(screen.getByRole('button', { name: /＋ Feed-Forward Network/ }))
    fireEvent.change(screen.getByLabelText('FFN activation'), { target: { value: 'gelu' } })
    fireEvent.click(screen.getByLabelText('Linear biases'))
    expect(latest.assembly!.nodes.find(n => n.kind === 'swiglu')!.params).toMatchObject({ activation: 'gelu', bias: true })
    fireEvent.click(screen.getByRole('button', { name: /＋ Positional Encoding/ }))
    fireEvent.change(screen.getByLabelText('Absolute position encoding'), { target: { value: 'learned' } })
    expect(latest.assembly!.nodes.find(n => n.kind === 'position')!.params).toMatchObject({ encoding: 'learned' })
    expect(screen.queryByLabelText('Sinusoidal base')).toBeNull()
    const cfg = { ...defaultTransformerConfig(), d_model: 64, vocab_size: 32, max_seq_len: 256, assembly: { version: 1, nodes: [{ id: 'p', kind: 'position', params: { encoding: 'learned' } }], edges: [] } as AssemblyGraph }
    expect(estimateParams(cfg)).toBe(256 * 64)
  })
  it('builds a known architecture as editable nodes and can undo the replacement', async () => {
    await guided()
    const before = structuredClone(latest)
    fireEvent.click(screen.getByRole('button', { name: 'Architecture templates' }))
    await screen.findByRole('option', { name: 'Qwen 2.5' })
    fireEvent.change(screen.getByLabelText('Architecture template'), { target: { value: 'qwen25' } })
    fireEvent.change(screen.getByLabelText('Starting size'), { target: { value: 'tiny' } })
    fireEvent.click(screen.getByRole('button', { name: 'Build editable template' }))
    await waitFor(() => expect(latest.assembly_template).toBe('qwen25'))
    expect(latest.d_model).toBe(128)
    expect(latest.tokenizer_name).toBe(before.tokenizer_name)
    fireEvent.doubleClick(document.querySelector('.react-flow__node[data-id="block"]')!)
    fireEvent.click(screen.getByRole('button', { name: /＋ Feed-Forward Network/ }))
    expect(latest.assembly!.nodes.find(n => n.id === 'block')!.body!.nodes.some(n => n.kind === 'swiglu')).toBe(true)
    fireEvent.click(screen.getByRole('button', { name: /↶ Undo/ }))
    fireEvent.click(screen.getByRole('button', { name: /↶ Undo/ }))
    expect(latest).toEqual(before)
  })
  it('resizes the current graph globally through the runtime without dropping local changes', async () => {
    await guided()
    fireEvent.doubleClick(document.querySelector('.react-flow__node[data-id="block"]')!)
    fireEvent.click(document.querySelector('.react-flow__node[data-id="attn"]')!)
    fireEvent.change(screen.getByLabelText('Attention positional method'), { target: { value: 'alibi' } })
    fireEvent.change(screen.getByLabelText('Global model dimension'), { target: { value: '128' } })
    fireEvent.change(screen.getByLabelText('Global layer count'), { target: { value: '4' } })
    fireEvent.click(screen.getByRole('button', { name: 'Apply global size' }))
    await waitFor(() => expect(latest.d_model).toBe(128))
    expect(latest.n_layers).toBe(4)
    expect(latest.assembly!.nodes.find(n => n.id === 'block')!.body!.nodes.find(n => n.id === 'attn')!.params!.position).toBe('alibi')
    expect(requests.find(r => r.url.endsWith('/resize'))!.body).toMatchObject({ width:128, layers:4 })
  })
  it('estimates the parameters of the graph rather than the legacy layer settings', () => {
    const cfg = { ...defaultTransformerConfig(), d_model: 64, vocab_size: 32, assembly: template.assembly }
    const count = estimateParams(cfg)
    expect(estimateParams({ ...cfg, n_layers: 63, d_ff: 60000 })).toBe(count)
    const copy = structuredClone(cfg)
    copy.assembly.nodes.find(n => n.kind === 'repeat')!.params!.count = 3
    expect(estimateParams(copy)).toBeGreaterThan(count)
  })
})
