"""Executable, typed assembly DAG with optional reference-backed modern bricks.

Graph positions are presentation data. Edges alone determine execution order.
Every declared brick must contribute to the output; unused/disconnected bricks
are rejected rather than silently omitted. Repetition instantiates independent weights.
"""
from contextlib import nullcontext
import math
import re
import hashlib
import json
import torch
from torch import nn
from torch.nn import functional as F


class AssemblyError(ValueError):
    def __init__(self, message, node_id=None, hint=''):
        super().__init__(message)
        self.node_id, self.hint = node_id, hint


def canonical_assembly(graph):
    return {'version': graph['version'], 'nodes': sorted([
        {k: canonical_assembly(v) if k == 'body' else v for k, v in n.items() if k not in {'position', 'label'}}
        for n in graph['nodes']], key=lambda n: n['id']),
        'edges': sorted([{k:v for k,v in e.items() if k != 'id'} for e in graph['edges']],
                        key=lambda e: (e['target'], e.get('targetHandle', 'in'), e['source']))}


def assembly_signature(config):
    value = {'graph': canonical_assembly(config.assembly), 'width': config.d_model, 'vocab': config.vocab_size,
             'context': config.max_seq_len, 'tie_embeddings': config.tie_word_embeddings}
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def validate_assembly(graph, width, vocab_size, body=False, depth=0):
    from .assembly_schema import MODERN_BRICKS, INPUT_KINDS, OUTPUT_KINDS, validate_graph
    def modern(g):
        if not isinstance(g,dict): return False
        if g.get('version')==2: return True
        if not isinstance(g.get('nodes',[]),list): return False
        return any(isinstance(n,dict) and (n.get('kind') in MODERN_BRICKS or n.get('kind') in {'input_tokens','input_kv','input_ple','output_kv'} or n.get('body') and modern(n['body'])) for n in g.get('nodes',[]))
    if modern(graph): return validate_graph(graph,width,vocab_size,body,depth)
    return _validate_legacy(graph,width,vocab_size,body,depth)


def _validate_legacy(graph, width, vocab_size, body=False, depth=0):
    if depth > 3:
        raise AssemblyError('Too many nested blocks.', hint='Use at most three nested block levels.')
    if not isinstance(graph, dict) or graph.get('version') != 1:
        raise AssemblyError('The graph must use format version 1.')
    if set(graph) - {'version', 'nodes', 'edges'}:
        raise AssemblyError('The graph contains unknown properties.')
    nodes, edges = graph.get('nodes', []), graph.get('edges', [])
    if not isinstance(nodes, list) or not isinstance(edges, list) or not 1 <= len(nodes) <= 128 or len(edges) > 256:
        raise AssemblyError('Use between 1 and 128 layers and at most 256 connections.')
    index, incoming, outgoing = {}, {}, {}
    kinds = {'tokens', 'embedding', 'rmsnorm', 'layernorm', 'position', 'attention', 'swiglu', 'add', 'head', 'repeat', 'input_hidden', 'output_hidden'}
    for node in nodes:
        if not isinstance(node, dict):
            raise AssemblyError('Each layer must be an object.')
        id = node.get('id', '')
        if not isinstance(id, str) or not re.fullmatch(r'[A-Za-z][A-Za-z0-9_-]{0,63}', id) or id in index:
            raise AssemblyError('Each layer must have a unique, simple ID.', str(id))
        if set(node) - {'id', 'kind', 'params', 'position', 'label', 'body'}:
            raise AssemblyError('This layer contains unknown properties.', id)
        if node.get('kind') not in kinds:
            raise AssemblyError('This layer is not supported yet.', id)
        if not isinstance(node.get('params', {}), dict):
            raise AssemblyError('Layer parameters must be an object.', id)
        allowed = {'embedding': {'width', 'scale'}, 'rmsnorm': {'width', 'eps', 'unit_offset'}, 'layernorm': {'width', 'eps'}, 'position': {'width', 'encoding', 'base'},
                   'attention': {'width', 'heads', 'kv_heads', 'mechanism', 'position', 'rope_fraction', 'rope_base', 'latent', 'window', 'qk_norm', 'gate', 'qkv_bias', 'rope_style', 'backend'},
                   'swiglu': {'width', 'hidden', 'activation', 'bias'}, 'repeat': {'count'}, 'head': {'width'}}.get(node['kind'], set())
        if set(node.get('params', {})) - allowed:
            raise AssemblyError('This parameter has no effect in this layer.', id, 'Remove the unknown parameter; it cannot be silently ignored.')
        if node['kind'] != 'repeat' and 'body' in node:
            raise AssemblyError('Only a repeated block can contain a subgraph.', id)
        index[id], incoming[id], outgoing[id] = node, {}, []
    for edge in edges:
        if not isinstance(edge, dict) or set(edge) - {'id', 'source', 'target', 'targetHandle'}:
            raise AssemblyError('A connection contains unknown properties.')
        source, target, port = edge.get('source'), edge.get('target'), edge.get('targetHandle', 'in')
        if source not in index or target not in index:
            raise AssemblyError('A connection refers to a missing layer.', target)
        ports = {'a', 'b'} if index[target]['kind'] == 'add' else {'in'}
        if port not in ports or port in incoming[target]:
            raise AssemblyError('Each input port accepts exactly one connection.', target, 'Use Residual Add to combine two paths.')
        incoming[target][port] = source
        outgoing[source].append(target)
    input_kind, output_kind = ('input_hidden', 'output_hidden') if body else ('tokens', 'head')
    starts = [n['id'] for n in nodes if n['kind'] == input_kind]
    ends = [n['id'] for n in nodes if n['kind'] == output_kind]
    if len(starts) != 1 or len(ends) != 1:
        raise AssemblyError('The graph needs exactly one input and one output.', hint=f'Add one {input_kind} and one {output_kind} layer.')
    if any(n['kind'] in ({'tokens', 'embedding', 'head'} if body else {'input_hidden', 'output_hidden'}) for n in nodes):
        raise AssemblyError('This layer belongs at another graph level.')
    if not body and sum(n['kind'] == 'embedding' for n in nodes) != 1:
        raise AssemblyError('This version requires exactly one Token Embedding.')
    shapes, order, pending = {}, [], set(index)
    expanded = len(nodes)
    while pending:
        ready = [n['id'] for n in nodes if n['id'] in pending and all(s in shapes for s in incoming[n['id']].values())]
        if not ready:
            raise AssemblyError('The graph contains a cycle and cannot be executed.', next(iter(pending)), 'Use a repeated block to repeat a computation.')
        for id in ready:
            node, kind = index[id], index[id]['kind']
            required = set() if kind == input_kind else ({'a', 'b'} if kind == 'add' else {'in'})
            if set(incoming[id]) != required:
                raise AssemblyError('This layer needs a connection on every input port.', id, 'Connect the colored input ports, or remove this layer.')
            params = node.get('params', {})
            size = params.get('width', width)
            if type(size) is not int or not 1 <= size <= 8192:
                raise AssemblyError('Dimension must be an integer between 1 and 8192.', id)
            if kind == input_kind:
                shape = ('hidden', width) if body else ('tokens', None)
            elif kind == 'embedding':
                if shapes[incoming[id]['in']][0] != 'tokens':
                    raise AssemblyError('Token Embedding expects token IDs rather than hidden states.', id)
                shape = ('hidden', size)
            elif kind == 'add':
                a, b = (shapes[incoming[id][p]] for p in ['a', 'b'])
                if a != b or a[0] != 'hidden':
                    raise AssemblyError(f'Cannot add tensors with shapes {a} and {b}.', id, 'Both paths must carry hidden states of the same dimension.')
                shape = a
            else:
                actual = shapes[incoming[id]['in']]
                if actual != ('hidden', size):
                    raise AssemblyError(f'This layer expects hidden states of dimension {size}, but receives {actual[1] if actual[0] == "hidden" else "token IDs"}.', id,
                                        'Match the layer dimension to the incoming hidden states.')
                shape = ('logits', vocab_size) if kind == 'head' else actual
            if kind == 'embedding' and params.get('scale', 'sqrt_dim') not in {'sqrt_dim', 'none'}:
                raise AssemblyError('Embedding scale must be sqrt_dim or none.', id)
            if kind == 'rmsnorm' and type(params.get('unit_offset', False)) is not bool:
                raise AssemblyError('RMSNorm unit offset must be boolean.', id)
            if kind == 'attention':
                if params.get('backend', 'auto') not in {'auto', 'math'}:
                    raise AssemblyError('Choose auto SDPA dispatch or the math reference backend.', id)
                heads = params.get('heads', 4)
                mechanism = params.get('mechanism', 'gqa')
                if mechanism not in {'mha', 'mqa', 'gqa', 'mla'}:
                    raise AssemblyError('Unknown attention mechanism.', id)
                kv = heads if mechanism in {'mha', 'mla'} else 1 if mechanism == 'mqa' else params.get('kv_heads', 2)
                positional = params.get('position', 'rope')
                if positional not in {'rope', 'alibi', 'none'}:
                    raise AssemblyError('Choose RoPE, ALiBi or None inside attention.', id,
                                        'Learned and sinusoidal positions are separate additive layers after Token Embedding.')
                if (type(heads) is not int or type(kv) is not int or heads < 1 or kv < 1
                        or size % heads or heads % kv or (positional == 'rope' and (size // heads) % 2)):
                    raise AssemblyError('Incompatible attention head dimensions.', id,
                                        'd_model must divide evenly across query heads; GQA query heads must divide across K/V heads. RoPE needs an even head dimension.')
                if mechanism != 'gqa' and 'kv_heads' in params:
                    raise AssemblyError('K/V head count is only configurable for GQA.', id,
                                        'MHA and MLA use one K/V head per query head; MQA uses one K/V head.')
                if mechanism == 'mla':
                    if type(params.get('latent', max(1, size // 4))) is not int or not 1 <= params.get('latent', max(1, size // 4)) <= 8192:
                        raise AssemblyError('Latent KV dimension must be between 1 and 8192.', id)
                elif 'latent' in params:
                    raise AssemblyError('Latent dimension applies only to MLA.', id)
                if params.get('rope_style', 'legacy') not in {'legacy', 'standard'}:
                    raise AssemblyError('RoPE frequency style must be legacy or standard.', id)
                if 'rope_style' in params and positional != 'rope':
                    raise AssemblyError('RoPE frequency style requires RoPE.', id)
                if positional == 'rope':
                    if not isinstance(params.get('rope_fraction', 0.5), (int, float)) or not 0 < params.get('rope_fraction', 0.5) <= 1:
                        raise AssemblyError('RoPE fraction must be greater than 0 and at most 1.', id)
                    if not isinstance(params.get('rope_base', 10000), (int, float)) or not math.isfinite(params.get('rope_base', 10000)) or params.get('rope_base', 10000) <= 1:
                        raise AssemblyError('RoPE base must be finite and greater than 1.', id)
                elif {'rope_fraction', 'rope_base'} & set(params):
                    raise AssemblyError('RoPE parameters require RoPE positional encoding.', id)
                if type(params.get('window', 0)) is not int or not 0 <= params.get('window', 0) <= 32768:
                    raise AssemblyError('Sliding window must be 0 (full causal) or a positive integer up to 32768.', id)
                if any(type(params.get(flag, default)) is not bool for flag, default in [('qk_norm', True), ('gate', False), ('qkv_bias', False)]):
                    raise AssemblyError('Attention normalization and gating must be boolean.', id)
            if kind in {'rmsnorm', 'layernorm'} and (not isinstance(params.get('eps', 1e-6), (int, float)) or not 0 < params.get('eps', 1e-6) < 1):
                raise AssemblyError('Epsilon must be positive and less than 1.', id)
            if kind == 'position':
                if params.get('encoding', 'sinusoidal') not in {'sinusoidal', 'learned'}:
                    raise AssemblyError('This layer adds sinusoidal or learned positional embeddings.', id,
                                        'RoPE rotates Q/K and ALiBi biases attention scores: configure them inside Self-Attention.')
                if 'base' in params and (params.get('encoding', 'sinusoidal') != 'sinusoidal' or not isinstance(params['base'], (int, float)) or not math.isfinite(params['base']) or params['base'] <= 1):
                    raise AssemblyError('A finite base greater than 1 applies only to sinusoidal encoding.', id)
            if kind == 'swiglu':
                if type(params.get('hidden', 4 * size)) is not int or not 1 <= params.get('hidden', 4 * size) <= 65536:
                    raise AssemblyError('FFN hidden dimension must be an integer between 1 and 65536.', id)
                if params.get('activation', 'swiglu') not in {'swiglu', 'geglu', 'geglu_tanh', 'reglu', 'gelu', 'relu', 'relu2', 'silu', 'leaky_relu'}:
                    raise AssemblyError('Unknown FFN activation.', id)
                if type(params.get('bias', False)) is not bool:
                    raise AssemblyError('FFN bias must be boolean.', id)
            if kind == 'repeat':
                count = params.get('count', 1)
                if type(count) is not int or not 1 <= count <= 64:
                    raise AssemblyError('A block must repeat between 1 and 64 times.', id)
                try:
                    child = validate_assembly(node.get('body'), size, vocab_size, True, depth + 1)
                except AssemblyError as error:
                    error.path = [id] + getattr(error, 'path', [])
                    raise
                expanded += count * child['expanded']
                if expanded > 2048:
                    raise AssemblyError('This model exceeds 2,048 operations after expanding repeated blocks.', id, 'Reduce the number of repetitions before testing.')
                if child['shapes'][child['output']] != ('hidden', size):
                    raise AssemblyError('A repeated block must return the same dimension as its input.', id)
            shapes[id] = shape
            order.append(id)
            pending.remove(id)
    ancestors, todo = set(), [ends[0]]
    while todo:
        id = todo.pop()
        if id not in ancestors:
            ancestors.add(id)
            todo.extend(incoming[id].values())
    unused = set(index) - ancestors
    if unused:
        raise AssemblyError('This layer does not contribute to the model output.', sorted(unused)[0], 'Connect it to the output path or remove it.')
    if not body and shapes[incoming[ends[0]]['in']] != ('hidden', width):
        raise AssemblyError('The output must receive the model dimension (d_model).', ends[0])
    return {'order': order, 'shapes': shapes, 'incoming': incoming, 'output': ends[0], 'input': starts[0], 'expanded': expanded}


class AssemblyAttention(nn.Module):
    def __init__(self, width, heads, kv_heads, max_seq_len, fraction, params=None):
        super().__init__()
        self.heads, self.kv_heads, self.dk = heads, kv_heads, width // heads
        p = params or {}
        self.backend = p.get('backend', 'auto')
        self.position = p.get('position', 'rope')
        self.window = p.get('window', 0)
        self.base = p.get('rope_base', 10000)
        self.latent = p.get('mechanism', 'gqa') == 'mla'
        self.rope_style = p.get('rope_style', 'legacy')
        self.q = nn.Linear(width, width, bias=p.get('qkv_bias', False))
        source_width = p.get('latent', max(1, width // 4)) if self.latent else width
        if self.latent: self.kv_down = nn.Linear(width, source_width, bias=False)
        self.k = nn.Linear(source_width, kv_heads * self.dk, bias=p.get('qkv_bias', False))
        self.v = nn.Linear(source_width, kv_heads * self.dk, bias=p.get('qkv_bias', False))
        self.gate = nn.Linear(width, width, bias=False) if p.get('gate', False) else None
        self.out = nn.Linear(width, width, bias=False)
        self.q_norm = nn.RMSNorm(self.dk, eps=1e-6) if p.get('qk_norm', True) else nn.Identity()
        self.k_norm = nn.RMSNorm(self.dk, eps=1e-6) if p.get('qk_norm', True) else nn.Identity()
        self.max_seq_len, self.fraction = max_seq_len, fraction
        self.register_buffer('cos', None, persistent=False)
        self.register_buffer('sin', None, persistent=False)

    def build_tables(self, device):
        pairs = self.dk // 2
        active = min(pairs, max(1, round(pairs * self.fraction)))
        spectrum = torch.arange(active, device='cpu', dtype=torch.float32) / active if self.rope_style == 'standard' else torch.linspace(0, 1, active, device='cpu', dtype=torch.float32)
        freq = (1 / self.base) ** spectrum
        freq = torch.cat([freq, freq.new_zeros(pairs - active)])
        angle = torch.arange(self.max_seq_len, device='cpu', dtype=torch.float32)[:, None] * freq
        self.cos, self.sin = angle.cos().to(device), angle.sin().to(device)

    def rotate(self, x):
        if self.cos is None or self.cos.device != x.device:
            self.build_tables(x.device)
        a, b = x.float().chunk(2, dim=-1)
        cos, sin = self.cos[None, None, :x.shape[2]], self.sin[None, None, :x.shape[2]]
        if self.rope_style == 'standard':
            return torch.cat([a * cos - b * sin, b * cos + a * sin], -1).to(x.dtype)
        return torch.cat([a * cos + b * sin, b * cos - a * sin], -1).to(x.dtype)

    def forward(self, x, doc_ids=None):
        B, T, _ = x.shape
        latent = self.kv_down(x) if self.latent else x
        q = self.q_norm(self.q(x).view(B, T, self.heads, self.dk)).transpose(1, 2)
        k = self.k_norm(self.k(latent).view(B, T, self.kv_heads, self.dk)).transpose(1, 2)
        v = self.v(latent).view(B, T, self.kv_heads, self.dk).transpose(1, 2)
        if self.position == 'rope': q, k = self.rotate(q), self.rotate(k)
        k, v = (t.repeat_interleave(self.heads // self.kv_heads, dim=1) for t in (k, v))
        mask = None
        if doc_ids is not None or self.window or self.position == 'alibi':
            positions = torch.arange(T, device=x.device)
            distance = positions[:, None] - positions[None, :]
            mask = (distance >= 0)[None, None]
            if self.window: mask = mask & (distance < self.window)[None, None]
            if doc_ids is not None: mask = mask & (doc_ids[:, None, :, None] == doc_ids[:, None, None, :])
            if self.position == 'alibi':
                slopes = torch.tensor(alibi_slopes(self.heads), device=x.device, dtype=torch.float32)
                bias = -slopes[None, :, None, None] * distance[None, None].clamp_min(0)
                mask = bias.masked_fill(~mask, float('-inf')).to(q.dtype)
        from torch.nn.attention import sdpa_kernel, SDPBackend
        with sdpa_kernel(SDPBackend.MATH) if self.backend == 'math' else nullcontext():
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=mask is None)
        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        if self.gate is not None: out = out * torch.sigmoid(self.gate(x))
        return self.out(out)


def alibi_slopes(heads):
    def power2(n):
        start = 2 ** (-(2 ** -(math.log2(n) - 3)))
        return [start ** (i + 1) for i in range(n)]
    closest = 2 ** math.floor(math.log2(heads))
    return power2(closest) + power2(2 * closest)[::2][:heads - closest]


class UnitOffsetRMSNorm(nn.RMSNorm):
    """Gemma-style RMSNorm, preserving the norm module type for optimizers."""
    def __init__(self, width, eps):
        super().__init__(width, eps=eps)
        nn.init.zeros_(self.weight)

    def forward(self, x):
        normalized = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + self.eps)
        return (normalized * (1 + self.weight.float())).to(x.dtype)


class AssemblyPosition(nn.Module):
    def __init__(self, width, context, encoding, base=10000):
        super().__init__()
        self.encoding, self.width, self.base = encoding, width, base
        if encoding == 'learned': self.positions = nn.Embedding(context, width)

    def forward(self, x):
        positions = torch.arange(x.shape[1], device=x.device)
        if self.encoding == 'learned': return x + self.positions(positions).to(x.dtype)
        frequencies = self.base ** (-torch.arange(0, self.width, 2, device=x.device, dtype=torch.float32) / self.width)
        angle = positions[:, None].float() * frequencies[None, :]
        encoding = torch.stack((angle.sin(), angle.cos()), dim=-1).flatten(-2)[:, :self.width]
        return x + encoding.to(x.dtype)


class AssemblyFFN(nn.Module):
    def __init__(self, width, hidden, activation='swiglu', bias=False):
        super().__init__()
        self.activation = activation
        self.gated = activation in {'swiglu', 'geglu', 'geglu_tanh', 'reglu'}
        if self.gated: self.gate = nn.Linear(width, hidden, bias=bias)
        self.up, self.down = nn.Linear(width, hidden, bias=bias), nn.Linear(hidden, width, bias=bias)

    def forward(self, x):
        activations = {'swiglu': F.silu, 'geglu': F.gelu, 'geglu_tanh': lambda t: F.gelu(t, approximate='tanh'), 'reglu': F.relu,
                       'gelu': F.gelu, 'relu': F.relu, 'relu2': lambda t: F.relu(t).square(),
                       'silu': F.silu, 'leaky_relu': F.leaky_relu}
        fn = activations[self.activation]
        hidden = fn(self.gate(x)) * self.up(x) if self.gated else fn(self.up(x))
        return self.down(hidden)


class AssemblyRuntime(nn.Module):
    def __init__(self, graph, width, vocab_size, max_seq_len, tie_embeddings=False, body=False):
        super().__init__()
        self.spec, self.width, self.max_seq_len = graph, width, max_seq_len
        self.plan = validate_assembly(graph, width, vocab_size, body)
        self.bricks = nn.ModuleDict()
        self.aux_loss = None
        from .assembly_schema import MODERN_BRICKS
        self.modern_kinds = set(MODERN_BRICKS)
        for node in graph['nodes']:
            id, kind, p = node['id'], node['kind'], node.get('params', {})
            size = p.get('width', width)
            if kind in self.modern_kinds:
                from .modern_bricks import build_modern
                self.bricks[id] = build_modern(kind,p,width,vocab_size,max_seq_len)
            elif kind == 'embedding': self.bricks[id] = nn.Embedding(vocab_size, size)
            elif kind == 'head': self.bricks[id] = nn.Linear(size, vocab_size, bias=False)
            elif kind == 'rmsnorm': self.bricks[id] = (UnitOffsetRMSNorm if p.get('unit_offset', False) else nn.RMSNorm)(size, eps=p.get('eps', 1e-6))
            elif kind == 'layernorm': self.bricks[id] = nn.LayerNorm(size, eps=p.get('eps', 1e-6))
            elif kind == 'position': self.bricks[id] = AssemblyPosition(size, max_seq_len, p.get('encoding', 'sinusoidal'), p.get('base', 10000))
            elif kind == 'attention':
                heads = p.get('heads', 4)
                mechanism = p.get('mechanism', 'gqa')
                kv = heads if mechanism in {'mha', 'mla'} else 1 if mechanism == 'mqa' else p.get('kv_heads', 2)
                self.bricks[id] = AssemblyAttention(size, heads, kv, max_seq_len, p.get('rope_fraction', 0.5), p)
            elif kind == 'swiglu': self.bricks[id] = AssemblyFFN(size, p.get('hidden', 4 * size), p.get('activation', 'swiglu'), p.get('bias', False))
            elif kind == 'repeat': self.bricks[id] = nn.ModuleList([AssemblyRuntime(node['body'], size, vocab_size, max_seq_len, body=True) for _ in range(p.get('count', 1))])
        if tie_embeddings and not body:
            embedding = next(self.bricks[n['id']] for n in graph['nodes'] if n['kind'] == 'embedding')
            head = self.bricks[self.plan['output']]
            if head.weight.shape != embedding.weight.shape:
                raise AssemblyError('LM Head and Token Embedding must have the same dimension to tie their weights.', self.plan['output'])
            head.weight = embedding.weight

    def forward(self, x, doc_ids=None, return_hidden=False, ports=None):
        from .assembly_schema import INPUT_KINDS, OUTPUT_KINDS
        ports = ports or {}
        values, losses = {}, []
        handles = self.plan.get('incoming_handles', {})
        def source_value(target, port):
            source = self.plan['incoming'][target][port]
            value = values[source]
            handle = handles.get(target, {}).get(port, 'out')
            return value[handle] if isinstance(value, dict) else value
        for id in self.plan['order']:
            node = next(n for n in self.spec['nodes'] if n['id'] == id)
            kind, sources = node['kind'], self.plan['incoming'][id]
            if kind in INPUT_KINDS:
                name = INPUT_KINDS[kind]
                value = x if name == 'in' else ports[name]
            elif not sources: value = x
            elif kind == 'add': value = source_value(id,'a') + source_value(id,'b')
            else:
                value = source_value(id,'in')
                extra = {p:source_value(id,p) for p in sources if p != 'in'}
                if kind == 'repeat':
                    for block in self.bricks[id]:
                        value = block(value['out'] if isinstance(value,dict) else value, doc_ids, ports=extra)
                        if block.aux_loss is not None: losses.append(block.aux_loss)
                elif kind == 'embedding': value = self.bricks[id](value) * (1 if node.get('params', {}).get('scale') == 'none' else math.sqrt(self.bricks[id].embedding_dim))
                elif kind == 'attention': value = self.bricks[id](value, doc_ids)
                elif kind in OUTPUT_KINDS: pass
                elif kind == 'head' and return_hidden: pass
                elif kind in self.modern_kinds:
                    module = self.bricks[id]
                    value = module(value) if kind in {'gemma4_norm','qwen_norm'} else module(value,extra,doc_ids)
                    aux = getattr(module,'aux_loss',None)
                    if aux is not None: losses.append(aux)
                else:
                    value = self.bricks[id](value)
                    if kind == 'head' and node.get('params',{}).get('softcap',0):
                        cap = node['params']['softcap']; value = torch.tanh(value/cap)*cap
            values[id] = value
        self.aux_loss = torch.stack(losses).sum() if losses else None
        output = values[self.plan['output']]
        extras = {OUTPUT_KINDS[n['kind']]:values[n['id']] for n in self.spec['nodes'] if n['kind'] in OUTPUT_KINDS and n['kind']!='output_hidden'}
        return {'out':output, **extras} if extras else output
