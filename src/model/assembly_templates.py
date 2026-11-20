"""Small, explicit examples that can be dismantled in the workshop."""
def transformer_body(width=64, heads=4, kv_heads=2, hidden=128):
    kinds = [('entry', 'input_hidden'), ('norm1', 'rmsnorm'), ('attn', 'attention'), ('add1', 'add'),
             ('norm2', 'rmsnorm'), ('ffn', 'swiglu'), ('add2', 'add'), ('exit', 'output_hidden')]
    nodes = [{'id': id, 'kind': kind, 'position': {'x': 160, 'y': (len(kinds) - 1 - i) * 145}, 'params': {}} for i, (id, kind) in enumerate(kinds)]
    nodes[2]['params'] = {'heads': heads, 'kv_heads': kv_heads, 'rope_fraction': 0.5}
    nodes[5]['params'] = {'hidden': hidden}
    links = [('entry', 'norm1', 'in'), ('norm1', 'attn', 'in'), ('attn', 'add1', 'a'), ('entry', 'add1', 'b'),
             ('add1', 'norm2', 'in'), ('norm2', 'ffn', 'in'), ('ffn', 'add2', 'a'), ('add1', 'add2', 'b'), ('add2', 'exit', 'in')]
    return {'version': 1, 'nodes': nodes, 'edges': [{'id': f'e{i}', 'source': s, 'target': t, 'targetHandle': p} for i, (s,t,p) in enumerate(links)]}


def transformer_assembly(width=64, heads=4, kv_heads=2, hidden=128, layers=2):
    nodes = [{'id': 'tokens', 'kind': 'tokens'}, {'id': 'embedding', 'kind': 'embedding'},
             {'id': 'block', 'kind': 'repeat', 'params': {'count': layers}, 'body': transformer_body(width, heads, kv_heads, hidden)},
             {'id': 'final_norm', 'kind': 'rmsnorm'}, {'id': 'head', 'kind': 'head'}]
    for i, node in enumerate(nodes): node['position'] = {'x': 160, 'y': (len(nodes) - 1 - i) * 145}
    return {'version': 1, 'nodes': nodes, 'edges': [{'id': f'e{i}', 'source': a['id'], 'target': b['id'], 'targetHandle': 'in'} for i,(a,b) in enumerate(zip(nodes,nodes[1:]))]}


# Scaled educational architectures, with explicit reference generations. No weights
# or tokenizer are downloaded, and these are not checkpoint conversion recipes.
FAMILIES = {
    'llama3': {
        'name': 'Llama 3',
        'description': 'Dense pre-norm decoder · GQA · SwiGLU · untied LM Head · RoPE θ=500k',
        'source': 'https://github.com/meta-llama/llama3/blob/main/llama/model.py',
        'ratio': 3.5, 'activation': 'swiglu', 'eps': 1e-5, 'theta': 500000,
        'tie': False, 'unit_offset': False, 'qkv_bias': False,
    },
    'gemma1': {
        'name': 'Gemma 1',
        'description': '2B-style decoder · MQA · tanh-GeGLU · unit-offset RMSNorm · scaled embeddings',
        'source': 'https://github.com/google/gemma_pytorch/blob/main/gemma/model.py',
        'ratio': 8, 'activation': 'geglu_tanh', 'eps': 1e-6, 'theta': 10000,
        'tie': True, 'unit_offset': True, 'qkv_bias': False,
    },
    'qwen25': {
        'name': 'Qwen 2.5',
        'description': '0.5B-style decoder · GQA · SwiGLU · Q/K/V biases · tied LM Head · RoPE θ=1M',
        'source': 'https://huggingface.co/Qwen/Qwen2.5-0.5B/blob/main/config.json',
        'ratio': 38 / 7, 'activation': 'swiglu', 'eps': 1e-6, 'theta': 1000000,
        'tie': True, 'unit_offset': False, 'qkv_bias': True,
    },
}
SIZE_NAMES = ('nano', 'tiny', 'small', 'medium')


def size_defaults(family, size):
    from .modern_templates import MODERN_FAMILIES, modern_size
    if family in MODERN_FAMILIES: return modern_size(family,size)
    if family not in FAMILIES or size not in SIZE_NAMES:
        raise ValueError('Unknown architecture template or size.')
    i = SIZE_NAMES.index(size)
    width = ([112, 224, 448, 896] if family == 'qwen25' else [64, 128, 256, 512])[i]
    heads = ([7, 14, 14, 14] if family == 'qwen25' else [4, 4, 8, 8])[i]
    kv = 1 if family == 'gemma1' else heads // (7 if family == 'qwen25' else 4)
    hidden = round(width * FAMILIES[family]['ratio'] / 8) * 8
    return dict(d_model=width, n_layers=[2, 4, 6, 8][i], n_heads=heads, n_kv_heads=kv,
                d_ff=hidden, max_seq_len=[256, 512, 1024, 2048][i])


def template_catalog():
    from .modern_templates import catalog
    return catalog() + [{'id': id, 'name': f['name'], 'description': f['description'], 'source': f['source'],
             'sizes': {size: size_defaults(id, size) for size in SIZE_NAMES},
             'status': 'Dense stable candidate · scaled educational text architecture'}
            for id, f in FAMILIES.items()]


def family_template(family, size='nano'):
    from .modern_templates import MODERN_FAMILIES, modern_template
    if family in MODERN_FAMILIES: return modern_template(family,size)
    from copy import deepcopy
    if family not in FAMILIES: raise ValueError('Unknown architecture template.')
    f, dimensions = FAMILIES[family], size_defaults(family, size)
    width, heads, kv, hidden, layers = (dimensions[k] for k in ['d_model', 'n_heads', 'n_kv_heads', 'd_ff', 'n_layers'])
    graph = transformer_assembly(width, heads, kv, hidden, layers)
    graph['nodes'][1]['params'] = {'scale': 'sqrt_dim' if family == 'gemma1' else 'none'}
    body = graph['nodes'][2]['body']
    body['nodes'][2]['params'] = {'mechanism': 'mqa' if family == 'gemma1' else 'gqa', 'heads': heads,
        'position': 'rope', 'rope_style': 'standard', 'rope_fraction': 1.0, 'rope_base': f['theta'],
        'qk_norm': False, 'qkv_bias': f['qkv_bias']}
    if family != 'gemma1': body['nodes'][2]['params']['kv_heads'] = kv
    body['nodes'][5]['params'] = {'hidden': hidden, 'activation': f['activation'], 'bias': False}
    for node in [body['nodes'][1], body['nodes'][4], graph['nodes'][3]]:
        node['params'] = {'eps': f['eps'], 'unit_offset': f['unit_offset']}
    return {**deepcopy(dimensions), 'assembly': graph, 'assembly_template': family,
            'dropout': 0.0, 'tie_word_embeddings': f['tie']}


def resize_assembly(config, width, layers, context):
    """Resize the actual graph while retaining its topology and local layer choices.

    Per-node FFN/latent ratios survive; head ratios are kept where dimensions
    permit. Repetition counts inside the main block are preserved.
    """
    from copy import deepcopy
    from model.assembly_runtime import AssemblyError
    if type(width) is not int or not 2 <= width <= 8192 or width % 2:
        raise ValueError('Global model dimension must be an even integer between 2 and 8192.')
    if type(layers) is not int or not 1 <= layers <= 64:
        raise ValueError('Layer count must be an integer between 1 and 64.')
    if type(context) is not int or not 2 <= context <= 32768:
        raise ValueError('Context length must be between 2 and 32768.')
    from .modern_templates import MODERN_FAMILIES, resize_modern
    if config.assembly_template in MODERN_FAMILIES:
        return resize_modern(config,width,layers,context)
    graph = deepcopy(config.assembly)
    repeats = [n for n in graph['nodes'] if n['kind'] == 'repeat']
    if len(repeats) != 1:
        raise AssemblyError('Global depth resizing needs one top-level repeated block.', hint='Edit individual block counts in this custom graph.')
    repeats[0]['params'] = {**repeats[0].get('params', {}), 'count': layers}
    def visit(g):
        for node in g['nodes']:
            p = node.setdefault('params', {})
            old_width = p.get('width', config.d_model)
            if 'width' in p: p['width'] = width
            if node['kind'] == 'swiglu': p['hidden'] = max(8, round(p.get('hidden', 4 * old_width) * width / old_width / 8) * 8)
            if node['kind'] == 'attention':
                old_heads = p.get('heads', 4)
                # Preserve head dimension when possible, otherwise choose the
                # nearest valid head count. RoPE needs an even head dimension.
                wanted = width / (old_width / old_heads)
                candidates = [h for h in range(1, min(width, 256) + 1) if width % h == 0 and (width // h) % 2 == 0]
                heads = min(candidates, key=lambda h: (abs(h - wanted), h))
                p['heads'] = heads
                if p.get('mechanism', 'gqa') == 'gqa':
                    wanted_kv = heads * p.get('kv_heads', 2) / old_heads
                    p['kv_heads'] = min([k for k in range(1, heads + 1) if heads % k == 0], key=lambda k: (abs(k - wanted_kv), k))
                if p.get('mechanism') == 'mla': p['latent'] = max(1, round(p.get('latent', max(1, old_width // 4)) * width / old_width))
            if node.get('body'): visit(node['body'])
    visit(graph)
    result = deepcopy(config)
    result.assembly, result.d_model, result.n_layers, result.max_seq_len = graph, width, layers, context
    result.__post_init__()
    return result
