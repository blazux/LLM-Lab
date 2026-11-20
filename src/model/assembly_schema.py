"""Typed multi-port graphs and the modern brick catalog (no tensor allocation)."""


def number(default, low=1, high=8192, step=1):
    return dict(type='number', default=default, min=low, max=high, step=step)


def choice(default, *values):
    return dict(type='select', default=default, options=list(values))


def flag(default=False):
    return dict(type='boolean', default=default)


MODERN_BRICKS = {
    'modern_attention': dict(title='Modern Self-Attention', help='Reference attention from Gemma 4, Llama 4 or Qwen. Head dimension is independent of model width. K/V sharing is an explicit input connection.',
        inputs=['in'], outputs=['out', 'kv'], params=dict(
            family=choice('qwen35', 'qwen35', 'qwen4', 'gemma4', 'llama4'), width=number(64), heads=number(4, high=256), kv_heads=number(1, high=256), head_dim=number(16, high=512, step=2),
            position=choice('rope', 'rope', 'none'), rope_base=number(10000000, 2, 1000000000), rotary_fraction=number(.25, .01, 1, .01),
            window=number(0, 0, 32768), qk_norm=flag(True), temperature=flag(False), temperature_scale=number(.1, 0, 10, .01), temperature_floor=number(8192, 1, 10000000),
            k_eq_v=flag(False), kv_shared=flag(False), eps=number(1e-6, 1e-12, .1, .000001),
            index_heads=number(2, high=64), index_dim=number(16, high=512, step=2), budget=number(32, 1, 32768), compression=number(4, 1, 128), index_loss=number(.01, 0, 1, .001))),
    'gated_deltanet': dict(title='Gated DeltaNet', help='Causal depthwise convolution, separate Q/K and V heads, gated delta-rule memory and gated RMSNorm. Uses the reference Qwen implementation; no RoPE is applied here.', inputs=['in'], outputs=['out'], params=dict(
        family=choice('qwen35','qwen35','qwen4'), width=number(64), key_heads=number(2, high=128), value_heads=number(4, high=128), key_dim=number(16, high=512), value_dim=number(16, high=512), conv_kernel=number(4, high=32), eps=number(1e-6, 1e-12,.1,.000001))),
    'moe': dict(title='Mixture of Experts', help='Token router and routed FFNs. Llama and Qwen include a shared expert; Gemma uses a separately connected dense FFN branch. Router balance loss is returned to the trainer.', inputs=['in'], outputs=['out'], params=dict(
        family=choice('qwen35','qwen35','qwen4','llama4','gemma4'), width=number(64), experts=number(8, 2, 512), top_k=number(2, 1, 16), hidden=number(32, 1, 65536), shared_hidden=number(32, 1, 65536), aux_weight=number(.001,0,1,.001), eps=number(1e-6,1e-12,.1,.000001))),
    'gemma4_norm': dict(title='Gemma 4 RMSNorm', help='Gemma 4 scale convention and optional parameter-free RMS normalization. This is distinct from Gemma 1 unit-offset normalization.', inputs=['in'], outputs=['out'], params=dict(width=number(64),eps=number(1e-6,1e-12,.1,.000001), learned_scale=flag(True))),
    'qwen_norm': dict(title='Qwen RMSNorm', help='Unit-offset RMSNorm used by the Qwen 3.5 and Qwen4 text backbones.', inputs=['in'],outputs=['out'],params=dict(width=number(64),eps=number(1e-6,1e-12,.1,.000001))),
    'ple_embedding': dict(title='Per-Layer Embeddings', help='Gemma 4 token-identity lookup plus normalized contextual projection, producing a separate input for every decoder layer.',inputs=['in','tokens'],outputs=['out'],params=dict(width=number(64),dim=number(16,high=2048),layers=number(5,high=64),eps=number(1e-6,1e-12,.1,.000001))),
    'ple_apply': dict(title='PLE Gated Injection',help='Selects one layer’s PLE features, gates them from hidden states, projects and normalizes them, then adds the residual.',inputs=['in','ple'],outputs=['out'],params=dict(width=number(64),dim=number(16,high=2048),index=number(0,0,63),eps=number(1e-6,1e-12,.1,.000001))),
    'residual_expand': dict(title='Expand Residual Streams',help='Replicates the residual stream into parallel branches for Qwen4 Gated Residual.',inputs=['in'],outputs=['out'],params=dict(width=number(64),streams=number(4,1,8))),
    'gr_read': dict(title='Gated Residual Read',help='Learned read gates mix parallel residual streams. The state output carries the residual and write gates to Gated Residual Write.',inputs=['in'],outputs=['out','state'],params=dict(width=number(64),streams=number(4,1,8),rank=number(16,high=2048),eps=number(1e-6,1e-12,.1,.000001))),
    'gr_write': dict(title='Gated Residual Write',help='Injects the sublayer output into each residual branch using the read node’s write gates.',inputs=['in','state'],outputs=['out'],params=dict(width=number(64),streams=number(4,1,8))),
    'gr_collapse': dict(title='Combine Residual Streams',help='Final learned read of the parallel residual branches before final normalization and LM Head.',inputs=['in'],outputs=['out'],params=dict(width=number(64),streams=number(4,1,8),rank=number(16,high=2048),eps=number(1e-6,1e-12,.1,.000001))),
    'ngram': dict(title='N-gram Embedding',help='Qwen4 hashed local-context embeddings, stream-dependent gates and causal dilated convolution. This is distinct from Gemma PLE. Returns an injection to add to the residual streams.',inputs=['in','tokens'],outputs=['out'],params=dict(width=number(64),streams=number(4,1,8),dim=number(64,high=8192),ngram_size=number(3,2,4),heads_per_ngram=number(2,1,16),buckets=number(257,2,20000000),conv_kernel=number(4,1,16),eos_id=number(0,0,1000000),seed=number(1234,0,2147483647),eps=number(1e-6,1e-12,.1,.000001))),
    'scale': dict(title='Residual Scale',help='Multiplies the residual stream by a fixed scalar.',inputs=['in'],outputs=['out'],params=dict(width=number(64),factor=number(1,-100,100,.01))),
}
INPUT_KINDS = {'input_hidden':'in','input_tokens':'tokens','input_kv':'kv','input_ple':'ple'}
OUTPUT_KINDS = {'output_hidden':'out','output_kv':'kv'}


def defaults(kind, width=64):
    p = {k: v['default'] for k,v in MODERN_BRICKS[kind]['params'].items()}
    if 'width' in p: p['width'] = width
    return p


def input_ports(node):
    kind, p = node['kind'], node.get('params', {})
    if kind in INPUT_KINDS or kind == 'tokens': return []
    if kind == 'add': return ['a','b']
    if kind == 'repeat':
        return [INPUT_KINDS[n['kind']] for n in node['body']['nodes'] if n['kind'] in INPUT_KINDS]
    if kind in MODERN_BRICKS:
        ports = list(MODERN_BRICKS[kind]['inputs'])
        if kind == 'modern_attention' and p.get('kv_shared'): ports.append('kv')
        if kind == 'moe' and p.get('family','qwen35') == 'gemma4': ports.append('router')
        return ports
    return ['in']


def validate_modern_params(node):
    from .assembly_runtime import AssemblyError
    kind, p, id = node['kind'], node.get('params', {}), node['id']
    specs = MODERN_BRICKS[kind]['params']
    if set(p) - set(specs): raise AssemblyError('Unknown layer parameters: ' + ', '.join(sorted(set(p)-set(specs))),id)
    for key,spec in specs.items():
        value = p.get(key,spec['default'])
        valid = (type(value) is bool if spec['type']=='boolean' else value in spec['options'] if spec['type']=='select' else
                 type(value) in (int,float) and spec['min'] <= value <= spec['max'] and (spec['step'] != 1 or type(value) is int))
        if not valid: raise AssemblyError(f'Invalid {key}: {value}.',id)
    q = {**defaults(kind), **p}
    if kind == 'modern_attention':
        if q['heads'] % q['kv_heads']: raise AssemblyError('Query heads must be divisible by KV heads.',id)
        rotary = int(q['head_dim'] * q['rotary_fraction'])
        if q['position']=='rope' and (rotary < 2 or rotary % 2): raise AssemblyError('The rotary head dimension must be positive and even.',id)
        if q['family']=='qwen4' and (q['compression']>q['budget'] or q['index_dim']<rotary): raise AssemblyError('QSA needs budget ≥ compression and index dimension ≥ rotary dimension.',id)
        if q['family']!='gemma4' and (q['kv_shared'] or q['k_eq_v']): raise AssemblyError('KV sharing and K=V belong to Gemma 4 attention.',id)
        if q['family'] in {'qwen35','qwen4'} and q['position']!='rope': raise AssemblyError('Qwen reference attention requires rotary positions.',id)
        if q['family'] in {'llama4','gemma4'} and q['rotary_fraction']!=1 and q['family']=='llama4': raise AssemblyError('Llama 4 uses full-head RoPE.',id)
    if kind == 'modern_attention':
        exclusive={'kv_shared':'gemma4','k_eq_v':'gemma4','temperature':'llama4','temperature_scale':'llama4','temperature_floor':'llama4',
                   'index_heads':'qwen4','index_dim':'qwen4','budget':'qwen4','compression':'qwen4','index_loss':'qwen4'}
        for key,family in exclusive.items():
            if key in p and q['family']!=family and p[key]!=specs[key]['default']:
                raise AssemblyError(f'{key} only applies to {family}.',id)
        if q['family']=='gemma4' and q['k_eq_v'] and q['window']:
            raise AssemblyError('K=V is a global-attention option; disable the local window.',id)
    if kind == 'gated_deltanet' and q['value_heads'] % q['key_heads']: raise AssemblyError('Value heads must be a multiple of Q/K heads.',id)
    if kind == 'moe' and q['top_k'] > q['experts']: raise AssemblyError('Active experts cannot exceed the expert count.',id)
    if kind == 'moe' and q['family']=='gemma4' and 'shared_hidden' in p and p['shared_hidden']!=specs['shared_hidden']['default']:
        raise AssemblyError('Gemma 4 uses an explicitly connected dense branch; shared_hidden does not apply.',id)
    if kind == 'ngram' and q['dim'] % ((q['ngram_size']-1)*q['heads_per_ngram']): raise AssemblyError('N-gram embedding dimension must divide evenly over its hash heads.',id)


def validate_graph(graph, width, vocab, body=False, depth=0):
    from .assembly_runtime import AssemblyError, _validate_legacy
    import re
    def fail(message,id=None,hint=''): raise AssemblyError(message,id,hint)
    if depth>4 or not isinstance(graph,dict) or graph.get('version') not in (1,2): fail('Invalid graph version or nesting depth.')
    if set(graph)-{'version','nodes','edges'}: fail('Unknown graph properties.')
    nodes,edges=graph.get('nodes'),graph.get('edges')
    if not isinstance(nodes,list) or not 1<=len(nodes)<=1024 or not isinstance(edges,list) or len(edges)>4096: fail('Use at most 1,024 layers and 4,096 connections per graph.')
    index, incoming, handles = {},{},{}
    legacy={'tokens','embedding','rmsnorm','layernorm','position','attention','swiglu','add','head','repeat'}
    for n in nodes:
        if not isinstance(n,dict): fail('Each layer must be an object.')
        id=n.get('id'); kind=n.get('kind')
        if not isinstance(id,str) or not re.fullmatch(r'[A-Za-z][A-Za-z0-9_-]{0,63}',id) or id in index: fail('Layer IDs must be unique and simple.',str(id))
        if set(n)-{'id','kind','params','position','label','body'} or kind not in legacy|set(MODERN_BRICKS)|set(INPUT_KINDS)|set(OUTPUT_KINDS): fail('Unknown layer properties or kind.',id)
        if not isinstance(n.get('params',{}),dict): fail('Parameters must be an object.',id)
        if kind!='repeat' and 'body' in n: fail('Only a block can contain a subgraph.',id)
        index[id]=n; incoming[id]={}; handles[id]={}
    for e in edges:
        if not isinstance(e,dict) or set(e)-{'id','source','target','sourceHandle','targetHandle'}: fail('Unknown connection properties.')
        source,target,port=e.get('source'),e.get('target'),e.get('targetHandle','in')
        if source not in index or target not in index: fail('Connection refers to a missing layer.',target)
        if port in incoming[target]: fail('An input port accepts only one connection.',target)
        incoming[target][port]=source; handles[target][port]=e.get('sourceHandle') or 'out'
    starts=[n for n in nodes if n['kind']==('input_hidden' if body else 'tokens')]
    ends=[n for n in nodes if n['kind']==('output_hidden' if body else 'head')]
    if len(starts)!=1 or len(ends)!=1: fail('A graph needs one main input and one main output.')
    if not body and (any(n['kind'] in set(INPUT_KINDS)|set(OUTPUT_KINDS) for n in nodes) or sum(n['kind']=='embedding' for n in nodes)!=1): fail('Root graph needs one Token Embedding and no block input/output layers.')
    if body and any(n['kind'] in {'tokens','head','embedding'} for n in nodes): fail('Token Embedding and LM Head belong at root level.')
    for kind in INPUT_KINDS|OUTPUT_KINDS:
        if sum(n['kind']==kind for n in nodes)>1: fail('Duplicate block boundary port: '+kind)
    order,shapes,outputs,children=[],{},{},{}
    pending=set(index); expanded=len(nodes)
    while pending:
        ready=[n for n in nodes if n['id'] in pending and all(s in outputs for s in incoming[n['id']].values())]
        if not ready: fail('The graph contains a cycle.',next(iter(pending)))
        for n in ready:
            id,kind,p=n['id'],n['kind'],n.get('params',{})
            size=p.get('width',width)
            if type(size) is not int or not 1<=size<=32768: fail('Invalid layer dimension.',id)
            if kind=='repeat':
                if set(p)-{'width','count'} or type(p.get('count',1)) is not int or not 1<=p.get('count',1)<=64: fail('Invalid repeated block parameters.',id)
                try: child=validate_graph(n.get('body'),size,vocab,True,depth+1)
                except AssemblyError as err: err.path=[id]+getattr(err,'path',[]); raise
                children[id]=child; expanded+=p.get('count',1)*child['expanded']
            try: expected=set(input_ports(n))
            except (KeyError,TypeError): fail('Invalid block inputs.',id)
            if set(incoming[id])!=expected: fail('Connect every required input: '+', '.join(sorted(expected)),id)
            actual={}
            for port,src in incoming[id].items():
                handle=handles[id][port]
                if handle not in outputs[src]: fail('The source layer has no output port '+handle,id)
                actual[port]=outputs[src][handle]
            out={}
            if kind in MODERN_BRICKS:
                validate_modern_params(n); q={**defaults(kind,width),**p}
                def expect(port,shape):
                    if actual.get(port)!=shape: fail(f'{port} expects {shape}, received {actual.get(port)}.',id)
                if kind=='ple_embedding':
                    expect('in',('hidden',size)); expect('tokens',('tokens',None)); shape=('ple',q['layers']*q['dim'])
                elif kind=='ple_apply':
                    expect('in',('hidden',size))
                    ple=actual.get('ple')
                    if not ple or ple[0]!='ple' or ple[1]%q['dim'] or q['index']>=ple[1]//q['dim']: fail('PLE layer index or dimension is incompatible with its input.',id)
                    shape=('hidden',size)
                elif kind in {'gr_read','gr_collapse','ngram'}:
                    expect('in',('hidden',size*q['streams']))
                    if kind=='ngram': expect('tokens',('tokens',None))
                    shape=('hidden',size*q['streams'] if kind=='ngram' else size)
                    if kind=='gr_read': out['state']=('residual_state',size*q['streams'])
                elif kind=='gr_write':
                    expect('in',('hidden',size)); expect('state',('residual_state',size*q['streams'])); shape=('hidden',size*q['streams'])
                else:
                    expect('in',('hidden',size)); shape=('hidden',size*q['streams'] if kind=='residual_expand' else size)
                if kind=='moe' and q['family']=='gemma4': expect('router',('hidden',size))
                if kind=='modern_attention' and q['family']=='gemma4':
                    out['kv']=('kv',q['kv_heads']*q['head_dim'])
                    if q['kv_shared']: expect('kv',out['kv'])
            elif kind=='tokens' or kind=='input_tokens':
                if p: fail('Token input ports have no parameters.',id)
                shape=('tokens',None)
            elif kind=='input_hidden':
                if p: fail('Main block input dimension is set by its container.',id)
                shape=('hidden',width)
            elif kind in {'input_kv','input_ple'}:
                if set(p)-{'width'}: fail('Boundary ports only accept a dimension.',id)
                shape=('kv' if kind=='input_kv' else 'ple',size)
            elif kind=='repeat':
                child=children[id]
                for entry in n['body']['nodes']:
                    if entry['kind'] in INPUT_KINDS and actual[INPUT_KINDS[entry['kind']]]!=child['shapes'][entry['id']]: fail('Block input dimensions do not match.',id)
                shape=child['shapes'][child['output']]
                if shape!=('hidden',size): fail('Repeated blocks must preserve their main dimension.',id)
                for exit in n['body']['nodes']:
                    if exit['kind']=='output_kv': out['kv']=child['shapes'][exit['id']]
            elif kind=='output_kv':
                if p: fail('Output boundary ports have no parameters.',id)
                if actual['in'][0]!='kv': fail('KV output expects key/value states.',id)
                shape=actual['in']
            elif kind=='output_hidden':
                if p: fail('Output boundary ports have no parameters.',id)
                shape=actual['in']
                if shape!=('hidden',width): fail('Block output dimension differs from its input.',id)
            elif kind=='add':
                if p: fail('Residual Add has no parameters.',id)
                if actual['a']!=actual['b'] or actual['a'][0]!='hidden': fail('Addition needs hidden states of the same dimension.',id)
                shape=actual['a']
            else:
                # Reuse the strict v1 parameter contract for legacy operations.
                synthetic={'version':1,'nodes':[{'id':'tokens','kind':'tokens'},{'id':'emb','kind':'embedding'}, {'id':'head','kind':'head'}],
                           'edges':[{'source':'tokens','target':'emb'},{'source':'emb','target':'head'}]}
                if kind=='embedding':
                    if set(p)-{'width','scale'} or p.get('scale','sqrt_dim') not in {'sqrt_dim','none'} or actual['in']!=('tokens',None): fail('Invalid Token Embedding parameters or input.',id)
                    shape=('hidden',size)
                else:
                    if actual['in']!=('hidden',size): fail('Incoming hidden dimension does not match this layer.',id)
                    shape=('logits',vocab) if kind=='head' else ('hidden',size)
                    if kind=='head':
                        if set(p)-{'width','softcap'} or not isinstance(p.get('softcap',0),(int,float)) or not 0<=p.get('softcap',0)<=1000: fail('Invalid LM Head softcap.',id)
                    else:
                        # One standalone legacy layer, validated without instantiating any modules.
                        synthetic['nodes'].insert(2,{**n,'id':'op'})
                        synthetic['nodes'][1]['params']={'width':size}
                        synthetic['nodes'][-1]['params']={'width':size}
                        synthetic['edges']=[{'source':'tokens','target':'emb'},{'source':'emb','target':'op'},{'source':'op','target':'head'}]
                        try: _validate_legacy(synthetic,size,vocab)
                        except AssemblyError as err: err.node_id=id; raise
            out['out']=shape; outputs[id]=out; shapes[id]=shape; order.append(id); pending.remove(id)
    if expanded>16384: fail('Too many expanded operations.')
    destinations=[n['id'] for n in ends]+[n['id'] for n in nodes if n['kind']=='output_kv']
    used=set(); todo=destinations[:]
    while todo:
        id=todo.pop()
        if id not in used: used.add(id); todo.extend(incoming[id].values())
    if set(index)-used: fail('This layer does not contribute to an output.',sorted(set(index)-used)[0])
    return dict(order=order,shapes=shapes,outputs=outputs,incoming=incoming,incoming_handles=handles,input=starts[0]['id'],output=ends[0]['id'],expanded=expanded)
