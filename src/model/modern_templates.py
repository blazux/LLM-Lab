"""Scaled, editable text-backbone graphs for published modern architecture variants."""
from copy import deepcopy

MODERN_FAMILIES = {
 'qwen35_dense': dict(name='Qwen 3.5 · dense text',family='qwen35',cycle=4,moe=False,ple=False,shared=False,source='https://huggingface.co/Qwen/Qwen3.5-4B/blob/main/config.json',description='3 Gated DeltaNet blocks + 1 gated full-attention block · SwiGLU · partial RoPE'),
 'qwen35_moe': dict(name='Qwen 3.5 · MoE text',family='qwen35',cycle=4,moe=True,ple=False,shared=False,source='https://huggingface.co/Qwen/Qwen3.5-35B-A3B/blob/main/config.json',description='3:1 DeltaNet / attention · routed + gated shared experts'),
 'llama4_scout': dict(name='Llama 4 Scout · text',family='llama4',cycle=4,moe=True,ple=False,shared=False,source='https://huggingface.co/docs/transformers/model_doc/llama4',description='MoE in every block · RoPE / NoPE · chunked attention · L2 Q/K normalization'),
 'llama4_maverick': dict(name='Llama 4 Maverick · text',family='llama4',cycle=4,moe=True,ple=False,shared=False,source='https://huggingface.co/docs/transformers/model_doc/llama4',description='Alternating dense / MoE FFNs · RoPE / NoPE · shared expert · temperature tuning'),
 'gemma4_e2b': dict(name='Gemma 4 E2B · text',family='gemma4',cycle=5,moe=False,ple=True,shared=True,source='https://huggingface.co/google/gemma-4-E2B-it/blob/main/config.json',description='4 local + 1 global · PLE · shared KV · double-width FFN in shared layers'),
 'gemma4_e4b': dict(name='Gemma 4 E4B · text',family='gemma4',cycle=6,moe=False,ple=True,shared=True,source='https://huggingface.co/google/gemma-4-E4B-it/blob/main/config.json',description='5 local + 1 global · PLE · shared KV · dual rotary configurations'),
 'gemma4_31b': dict(name='Gemma 4 31B · text',family='gemma4',cycle=6,moe=False,ple=False,shared=False,source='https://huggingface.co/google/gemma-4-31B-it/blob/main/config.json',description='5 local + 1 global · dense FFN · global K=V · pre/post RMSNorm'),
 'gemma4_moe': dict(name='Gemma 4 26B-A4B · text',family='gemma4',cycle=6,moe=True,ple=False,shared=False,source='https://huggingface.co/google/gemma-4-26B-A4B-it/blob/main/config.json',description='Local / global attention · parallel dense + routed FFNs · global K=V'),
 'qwen4_preview': dict(name='Qwen4 preview · Qwen3.8-Flash-Next text',family='qwen4',cycle=4,moe=True,ple=False,shared=False,source='https://github.com/QwenLM/Qwen3.8-Flash-Next',description='3:1 DeltaNet / QSA · 4 gated residual streams · n-gram embeddings · MoE'),
}
SIZE_NAMES=('nano','tiny','small','medium')


def modern_size(family,size):
    f=MODERN_FAMILIES[family]; i=SIZE_NAMES.index(size); cycle=f['cycle']
    # Shared KV requires an unshared source of both local and global states.
    layers=cycle*([2,3,4,5][i] if f['shared'] else [1,2,3,4][i])
    width=[64,128,256,512][i]; heads=([6,12,12,24] if family=='qwen4_preview' else [4,4,8,8])[i]
    ratio=.25 if family in {'qwen35_moe','qwen4_preview'} else 3.6 if family=='qwen35_dense' else 1.6 if family=='llama4_scout' else 3.2 if family=='llama4_maverick' else .75 if family=='gemma4_moe' else 4
    hidden=max(8,round(width*ratio/8)*8)
    return dict(d_model=width,n_layers=layers,n_heads=heads,n_kv_heads=1 if family=='gemma4_e2b' else max(1,heads//12) if family=='qwen4_preview' else max(1,heads//4),d_ff=hidden,max_seq_len=[128,256,512,1024][i])


def catalog():
    return [dict(id=id,name=f['name'],description=f['description'],source=f['source'],
        sizes={size:modern_size(id,size) for size in SIZE_NAMES},status='Experimental integration · training and numerical validation pending',scope='Text backbone with random weights. Lab sizes reduce dimensions, depth and expert counts. No vision/audio encoders or MTP training heads.')
        for id,f in MODERN_FAMILIES.items()]


def node(id,kind,**params): return dict(id=id,kind=kind,params=params)


def edge(source,target,target_port='in',source_port='out'):
    result=dict(id=f'{source}_{source_port}_{target}_{target_port}',source=source,target=target,targetHandle=target_port)
    if source_port!='out': result['sourceHandle']=source_port
    return result


def layout(graph):
    levels={}; pending={n['id'] for n in graph['nodes']}
    while pending:
        ready=[id for id in pending if all(e['source'] in levels for e in graph['edges'] if e['target']==id)]
        if not ready: break
        for id in ready:
            levels[id]=max([-1]+[levels[e['source']] for e in graph['edges'] if e['target']==id])+1; pending.remove(id)
    maximum=max(levels.values(),default=0); cols={}
    for n in graph['nodes']:
        level=levels.get(n['id'],0); col=cols.get(level,0); cols[level]=col+1
        n['position']={'x':160+col*300,'y':(maximum-level)*145}
        if 'body' in n: layout(n['body'])
    return graph


def layer_graph(family,index,total,width,heads,kv,head_dim,ple_dim,shared,full):
    f=MODERN_FAMILIES[family]; kind=f['family']; qwen4=kind=='qwen4'; gemma=kind=='gemma4'
    nodes=[node('entry','input_hidden')]; edges=[]
    def add(id,brick,source=None,**p):
        nodes.append(node(id,brick,**p))
        if source: edges.append(edge(source,id))
        return id
    def norm(id,source):
        return add(id,'gemma4_norm' if gemma else 'qwen_norm' if kind in {'qwen35','qwen4'} else 'rmsnorm',source,width=width,eps=1e-5 if kind=='llama4' else 1e-6)
    if qwen4:
        if index==1:
            add('token_ids','input_tokens')
            add('ngram','ngram','entry',width=width,streams=4,dim=width,ngram_size=3,heads_per_ngram=2,buckets=max(257,width*4),conv_kernel=4,eos_id=0,seed=1234,eps=1e-6)
            edges.append(edge('token_ids','ngram','tokens')); add('ngram_add','add'); edges.extend([edge('entry','ngram_add','a'),edge('ngram','ngram_add','b')]); origin='ngram_add'
        else: origin='entry'
        pre=add('attn_read','gr_read',origin,width=width,streams=4,rank=max(8,width//8),eps=1e-6)
    else: origin='entry'; pre=norm('norm1',origin)
    if kind in {'qwen35','qwen4'} and not full:
        attn=add('mixer','gated_deltanet',pre,family=kind,width=width,key_heads=max(1,heads*2//3) if qwen4 else heads,value_heads=max(1,heads*2//3)*3 if qwen4 else heads*2,key_dim=max(8,head_dim//2),value_dim=max(8,head_dim//2),conv_kernel=4,eps=1e-6)
    else:
        params=dict(family=kind,width=width,heads=heads,kv_heads=kv,head_dim=head_dim,position='rope',rope_base=10000000 if kind in {'qwen35','qwen4'} else 500000 if kind=='llama4' else 1000000 if full else 10000,
            rotary_fraction=.25 if kind in {'qwen35','qwen4'} or (gemma and full) else 1.,window=512 if gemma and not full else 128 if kind=='llama4' and not full else 0,
            qk_norm=family=='llama4_scout' if kind=='llama4' else True,eps=1e-5 if kind=='llama4' else 1e-6)
        if kind=='llama4': params.update(position='none' if full else 'rope',temperature=full,temperature_scale=.1,temperature_floor=8192,rotary_fraction=1.)
        if gemma: params.update(k_eq_v=full and not f['ple'],kv_shared=shared)
        if qwen4: params.update(index_heads=2,index_dim=max(head_dim,16),budget=32,compression=4,index_loss=.01)
        attn=add('mixer','modern_attention',pre,**params)
    if gemma:
        post=norm('post_attention',attn)
        if shared:
            add('shared_kv','input_kv',width=kv*head_dim); edges.append(edge('shared_kv',attn,'kv'))
        add('kv_output','output_kv'); edges.append(edge(attn,'kv_output','in','kv'))
    else: post=attn
    if qwen4:
        first=add('attn_write','gr_write',post,width=width,streams=4); edges.append(edge('attn_read',first,'state','state'))
        ff_in=add('ffn_read','gr_read',first,width=width,streams=4,rank=max(8,width//8),eps=1e-6)
    else:
        first=add('add1','add'); edges.extend([edge(post,first,'a'),edge(origin,first,'b')]); ff_in=norm('norm2',first)
    use_moe=f['moe'] and not (family=='llama4_maverick' and index%2==0)
    expert_hidden=max(8,round(width*(1.6 if kind=='llama4' else .25)/8)*8)
    if use_moe and not gemma:
        ff=add('ffn','moe',ff_in,family=kind,width=width,experts=16 if family=='llama4_maverick' or qwen4 else 8,
            top_k=1 if kind=='llama4' else 2,hidden=expert_hidden,shared_hidden=expert_hidden,aux_weight=.001,eps=1e-6)
    else:
        ff=add('ffn','swiglu',ff_in,width=width,hidden=max(8,round(width*(8 if family=='gemma4_e2b' and shared else .75 if family=='gemma4_moe' else 3.6 if family=='qwen35_dense' else 3.2 if family=='llama4_maverick' else 4)/8)*8),activation='geglu_tanh' if gemma else 'swiglu',bias=False)
    if gemma and use_moe:
        dense=norm('dense_output_norm',ff)
        expert_input=norm('expert_input_norm',first)
        experts=add('experts','moe',expert_input,family='gemma4',width=width,experts=8,top_k=2,hidden=expert_hidden,aux_weight=.001,eps=1e-6)
        edges.append(edge(first,experts,'router'))
        expert_output=norm('expert_output_norm',experts)
        ff=add('combine_ffns','add'); edges.extend([edge(dense,ff,'a'),edge(expert_output,ff,'b')])
    if gemma: ff=norm('post_ffn',ff)
    if qwen4:
        out=add('ffn_write','gr_write',ff,width=width,streams=4); edges.append(edge('ffn_read',out,'state','state'))
    else:
        out=add('add2','add'); edges.extend([edge(ff,out,'a'),edge(first,out,'b')])
    if f['ple']:
        add('ple_input','input_ple',width=ple_dim*total)
        out=add('ple_injection','ple_apply',out,width=width,dim=ple_dim,index=index,eps=1e-6); edges.append(edge('ple_input',out,'ple'))
    if gemma: out=add('layer_scale','scale',out,width=width,factor=1.)
    add('exit','output_hidden',out)
    return layout(dict(version=2,nodes=nodes,edges=edges))


def modern_template(family,size='nano',dimensions=None):
    f=MODERN_FAMILIES[family]; dims=modern_size(family,size) if dimensions is None else dimensions
    w,L,h,kv=(dims[k] for k in ['d_model','n_layers','n_heads','n_kv_heads']); kind=f['family']; qwen4=kind=='qwen4'
    head_ratio=2.4 if kind=='qwen4' else 2 if family=='qwen35_moe' else 1.6 if kind=='qwen35' else 4/3 if family=='gemma4_e2b' else .8 if family=='gemma4_e4b' else 32/21 if family=='gemma4_31b' else 16/11 if family=='gemma4_moe' else 1
    hd=max(8,round(w/h*head_ratio/8)*8); ple_dim=max(8,round(w*(1/6 if family=='gemma4_e2b' else .1)/8)*8)
    nodes=[node('tokens','tokens'),node('embedding','embedding',scale='sqrt_dim' if kind=='gemma4' else 'none')]
    edges=[edge('tokens','embedding')]; previous='embedding'
    if f['ple']:
        nodes.append(node('ple_table','ple_embedding',width=w,dim=ple_dim,layers=L,eps=1e-6)); edges.extend([edge('embedding','ple_table'),edge('tokens','ple_table','tokens')])
    if qwen4:
        nodes.append(node('expand','residual_expand',width=w,streams=4)); edges.append(edge(previous,'expand')); previous='expand'
    shared_sources={}; sharing_start=max(f['cycle'],round(L*(15/35 if family=='gemma4_e2b' else 24/42)/f['cycle'])*f['cycle']) if f['shared'] else L
    for i in range(L):
        full=(i+1)%f['cycle']==0 or i==L-1
        # Keep local/global dimensions and KV counts independent of d_model.
        layer_hd=hd*2 if kind=='gemma4' and full else hd
        layer_kv=1 if family=='gemma4_e2b' else kv
        if kind=='gemma4' and full and not f['ple']: layer_kv=max(1,kv//2)
        shared=f['shared'] and i>=sharing_start and full in shared_sources
        id=f'layer_{i}'
        body=layer_graph(family,i,L,w,h,layer_kv,layer_hd,ple_dim,shared,full)
        label=f'Layer {i+1} · '+('QSA' if qwen4 and full else 'Gated Attention' if kind=='qwen35' and full else 'Gated DeltaNet' if kind in {'qwen35','qwen4'} else 'global' if full else 'local / RoPE')
        n=node(id,'repeat',count=1,width=w*4 if qwen4 else w); n.update(body=body,label=label); nodes.append(n); edges.append(edge(previous,id)); previous=id
        if f['ple']: edges.append(edge('ple_table',id,'ple'))
        if qwen4 and i==1: edges.append(edge('tokens',id,'tokens'))
        if kind=='gemma4':
            if shared: edges.append(edge(shared_sources[full],id,'kv','kv'))
            elif i<sharing_start: shared_sources[full]=id
    if qwen4:
        nodes.append(node('collapse','gr_collapse',width=w,streams=4,rank=max(8,w//8),eps=1e-6)); edges.append(edge(previous,'collapse')); previous='collapse'
    norm='gemma4_norm' if kind=='gemma4' else 'rmsnorm' if kind=='llama4' else 'qwen_norm'
    nodes.append(node('final_norm',norm,width=w,eps=1e-5 if kind=='llama4' else 1e-6)); edges.append(edge(previous,'final_norm'))
    nodes.append(node('head','head',softcap=30. if kind=='gemma4' else 0.)); edges.append(edge('final_norm','head'))
    return {**dims,'assembly':layout(dict(version=2,nodes=nodes,edges=edges)),'assembly_template':family,'dropout':0.,
            'tie_word_embeddings':f['ple'] or family=='gemma4_31b' or family=='gemma4_moe' or family=='qwen35_dense'}


def resize_modern(config,width,layers,context):
    """Keep edits in surviving layers; new layers inherit the selected family schedule."""
    from .assembly_runtime import validate_assembly
    family=config.assembly_template
    cycle=MODERN_FAMILIES[family]['cycle']
    minimum=cycle*(2 if MODERN_FAMILIES[family]['shared'] else 1)
    old=deepcopy(config.assembly); old_layers=[n for n in old['nodes'] if n['kind']=='repeat']
    original_count=len(old_layers)
    if layers!=original_count and (layers<minimum or layers%cycle):
        raise ValueError(f'This family needs a depth of at least {minimum}, in multiples of {cycle}, to preserve its layer schedule.')
    h=max(1,config.n_heads); kv=max(1,config.n_kv_heads)
    dimensions=dict(d_model=width,n_layers=layers,n_heads=h,n_kv_heads=kv,d_ff=4*width,max_seq_len=context)
    generated=modern_template(family,dimensions=dimensions)['assembly']
    old_index={n['id']:n for n in old['nodes']}
    factor=width/config.d_model
    from .assembly_schema import MODERN_BRICKS
    def scale_graph(graph):
        for n in graph['nodes']:
            p=n.setdefault('params',{})
            # Width-scaled tensor dimensions; dimensionless gates, expert counts,
            # budgets, windows and positional bases stay under user control.
            keys={'width','hidden','shared_hidden','dim','rank','head_dim','key_dim','value_dim','index_dim'}
            for key in keys & p.keys():
                p[key]=max(1,round(p[key]*factor))
                if key in {'head_dim','key_dim','value_dim','index_dim'}: p[key]=max(8,round(p[key]/8)*8)
            if n['kind']=='ple_embedding': p['layers']=layers
            if n['kind']=='input_ple': p['width']=max(8,round(config.d_model/4*factor))*layers
            if n['kind']=='repeat': scale_graph(n['body'])
    scale_graph(old)
    ple=next((n for n in old['nodes'] if n['kind']=='ple_embedding'),None)
    for container in old['nodes']:
        if container['kind']!='repeat': continue
        body=container['body']
        mixer=next((n for n in body['nodes'] if n['kind']=='modern_attention'),None)
        for entry in body['nodes']:
            if entry['kind']=='input_ple' and ple: entry['params']['width']=ple['params']['dim']*layers
            if entry['kind']=='input_kv' and mixer:
                entry['params']['width']=mixer['params']['kv_heads']*mixer['params']['head_dim']
    for n in generated['nodes']:
        previous=old_index.get(n['id'])
        if previous is not None and n['kind']=='repeat':
            # Surviving bodies remain user-authored. Boundary connections follow
            # their actual declared ports, not an assumed family label.
            n['body']=previous['body']; n['position']=previous.get('position',n['position'])
    # Preserve the custom root graph when depth is unchanged. Depth edits use
    # the family chain and retain surviving layer bodies (explicit in the UI).
    if layers==original_count:
        generated=old
    else:
        from .assembly_schema import input_ports
        if ple:
            generated['nodes']=[ple if n['kind']=='ple_embedding' else n for n in generated['nodes']]
        # Rebuild auxiliary wires from the retained bodies' actual ports. Their
        # sharing choices may differ from the new default schedule.
        generated['edges']=[e for e in generated['edges'] if e.get('targetHandle','in') not in {'kv','tokens','ple'}]
        if ple: generated['edges'].append(edge('tokens','ple_table','tokens'))
        kv_sources={}
        for n in generated['nodes']:
            if n['kind']!='repeat': continue
            body=n['body']; needed=set(input_ports(n))
            if 'tokens' in needed: generated['edges'].append(edge('tokens',n['id'],'tokens'))
            if 'ple' in needed and ple:
                generated['edges'].append(edge('ple_table',n['id'],'ple'))
                for entry in body['nodes']:
                    if entry['kind']=='input_ple': entry['params']['width']=ple['params']['dim']*layers
                    if entry['kind']=='ple_apply' and n['id'] not in old_index: entry['params']['dim']=ple['params']['dim']
            mixer=next((x for x in body['nodes'] if x['kind']=='modern_attention' and x.get('params',{}).get('family')=='gemma4'),None)
            if mixer:
                p=mixer['params']; key=(bool(p.get('window',0)),p['kv_heads'],p['head_dim'])
                if 'kv' in needed:
                    if key not in kv_sources and n['id'] not in old_index:
                        candidates=[k for k in kv_sources if k[0]==key[0]]
                        if candidates:
                            source_key=candidates[-1]
                            source_node=next(x for x in generated['nodes'] if x['id']==kv_sources[source_key])
                            source_mixer=next(x for x in source_node['body']['nodes'] if x['kind']=='modern_attention' and x.get('params',{}).get('family')=='gemma4')
                            for field in ['heads','kv_heads','head_dim']: p[field]=source_mixer['params'][field]
                            key=source_key
                            for entry in body['nodes']:
                                if entry['kind']=='input_kv': entry['params']['width']=p['kv_heads']*p['head_dim']
                    if key not in kv_sources: raise ValueError('No earlier compatible KV producer remains for '+n['id'])
                    generated['edges'].append(edge(kv_sources[key],n['id'],'kv','kv'))
                elif any(x['kind']=='output_kv' for x in body['nodes']): kv_sources[key]=n['id']
    result=deepcopy(config); result.assembly=layout(generated); result.d_model=width; result.n_layers=layers; result.max_seq_len=context; result.d_ff=round(config.d_ff*factor) if config.d_ff else 4*width
    result.__post_init__(); return result
