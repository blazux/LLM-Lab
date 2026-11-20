"""Reference-backed text-backbone bricks. Requires Transformers 5.20.1.

Adapters keep graph ports explicit. No pretrained weights or executable code are
fetched at runtime. Reference kernels use the installed Transformers package.
"""
import importlib
import math
from types import SimpleNamespace
import torch
from torch import nn
from torch.nn import functional as F
from .assembly_schema import defaults


def reference(family):
    module = {'gemma4':'gemma4', 'llama4':'llama4', 'qwen35':'qwen3_5', 'qwen4':'qwen4_exp'}[family]
    try:
        return importlib.import_module(f'transformers.models.{module}.modeling_{module}')
    except ImportError as error:
        raise ImportError('Modern architecture bricks require transformers==5.20.1. Install the project requirements.') from error


def configuration(family, p, vocab, context, kind):
    module = {'gemma4':'gemma4', 'llama4':'llama4', 'qwen35':'qwen3_5', 'qwen4':'qwen4_exp'}[family]
    name = {'gemma4':'Gemma4TextConfig','llama4':'Llama4TextConfig','qwen35':'Qwen3_5TextConfig','qwen4':'Qwen4ExpTextConfig'}[family]
    cls = getattr(importlib.import_module(f'transformers.models.{module}.configuration_{module}'), name)
    w=p.get('width',64); h=p.get('heads',4); kv=p.get('kv_heads',1); hd=p.get('head_dim',16)
    fraction=p.get('rotary_fraction',1); rd=max(2,int(hd*fraction)); pairs=rd//2
    sections=[pairs//3,pairs//3,pairs-2*(pairs//3)]
    rope={'rope_type':'default','rope_theta':p.get('rope_base',10000000),'partial_rotary_factor':fraction,'mrope_section':sections,'mrope_interleaved':True}
    args=dict(vocab_size=vocab,hidden_size=w,num_hidden_layers=1,num_attention_heads=h,num_key_value_heads=kv,
              head_dim=hd,max_position_embeddings=context,rms_norm_eps=p.get('eps',1e-6),attention_dropout=0.,attention_bias=False)
    if family=='gemma4':
        layer_type='sliding_attention' if p.get('window',0) else 'full_attention'
        args.update(num_hidden_layers=3,layer_types=[layer_type,layer_type,'full_attention'],global_head_dim=hd,
                    num_global_key_value_heads=kv,sliding_window=max(1,p.get('window',0)),attention_k_eq_v=p.get('k_eq_v',False),
                    num_kv_shared_layers=2,hidden_size_per_layer_input=0,hidden_activation='gelu_pytorch_tanh',
                    rope_parameters={layer_type:{'rope_type':'proportional' if fraction!=1 else 'default','rope_theta':p.get('rope_base',10000),'partial_rotary_factor':fraction},
                        **({'full_attention':{'rope_type':'default','rope_theta':10000}} if layer_type!='full_attention' else {})},
                    num_experts=p.get('experts',8),top_k_experts=p.get('top_k',2),moe_intermediate_size=p.get('hidden',32))
    elif family=='llama4':
        use_rope=p.get('position','rope')=='rope'
        args.update(no_rope_layers=[int(use_rope)],layer_types=['full_attention'],use_qk_norm=p.get('qk_norm',True),
                    attn_temperature_tuning=p.get('temperature',False),attn_scale=p.get('temperature_scale',.1),floor_scale=p.get('temperature_floor',8192),
                    rope_parameters={'rope_type':'default','rope_theta':p.get('rope_base',500000)},num_local_experts=p.get('experts',8),num_experts_per_tok=p.get('top_k',1),
                    intermediate_size=p.get('hidden',32),intermediate_size_mlp=p.get('shared_hidden',32))
    else:
        args.update(layer_types=['linear_attention' if kind in {'gated_deltanet','ngram'} else 'full_attention'],rope_parameters=rope,
                    linear_num_key_heads=p.get('key_heads',2),linear_num_value_heads=p.get('value_heads',4),
                    linear_key_head_dim=p.get('key_dim',16),linear_value_head_dim=p.get('value_dim',16),linear_conv_kernel_dim=p.get('conv_kernel',4))
        if family=='qwen4':
            args.update(hc_count=p.get('streams',4),hc_lowrank=p.get('rank',16),ple_layer_ids=[1] if kind=='ngram' else [],ple_embed_dim=p.get('dim',w),
                ngram_size=p.get('ngram_size',3),heads_per_ngram=p.get('heads_per_ngram',2),ngram_vocab_size_base=p.get('buckets',257),
                make_ngram_vocab_size_divisible_by=8,split_ngram_parts=1,seed=p.get('seed',1234),eos_token_id=p.get('eos_id',0),
                ple_conv_kernel_size=p.get('conv_kernel',4),indexer_n_heads=p.get('index_heads',2),indexer_kv_heads=1,
                indexer_head_dim=p.get('index_dim',16),indexer_budget=p.get('budget',32),indexer_compress_ratio=p.get('compression',4),
                moe_intermediate_size=p.get('hidden',32),shared_expert_intermediate_size=p.get('shared_hidden',32),
                num_experts=p.get('experts',8),num_experts_per_tok=p.get('top_k',2),output_gate_type='sigmoid')
    c=cls(**args); c._attn_implementation='eager'
    return c


def initialize(module, cfg, family):
    ref=reference(family)
    base=getattr(ref,{'gemma4':'Gemma4PreTrainedModel','llama4':'Llama4PreTrainedModel','qwen35':'Qwen3_5PreTrainedModel','qwen4':'Qwen4ExpPreTrainedModel'}[family])
    initializer=base(cfg)
    module.apply(initializer._init_weights)
    for child in module.modules():
        if 'Norm' in type(child).__name__: child._llmlab_no_weight_decay=True
    return module


def segmented(fn, x, inputs, doc_ids):
    """Reset recurrent/convolutional history at each packed-document boundary."""
    if doc_ids is None: return fn(x,inputs)
    batches=[]
    for b in range(x.shape[0]):
        changes=(doc_ids[b,1:]!=doc_ids[b,:-1]).nonzero().flatten().tolist()
        boundaries=[0]+[i+1 for i in changes]+[x.shape[1]]
        pieces=[]
        for a,z in zip(boundaries,boundaries[1:]):
            local={key:(value[b:b+1,a:z] if isinstance(value,torch.Tensor) else value) for key,value in inputs.items()}
            pieces.append(fn(x[b:b+1,a:z],local))
        batches.append(torch.cat(pieces,dim=1))
    return torch.cat(batches,dim=0)


class ReferenceAttention(nn.Module):
    def __init__(self,p,vocab,context):
        super().__init__(); self.p=p; self.family=p['family']; ref=reference(self.family)
        self.config=configuration(self.family,p,vocab,context,'modern_attention')
        names={'gemma4':('Gemma4TextAttention','Gemma4TextRotaryEmbedding'),'llama4':('Llama4TextAttention','Llama4TextRotaryEmbedding'),
               'qwen35':('Qwen3_5Attention','Qwen3_5TextRotaryEmbedding'),'qwen4':('Qwen4ExpTextAttention','Qwen4ExpTextRotaryEmbedding')}
        attention,rotary=names[self.family]
        idx=int(p.get('kv_shared',False)) if self.family=='gemma4' else 0
        self.attention=initialize(getattr(ref,attention)(self.config,idx),self.config,self.family)
        with torch.device('cpu'):
            self.rotary=getattr(ref,rotary)(self.config)
        self._rotary_ready = False
        self.aux_loss=None
        if not p.get('qk_norm',True) and self.family!='llama4':
            self.attention.q_norm=nn.Identity()
            if hasattr(self.attention,'k_norm'): self.attention.k_norm=nn.Identity()
        if self.family=='qwen4':
            pass

    def forward(self,x,inputs,doc_ids=None):
        self.aux_loss=None
        # HF loads under empty-weight initialization and may materialize unsaved
        # rotary buffers as uninitialized storage. Rebuild deterministic buffers
        # after loading, outside that context; rotary modules have no parameters.
        if not self._rotary_ready:
            with torch.device('cpu'):
                self.rotary = type(self.rotary)(self.config)
            self.rotary = self.rotary.to(x.device)
            self._rotary_ready = True
        B,T,_=x.shape; pos=torch.arange(T,device=x.device).unsqueeze(0).expand(B,-1)
        distance=pos[0,:,None]-pos[0,None,:]
        visible=(distance>=0)[None,None].expand(B,1,T,T)
        window=self.p.get('window',0)
        if window:
            if self.family=='llama4': visible=visible & ((pos[:,:,None]//window)==(pos[:,None,:]//window)).unsqueeze(1)
            else: visible=visible & (distance<window)[None,None]
        if doc_ids is not None: visible=visible & (doc_ids[:,None,:,None]==doc_ids[:,None,None,:])
        mask=torch.zeros((B,1,T,T),device=x.device,dtype=x.dtype).masked_fill(~visible,torch.finfo(x.dtype).min)
        if self.family=='gemma4':
            layer_type=self.attention.layer_type
            embeddings=self.rotary(x,pos,layer_type)
            if self.p.get('position')=='none': embeddings=(torch.ones_like(embeddings[0]),torch.zeros_like(embeddings[1]))
            states={}
            if self.p.get('kv_shared'): states[layer_type]=inputs['kv']
            output=self.attention(x,embeddings,mask,shared_kv_states=states)[0]
            return {'out':output,'kv':states[layer_type]}
        embeddings=self.rotary(x,pos if self.family=='llama4' else pos.unsqueeze(0).expand(3,-1,-1))
        output=self.attention(x,embeddings,mask)[0]
        if self.family=='qwen4' and self.training and self.p.get('index_loss',0):
            self.aux_loss=self._indexer_distillation(x,embeddings,visible)*self.p['index_loss']
        return output

    def _indexer_distillation(self,x,embeddings,visible):
        """Train hard top-k index selection against detached dense attention.

        The reference inference indexer has no differentiable path through top-k.
        This explicit KL objective trains its projections without changing QSA
        selection or allowing the future/documents to enter its targets.
        """
        ref=reference('qwen4'); a=self.attention; idx=a.indexer; B,T,_=x.shape
        cos,sin=embeddings
        with torch.no_grad():
            query,_=a.q_proj(x).view(B,T,-1,a.head_dim*2).chunk(2,-1)
            q=a.q_norm(query).transpose(1,2); k=a.k_norm(a.k_proj(x).view(B,T,-1,a.head_dim)).transpose(1,2)
            q,k=ref.apply_rotary_pos_emb(q,k,cos,sin)
            k=k.repeat_interleave(a.num_key_value_groups,dim=1)
            scores=(q.float()@k.float().transpose(-1,-2))*a.scaling
            teacher=scores.masked_fill(~visible,float('-inf')).softmax(-1).mean(1)
        raw_q,raw_k=idx.index_qk_proj(x).split([idx.index_n_heads*idx.index_head_dim,idx.index_kv_heads*idx.index_head_dim],-1)
        q=idx.q_layernorm(raw_q.view(B,T,idx.index_n_heads,idx.index_head_dim))
        q=ref.apply_rotary_pos_emb(q,cos=cos,sin=sin,unsqueeze_dim=2)
        k=raw_k.view(B,T,idx.index_head_dim)
        losses=[]
        for b in range(B):
            for t in range(T):
                indices=visible[b,0,t].nonzero().flatten(); count=len(indices)//idx.compress_ratio
                if count<2: continue
                groups=indices[:count*idx.compress_ratio].view(count,idx.compress_ratio)
                pooled=idx.k_layernorm(k[b,groups].float().mean(1).to(k.dtype))
                pooled=ref.apply_rotary_pos_emb(pooled[:,None],cos=cos[b,groups[:,0]],sin=sin[b,groups[:,0]]).squeeze(1)
                prediction=(q[b,t].float()@pooled.float().T).relu().sum(0)/math.sqrt(idx.index_head_dim)
                target=teacher[b,t,groups].sum(-1); target=target/target.sum().clamp_min(1e-9)
                losses.append(F.kl_div(prediction.log_softmax(-1),target,reduction='sum'))
        return torch.stack(losses).mean() if losses else idx.index_qk_proj.weight.sum()*0


class ReferenceDeltaNet(nn.Module):
    def __init__(self,p,vocab,context):
        super().__init__(); family=p['family']; c=configuration(family,p,vocab,context,'gated_deltanet')
        cls=getattr(reference(family),'Qwen3_5GatedDeltaNet' if family=='qwen35' else 'Qwen4ExpTextGatedDeltaNet')
        self.module=initialize(cls(c,0),c,family)
    def forward(self,x,inputs,doc_ids=None): return segmented(lambda t,i:self.module(t,cache_params=None,attention_mask=None),x,inputs,doc_ids)


class ReferenceMoE(nn.Module):
    def __init__(self,p,vocab,context):
        super().__init__(); self.p=p; self.family=p['family']; self.aux_loss=None; self.router_output=None
        c=configuration(self.family,p,vocab,context,'moe'); self.config=c
        if self.family=='gemma4':
            ref=reference('gemma4'); self.router=initialize(ref.Gemma4TextRouter(c),c,'gemma4'); self.experts=initialize(ref.Gemma4TextExperts(c),c,'gemma4')
        elif self.family=='qwen35':
            from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeTextConfig
            from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeSparseMoeBlock,Qwen3_5MoePreTrainedModel
            c=Qwen3_5MoeTextConfig(hidden_size=p['width'],intermediate_size=p['hidden'],moe_intermediate_size=p['hidden'],
                shared_expert_intermediate_size=p['shared_hidden'],num_experts=p['experts'],num_experts_per_tok=p['top_k'],num_hidden_layers=1)
            c._attn_implementation='eager'; self.module=Qwen3_5MoeSparseMoeBlock(c); self.module.apply(Qwen3_5MoePreTrainedModel(c)._init_weights)
        else:
            cls=getattr(reference(self.family),'Llama4TextMoe' if self.family=='llama4' else 'Qwen4ExpTextSparseMoeBlock')
            self.module=initialize(cls(c),c,self.family)
            if self.family=='llama4':
                self.module.shared_expert=initialize(reference('llama4').Llama4TextMLP(c,intermediate_size=p['shared_hidden']),c,'llama4')
        if self.family!='gemma4':
            (self.module.router if self.family=='llama4' else self.module.gate).register_forward_hook(self._capture)
    def _capture(self,module,args,out): self.router_output=out[1] if self.family=='llama4' else out[0]
    def forward(self,x,inputs,doc_ids=None):
        self.aux_loss=None
        if self.family=='gemma4':
            probs,weights,selected=self.router(inputs['router'].reshape(-1,x.shape[-1]))
            y=self.experts(x.reshape(-1,x.shape[-1]),selected,weights).reshape_as(x)
        else:
            out=self.module(x); y=(out[0] if isinstance(out,tuple) else out).reshape_as(x)
            probs=self.router_output.float().softmax(-1)
            selected=probs.topk(self.p['top_k'],-1).indices
        if self.training and self.p['aux_weight']:
            usage=F.one_hot(selected,self.p['experts']).float().mean((0,1))
            self.aux_loss=self.p['aux_weight']*self.p['experts']*(usage*probs.float().mean(0)).sum()
        return y


class GemmaPLE(nn.Module):
    def __init__(self,p,vocab,context):
        super().__init__(); ref=reference('gemma4'); w,d,L=p['width'],p['dim'],p['layers']
        self.config=SimpleNamespace(num_hidden_layers=L)
        self._embedding_scale_ready=False
        self.hidden_size_per_layer_input=d; self.per_layer_input_scale=2**-.5; self.per_layer_model_projection_scale=w**-.5
        self.embed_tokens_per_layer=ref.Gemma4TextScaledWordEmbedding(vocab,L*d,None,embed_scale=d**.5)
        self.embed_tokens_per_layer.embed_scale=torch.tensor(d**.5,device='cpu')
        self.per_layer_model_projection=nn.Linear(w,L*d,bias=False)
        self.per_layer_projection_norm=ref.Gemma4RMSNorm(d,eps=p['eps'])
        self.per_layer_projection_norm._llmlab_no_weight_decay=True
    def forward(self,x,inputs,doc_ids=None):
        if not self._embedding_scale_ready:
            embedding = self.embed_tokens_per_layer
            embedding.embed_scale = torch.tensor(embedding.scalar_embed_scale, device=x.device)
            self._embedding_scale_ready=True
        model=reference('gemma4').Gemma4TextModel
        identity=model.get_per_layer_inputs(self,inputs['tokens'],x)
        return model.project_per_layer_inputs(self,x,identity)


class GemmaPLEApply(nn.Module):
    def __init__(self,p):
        super().__init__(); self.index=p['index']; w,d=p['width'],p['dim']; ref=reference('gemma4')
        self.gate=nn.Linear(w,d,bias=False); self.projection=nn.Linear(d,w,bias=False); self.norm=ref.Gemma4RMSNorm(w,eps=p['eps'])
        self.norm._llmlab_no_weight_decay=True
    def forward(self,x,inputs,doc_ids=None):
        return x+self.norm(self.projection(F.gelu(self.gate(x),approximate='tanh')*inputs['ple'][:,:,self.index]))


class GatedResidual(nn.Module):
    def __init__(self,p,collapse=False):
        super().__init__(); self.collapse=collapse
        c=configuration('qwen4',p,32,32,'gr_read')
        self.module=initialize(reference('qwen4').Qwen4ExpTextGatedResidual(c,use_combine=not collapse),c,'qwen4')
    def forward(self,x,inputs,doc_ids=None):
        y=self.module(x)
        return y if self.collapse else {'out':y[0],'state':(y[1],y[2])}


class NGram(nn.Module):
    def __init__(self,p,vocab,context):
        super().__init__(); c=configuration('qwen4',p,vocab,context,'ngram')
        self.module=initialize(reference('qwen4').Qwen4ExpTextPLELayer(c,0,0),c,'qwen4')
    def forward(self,x,inputs,doc_ids=None): return segmented(lambda t,i:self.module(t,i['tokens'],past_key_values=None),x,inputs,doc_ids)


class SimpleBrick(nn.Module):
    def __init__(self,kind,p): super().__init__(); self.kind=kind; self.p=p
    def forward(self,x,inputs,doc_ids=None):
        if self.kind=='residual_expand': return x.repeat(1,1,self.p['streams'])
        if self.kind=='gr_write':
            residual,gates=inputs['state']
            return residual+(x.unsqueeze(-2)*gates.unsqueeze(-1)).flatten(-2)
        return x*self.p['factor']


def build_modern(kind,params,width,vocab,context):
    p={**defaults(kind,width),**params}
    classes={'modern_attention':ReferenceAttention,'gated_deltanet':ReferenceDeltaNet,'moe':ReferenceMoE,'ple_embedding':GemmaPLE,'ngram':NGram}
    if kind in classes: return classes[kind](p,vocab,context)
    if kind=='ple_apply': return GemmaPLEApply(p)
    if kind in {'gr_read','gr_collapse'}: return GatedResidual(p,kind=='gr_collapse')
    if kind=='gemma4_norm':
        module=reference('gemma4').Gemma4RMSNorm(p['width'],eps=p['eps'],with_scale=p['learned_scale']); module._llmlab_no_weight_decay=True; return module
    if kind=='qwen_norm':
        module=reference('qwen35').Qwen3_5RMSNorm(p['width'],eps=p['eps']); module._llmlab_no_weight_decay=True; return module
    return SimpleBrick(kind,p)
