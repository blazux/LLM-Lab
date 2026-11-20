"""Manual handoff checks for the modern text-backbone templates.

No datasets, tokenizers or pretrained weights are downloaded. Random-token loss
checks execution only; it says nothing about the model's ability to learn.
"""
import argparse
import json
from pathlib import Path
import sys
import tempfile

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))


def main():
    from model.modern_templates import MODERN_FAMILIES
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--family',choices=list(MODERN_FAMILIES))
    parser.add_argument('--all',action='store_true',help='Check every modern variant')
    parser.add_argument('--list',action='store_true')
    parser.add_argument('--size',choices=['nano','tiny','small','medium'],default='nano')
    parser.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    parser.add_argument('--length',type=int,default=16)
    parser.add_argument('--vocab-size',type=int,default=128)
    parser.add_argument('--backward',action='store_true',help='Run backward on random next-token loss and auxiliary losses')
    parser.add_argument('--checkpoint',action='store_true',help='Exercise activation checkpointing')
    parser.add_argument('--export',action='store_true',help='Export to a temporary directory and reload the standalone model')
    args=parser.parse_args()
    if args.list:
        print('\n'.join(f"{key}: {value['name']}" for key,value in MODERN_FAMILIES.items())); return
    if not args.family and not args.all: parser.error('Choose --family, --all or --list.')
    import torch
    from config import ModelConfig
    from model.assembly_templates import family_template
    from model.factory import build_model
    if args.device=='cuda' and not torch.cuda.is_available(): raise RuntimeError('CUDA was requested but is unavailable.')
    torch.set_num_threads(1)
    families=list(MODERN_FAMILIES) if args.all else [args.family]
    results=[]
    for family in families:
        torch.manual_seed(42)
        cfg=ModelConfig(**{**family_template(family,args.size),'vocab_size':args.vocab_size})
        if not 2<=args.length<cfg.max_seq_len: parser.error('Length must be at least 2 and less than context length.')
        model=build_model(cfg).to(args.device)
        model.train(args.backward)
        tokens=torch.randint(args.vocab_size,(2,args.length+1),device=args.device)
        # Include a packed-document boundary to exercise convolution/memory reset.
        docs=torch.zeros((2,args.length),device=args.device,dtype=torch.long); docs[:,args.length//2:]=1
        logits,aux=model(tokens[:,:-1],doc_ids=docs,use_checkpoint=args.checkpoint)
        loss=torch.nn.functional.cross_entropy(logits.float().reshape(-1,args.vocab_size),tokens[:,1:].reshape(-1))
        if aux is not None: loss=loss+aux
        if not torch.isfinite(loss): raise RuntimeError(f'{family}: non-finite loss')
        if args.backward:
            loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()): raise RuntimeError(f'{family}: non-finite gradient')
        result=dict(family=family,parameters=model.count_parameters(),shape=list(logits.shape),loss=float(loss.detach()),auxiliary_loss=None if aux is None else float(aux.detach()),backward=args.backward)
        if args.export:
            from export import export_to_local
            from transformers import AutoModelForCausalLM
            with tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp)
                torch.save({'model_config':cfg,'model_state_dict':model.state_dict()},root/'checkpoint.pt')
                export_to_local(str(root/'checkpoint.pt'),str(root/'export'),include_tokenizer=False)
                restored=AutoModelForCausalLM.from_pretrained(root/'export',trust_remote_code=True,local_files_only=True).to(args.device).eval()
                model.eval()
                with torch.no_grad():
                    reference=model(tokens[:,:-1])[0]
                    actual=restored(tokens[:,:-1]).logits
                    torch.testing.assert_close(actual,reference,rtol=0,atol=0)
                result['export_reload']=True
        print(json.dumps(result),flush=True); results.append(result)
        del model,logits,loss,aux
        if args.device=='cuda': torch.cuda.empty_cache()
    print(f'{len(results)} architecture checks completed.')


if __name__=='__main__': main()
