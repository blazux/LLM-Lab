"""Opt-in live private Hub acceptance. Creates one new repo; never overwrites a repo.

Requires an existing `hf auth login`, the dev dependencies, and a unique --repo-id.
The private synthetic model is retained for inspection. No token is logged or saved.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'tests'),str(ROOT/'src'),str(ROOT/'gui/backend')]

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-id',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    from huggingface_hub import HfApi, snapshot_download, get_token
    from huggingface_hub.errors import RepositoryNotFoundError
    api=HfApi(); token=get_token()
    if not token: parser.error('Authenticate with hf auth login first.')
    owner=api.whoami()['name']
    if args.repo_id.split('/')[0]!=owner: parser.error('Use your personal namespace for the disposable acceptance repo.')
    try: api.model_info(args.repo_id)
    except RepositoryNotFoundError: pass
    else: parser.error('Choose a new repo; existing repositories are never changed by this test.')
    from test_run_tracking import RunApiFixture
    from model.assembly_templates import family_template
    fixture=RunApiFixture('runTest')
    try:
        fixture.setUp()
        fixture.payload['model_cfg']={**family_template('llama3'), 'tokenizer_name':str(fixture.tokenizer), 'vocab_size':8, 'max_seq_len':16}
        start=fixture.client.post('/api/training/start',json=fixture.payload)
        start.raise_for_status(); fixture.wait_for_run()
        run=fixture.store.get(start.json()['run_id'])
        checkpoint=str(Path(run['output_dir'])/'final_model.pt')
        local=fixture.client.post('/api/export/local',json={'checkpoint_path':checkpoint,'output_dir':str(fixture.root/'export')})
        assert local.json()['success'], local.text
        # Concrete local artifact exists before the external publication step.
        pushed=fixture.client.post('/api/export/hub',json={'checkpoint_path':checkpoint,'repo_id':args.repo_id,
            'hf_token':token,'private':True,'model_description':'Disposable LLM-Lab release acceptance model. Synthetic tiny training; no language-quality claim.'})
        assert pushed.json()['success'], pushed.text
        info=api.model_info(args.repo_id)
        assert info.private, 'Acceptance repository must remain private'
        downloaded=Path(snapshot_download(args.repo_id,revision=info.sha,local_dir=fixture.root/'download',token=token))
        for name in ['model.safetensors','config.json','tokenizer.json','README.md','assembly.json','modeling_assembly.py']:
            assert (downloaded/name).is_file(),name
        code='''
import pathlib,sys,torch
from transformers import AutoModelForCausalLM,AutoTokenizer
root=pathlib.Path(sys.argv[1]); torch.set_num_threads(1)
a=AutoModelForCausalLM.from_pretrained(root/'export',trust_remote_code=True,local_files_only=True).eval()
b=AutoModelForCausalLM.from_pretrained(root/'download',trust_remote_code=True,local_files_only=True).eval()
t=AutoTokenizer.from_pretrained(root/'download',local_files_only=True)
x=t('a little',return_tensors='pt')['input_ids']
with torch.no_grad():
 torch.testing.assert_close(a(x).logits,b(x).logits,rtol=0,atol=0)
 aa=a.generate(x,max_new_tokens=3,do_sample=False,eos_token_id=t.eos_token_id,pad_token_id=t.pad_token_id)
 bb=b.generate(x,max_new_tokens=3,do_sample=False,eos_token_id=t.eos_token_id,pad_token_id=t.pad_token_id)
 assert torch.equal(aa,bb)
'''
        result=subprocess.run([sys.executable,'-I','-c',code,str(fixture.root)],cwd=fixture.root,
            env={**os.environ,'HF_HUB_OFFLINE':'1'},capture_output=True,text=True,timeout=60)
        assert result.returncode==0,result.stdout+result.stderr
        report={'repo_id':args.repo_id,'revision':info.sha,'private':True,'passed':True,
            'checks':['API train','API local export','API private Hub export','revision-pinned download','tokenizer/model card/code','isolated exact logits and greedy generation']}
        args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report))
    finally: fixture.doCleanups()

if __name__=='__main__': main()
