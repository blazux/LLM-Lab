"""Build a source release candidate from an explicit allowlist, including new sources.

Does not commit, tag, upload, publish, or include training data/credentials/caches.
The JSON manifest records hashes and sources not yet present in the Git index.
"""
import argparse
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import tarfile

ROOT=Path(__file__).resolve().parents[1]
DIRECTORIES=['assets','src','gui/backend','gui/frontend','docs','examples','recipes','scripts','tests','.github/workflows']
FILES=['VERSION','README.md','LICENSE','Dockerfile','.dockerignore','.gitignore','requirements.txt','requirements-dev.txt','requirements-gpu.txt','constraints.txt','llm-lab.sh','gui/start.sh','build-docker.sh','CHANGELOG.md']
EXCLUDED={'venv','.venv','env','node_modules','__pycache__','cache','.cache','dist','outputs','checkpoints','sft_checkpoints','rlhf_checkpoints','.git'}
EXTENSIONS={'.py','.js','.jsx','.ts','.tsx','.json','.md','.sh','.toml','.txt','.yml','.yaml','.html','.css','.cjs','.mjs','.png','.jpg','.jpeg','.svg'}
PRIVATE_RECORDS={'docs/roadmap-1.0.md','docs/release-validation.md','docs/assets/reference-learning.png'}


def is_private_record(path):
    relative=path.relative_to(ROOT).as_posix()
    return relative in PRIVATE_RECORDS or relative.startswith('docs/release-evidence/')


def sources():
    result=[ROOT/name for name in FILES if (ROOT/name).is_file()]
    for name in DIRECTORIES:
        for current,dirs,files in os.walk(ROOT/name):
            dirs[:]=sorted(d for d in dirs if d not in EXCLUDED and not (d == 'data' and Path(current).relative_to(ROOT).parts[:2] == ('gui', 'backend')) and not (Path(current)/d).is_symlink())
            for filename in sorted(files):
                path=Path(current)/filename
                if path.suffix in EXTENSIONS and not path.is_symlink() and not is_private_record(path):result.append(path)
    return sorted(set(result))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=ROOT/'dist')
    args=parser.parse_args()
    version=(ROOT/'VERSION').read_text().strip()
    frontend=json.loads((ROOT/'gui/frontend/package.json').read_text())
    lock=json.loads((ROOT/'gui/frontend/package-lock.json').read_text())
    if frontend['version']!=version or lock['version']!=version:parser.error('Version labels disagree')
    files=sources()
    try:
        tracked=set(subprocess.check_output(['git','ls-files','-z'],cwd=ROOT,stderr=subprocess.DEVNULL).decode().split('\0'))
        git_available=True
    except subprocess.CalledProcessError:
        tracked=set();git_available=False
    prefix='llm-lab-'+version
    args.output_dir.mkdir(parents=True,exist_ok=True)
    output=args.output_dir/(prefix+'.tar.gz')
    entries=[]
    with output.open('wb') as raw, gzip.GzipFile(filename='',mode='wb',fileobj=raw,mtime=0) as compressed, tarfile.open(fileobj=compressed,mode='w') as archive:
        for path in files:
            name=str(path.relative_to(ROOT));data=path.read_bytes()
            info=tarfile.TarInfo(prefix+'/'+name);info.size=len(data);info.mtime=0;info.mode=0o755 if os.access(path,os.X_OK) else 0o644
            archive.addfile(info,io.BytesIO(data))
            entries.append({'path':name,'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data)})
    digest=hashlib.sha256(output.read_bytes()).hexdigest()
    manifest={'git_index_available':git_available,'version':version,'archive':output.name,'sha256':digest,'files':entries,
              'sources_missing_from_git_index':[item['path'] for item in entries if item['path'] not in tracked]}
    (args.output_dir/(prefix+'.manifest.json')).write_text(json.dumps(manifest,indent=2)+'\n')
    (args.output_dir/'SHA256SUMS').write_text(digest+'  '+output.name+'\n')
    print(json.dumps({'archive':str(output),'sha256':digest,'files':len(files),'sources_missing_from_git_index':len(manifest['sources_missing_from_git_index'])}))

if __name__=='__main__':main()
