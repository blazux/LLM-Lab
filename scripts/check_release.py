"""Extract and test the source artifact outside the checkout, with isolated imports."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive',required=True,type=Path)
    args=parser.parse_args()
    with tempfile.TemporaryDirectory(prefix='llmlab-artifact-check-') as tmp:
        with tarfile.open(args.archive) as archive:archive.extractall(tmp,filter='data')
        roots=list(Path(tmp).iterdir())
        if len(roots)!=1:parser.error('Archive must contain one source root')
        root=roots[0]
        for name in ['src/data/__init__.py','src/data/data.py','src/model/assembly_runtime.py','VERSION','gui/frontend/postcss.config.js','gui/frontend/tailwind.config.js','gui/frontend/src/content/modern-bricks.json']:
            if not (root/name).is_file():parser.error('Required release file missing: '+name)
        env={**os.environ,'PYTHONPATH':'','HF_HUB_OFFLINE':'1'}
        subprocess.run([sys.executable,'-I','-m','unittest','discover','-s','tests','-p','test_*.py','-v'],cwd=root,env=env,check=True)
        print('Source artifact passed outside the checkout, version '+(root/'VERSION').read_text().strip())

if __name__=='__main__':main()
