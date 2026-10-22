"""Run the recorded STAND configuration; use --dry-run to inspect the command."""
import argparse, json, os, subprocess, sys
from pathlib import Path
from validate_data import validate

def main():
    repo=Path(__file__).resolve().parents[1]
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=repo/'configs/stand.json')
    parser.add_argument('--root',type=Path,default=Path(os.environ.get('STAND_ROOT',repo)))
    parser.add_argument('--dry-run',action='store_true')
    args=parser.parse_args()
    root=args.root.resolve();cfg=json.loads(args.config.read_text(encoding='utf-8'))
    validate(root)
    command=[sys.executable,str(repo/'code/distill.py'),cfg['tag']]+[f'{k}={v}' for k,v in cfg['args'].items()]
    if args.dry_run:
        print(json.dumps({'root':str(root),'command':command,'SAVE_HEADS':'1'},indent=2));return
    import torch
    if not torch.cuda.is_available():raise RuntimeError('The recorded training implementation requires a CUDA GPU. CPU prediction verification is available separately.')
    for sset in ['bld','inat','pool']:
        f=root/f'results/feats/{sset}__vit_base_patch16_dinov3.lvd1689m__384.npz'
        if not f.is_file():raise FileNotFoundError(f'Extract the frozen features first: {f}')
    for d in ['results/preds','results/distill','logs']:(root/d).mkdir(parents=True,exist_ok=True)
    env=dict(os.environ,STAND_ROOT=str(root),SAVE_HEADS='1')
    subprocess.run(command,cwd=repo,env=env,check=True)

if __name__=='__main__':main()
