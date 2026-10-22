"""Download only the fixed photographs listed in the release manifests."""
from pathlib import Path
import argparse, csv, os, time
import requests

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--set',choices=['eval','pool','all'],default='all')
    p.add_argument('--root',type=Path,default=Path(os.environ.get('STAND_ROOT',Path(__file__).resolve().parents[1])))
    p.add_argument('--limit',type=int)
    p.add_argument('--dry-run',action='store_true')
    a=p.parse_args();rows=[]
    for name in (['eval','pool'] if a.set=='all' else [a.set]):
        with (a.root/f'data/inat/{name}.csv').open(encoding='utf-8') as f:rows.extend(csv.DictReader(f))
    if a.limit is not None:rows=rows[:a.limit]
    target=(a.root/'data/inat/img').resolve()
    if a.dry_run:print(f'{len(rows)} photographs listed; destination {target}');return
    session=requests.Session();session.headers['User-Agent']='STAND-BLD research reproduction (https://github.com/liProj/STAND-BLD)'
    failed=[]
    for i,r in enumerate(rows,1):
        dst=(target/r['file']).resolve()
        if not dst.is_relative_to(target):raise ValueError('Unsafe filename in manifest')
        if dst.exists() and dst.stat().st_size>2000:continue
        ok=False
        for attempt in range(3):
            try:
                response=session.get(r['url'],timeout=45)
                response.raise_for_status()
                if not response.content.startswith(b'\xff\xd8'):raise ValueError('Response is not a JPEG')
                dst.parent.mkdir(parents=True,exist_ok=True)
                tmp=dst.with_suffix('.jpg.part');tmp.write_bytes(response.content);tmp.replace(dst)
                ok=True;break
            except (requests.RequestException,ValueError):time.sleep(2*(attempt+1))
        if not ok:failed.append(r['file'])
        if i%100==0:print(f'{i}/{len(rows)} processed; {len(failed)} unavailable',flush=True)
        time.sleep(.2)
    if failed:
        out=a.root/'results/unavailable_images.txt';out.parent.mkdir(parents=True,exist_ok=True);out.write_text('\n'.join(failed),encoding='utf-8')
        raise RuntimeError(f'{len(failed)} photographs unavailable. See {out}; do not silently replace split entries.')
    print(f'Completed {len(rows)} manifest entries')

if __name__=='__main__':main()
