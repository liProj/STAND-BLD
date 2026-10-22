"""Check manifest counts, ordering and photographer/capture-group separation."""
from pathlib import Path
import argparse, csv, json, os

def read(path):
    with Path(path).open(encoding='utf-8',newline='') as f:return list(csv.DictReader(f))

def validate(root,check_images=False):
    root=Path(root)
    src=read(root/'data/bld/meta.csv');groups=read(root/'data/bld/groups.csv')
    ev=read(root/'data/inat/eval.csv');pool=read(root/'data/inat/pool.csv')
    dev=[r for r in ev if r['part']=='dev'];test=[r for r in ev if r['part']=='test']
    assert (len(src),len(pool),len(dev),len(test))==(1934,2976,527,1856),'Unexpected release manifest counts'
    assert [r['file'] for r in src]==[r['file'] for r in groups], 'Source file ordering differs'
    for rows in [src,ev,pool]:assert len(rows)==len({r['file'] for r in rows}), 'Duplicate image'
    for a,b in [(dev,test),(dev,pool),(test,pool)]:
        assert {r['user'] for r in a}.isdisjoint({r['user'] for r in b}), 'Photographer leakage'
        assert {r['obs'] for r in a}.isdisjoint({r['obs'] for r in b}), 'Observation leakage'
    assert len({r['user'] for r in test})==1093
    mapping={r['file']:r['session'] for r in groups}
    split_count=0
    for f in sorted((root/'data/bld/splits').glob('session_f[0-9].csv')):
        sp=read(f)
        assert [r['file'] for r in sp]==[r['file'] for r in groups]
        parts={p:{mapping[r['file']] for r in sp if r['part']==p} for p in ['train','val','test']}
        for a,b in [('train','val'),('train','test'),('val','test')]:assert parts[a].isdisjoint(parts[b])
        split_count+=1
    assert split_count==5
    if check_images:
        missing=[]
        for rows,prefix in [(src,'data/bld/img'),(ev,'data/inat/img'),(pool,'data/inat/img')]:
            for r in rows:
                p=(root/prefix/r['file']).resolve()
                assert p.is_relative_to((root/prefix).resolve()), 'Image path escapes its dataset folder'
                if not p.is_file():missing.append(str(p.relative_to(root)))
        if missing:raise FileNotFoundError(f'{len(missing)} missing images. First missing file: {missing[0]}')
    return {'source_images':len(src),'adaptation_images':len(pool),'development_images':len(dev),'test_images':len(test),
            'test_photographers':1093,'capture_folds':split_count,'photographer_disjoint':True,'images_checked':check_images}

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path(os.environ.get('STAND_ROOT',Path(__file__).resolve().parents[1])))
    p.add_argument('--check-images',action='store_true')
    a=p.parse_args();print(json.dumps(validate(a.root,a.check_images),indent=2))
