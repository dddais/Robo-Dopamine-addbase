"""Append-only experiment I/O. Deliberately never invokes git."""
from __future__ import annotations
import datetime, hashlib, json, os, pathlib, platform, sys
ROOT=pathlib.Path(__file__).resolve().parents[2]
DATA=pathlib.Path('/home/dais/workspace/data/mydata_v2/new')
OLD=pathlib.Path('/home/dais/workspace/Robo-Dopamine/results/mydata_bench')
BASE=ROOT/'results/mydata_bench/experiments_v2_addbase/session_20260908'
RESEARCH=ROOT/'results/mydata_bench/experiments_v2_corssmodel/auto_research/session_20260908'

def fingerprint(obj):
    return hashlib.sha256(json.dumps(obj,sort_keys=True,ensure_ascii=False,allow_nan=False).encode()).hexdigest()

def read_rows(path):
    with pathlib.Path(path).open() as f:
        for line in f:
            if line.strip(): yield json.loads(line)

def output_path(path):
    path=pathlib.Path(path).resolve()
    if not path.is_relative_to(ROOT):raise ValueError(f'Writes outside target repository are forbidden: {path}')
    return path

def create_json(path,obj):
    path=output_path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f:json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')

def append(path,obj):
    path=output_path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a') as f:
        f.write(json.dumps(obj,ensure_ascii=False,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())

def timestamp():return datetime.datetime.now(datetime.timezone.utc).isoformat()

def provenance(config):
    import torch,transformers
    blobs={str(p.relative_to(ROOT)):p.read_bytes() for folder in ['auto_research_addbase','meter_eval','top_eval'] for p in (ROOT/'mydata_bench'/folder).glob('*.py')}
    files={name:hashlib.sha256(blob).hexdigest() for name,blob in blobs.items()}
    snapshots=BASE/'source_snapshots';snapshots.mkdir(parents=True,exist_ok=True)
    for name,digest in files.items():
        target=snapshots/(digest+'.py')
        if not target.exists():
            try:
                with target.open('xb') as handle:handle.write(blobs[name])
            except FileExistsError:
                pass  # Another own worker stored the same content-addressed file.
    return {'time':timestamp(),'config':config,'config_sha256':fingerprint(config),'source_snapshot_dir':str(snapshots),'source_sha256':files,'python':sys.version,'torch':torch.__version__,'transformers':transformers.__version__,'cwd':str(ROOT),'pid':os.getpid(),'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),'git':'not invoked per user instruction'}
