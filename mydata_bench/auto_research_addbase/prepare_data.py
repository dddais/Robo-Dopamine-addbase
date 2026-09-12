"""Freeze full/cohort/ranking inputs and group splits without model outcomes."""
from __future__ import annotations
import concurrent.futures,hashlib,json,pathlib
import cv2,numpy as np
from PIL import Image
from .common import *
from mydata_bench.data import load_episodes
from mydata_bench.io import sha256_file

def frames(video):
    digest=sha256_file(video);out=BASE/'inputs/frames'/digest
    manifest=out/'manifest.json'
    if manifest.exists():return json.loads(manifest.read_text())
    out.mkdir(parents=True,exist_ok=True)
    cap=cv2.VideoCapture(video)
    if not cap.isOpened():raise RuntimeError(f'Cannot open {video}')
    n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));fps=float(cap.get(cv2.CAP_PROP_FPS))
    indices=np.linspace(0,n-1,8).round().astype(int).tolist();chosen=set(indices);saved={};i=0
    while True:
        ok,frame=cap.read()
        if not ok:break
        if i in chosen:
            path=out/f'frame_{i:06d}.png'
            if not path.exists():
                if not cv2.imwrite(str(path),frame):raise IOError(path)
            saved[i]=str(path)
        i+=1
    cap.release()
    if i!=n or indices[-1]!=i-1 or set(saved)!=chosen:raise RuntimeError(f'Decoded frame count mismatch: {video}, {i}, expected {n}')
    record={'image_paths':[saved[i] for i in indices],'image_source_indices':indices,'video_sha256':digest,'image_sampling_record':{'terminal_source_index':n-1,'selected_source_indices':indices,'decoded_frame_count':n,'source_fps':fps}}
    create_json(manifest,record);return record

def endpoints(path):
    values={}
    for r in read_rows(path): values[r['example_id'],r.get('frame')]=r
    return {k:v for k,v in values.items() if v.get('status')=='ok'}

def main():
    target=BASE/'inputs';target.mkdir(parents=True,exist_ok=True)
    if (target/'manifest.json').exists():print('Frozen inputs already complete');return
    episodes=list(load_episodes(DATA,'all',compute_hash=False)); videos=sorted({e.video_path for e in episodes})
    print(f'Preparing {len(episodes)} examples, {len(videos)} video paths',flush=True)
    cache={}
    # Different instructions can refer to byte-identical files. Hash first, decode each video only once.
    paths_by_hash={}
    for v in videos:paths_by_hash.setdefault(sha256_file(v),[]).append(v)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures={pool.submit(frames,v[0]):v for v in paths_by_hash.values()}
        for i,f in enumerate(concurrent.futures.as_completed(futures)):
            result=f.result()
            for v in futures[f]:cache[v]=result
            if i%20==0:print(f'frames {i+1}/{len(futures)}',flush=True)
    frozen=set(json.loads((OLD/'cohorts/auto_grounded_v2/example_ids.json').read_text()))
    ground=endpoints(OLD/'grounding_v2/sam3/grounding.jsonl')
    ranking_ids={r['id'] for r in read_rows(DATA/'ranking_data.jsonl')}
    ranking_ground=endpoints(OLD/'ranking_grounding_v2/sam3/grounding.jsonl')
    full=[];cohort=[];ranking=[];labels=[]
    for e in episodes:
        r={'example_id':e.example_id,'task':e.task,'video_path':e.video_path,**cache[e.video_path]}
        full.append(r)
        labels.append({'example_id':e.example_id,'reward':e.reward,'split':e.split,'subset':e.subset,'source_suc_id':e.source_suc_id,'video_sha256':r['video_sha256']})
        for accepted,g,dst in [(e.example_id in frozen,ground,cohort),(e.example_id in ranking_ids,ranking_ground,ranking)]:
            if not accepted:continue
            first=g.get((e.example_id,'first'));last=g.get((e.example_id,'last'))
            if not first or not last:
                if dst is cohort:raise ValueError(f'Missing grounding {e.example_id}')
                continue
            p=last['provenance'];track=p['tracking_path']
            if not pathlib.Path(track).is_file():raise FileNotFoundError(track)
            dst.append({**r,'first_bbox':first['bbox'],'last_bbox':last['bbox'],'first_image_path':first['provenance']['image_path'],'last_image_path':p['image_path'],'tracking_path':track})
    assert len(full)==1213 and len(cohort)==846
    rank_hashes={r['video_sha256'] for r in ranking}
    dev=[];confirm=[]
    for r in cohort:
        if r['video_sha256'] in rank_hashes:continue
        bucket=int(hashlib.sha256(('addbase-v1:'+r['video_sha256']).encode()).hexdigest()[:8],16)%10
        (dev if bucket<3 else confirm).append(r['example_id'])
    for name,rows in [('full',full),('cohort',cohort),('ranking',ranking),('labels',labels)]:
        path=target/f'{name}.jsonl'
        with path.open('x') as f:
            for r in rows:f.write(json.dumps(r,ensure_ascii=False)+'\n')
    create_json(target/'splits.json',{'method':'sha256(video bytes), exclude ranking groups then deterministic 30/70 development/confirmation','development':dev,'confirmation':confirm,'ranking_ids':[r['example_id'] for r in ranking],'ranking_video_sha256':sorted(rank_hashes)})
    manifest={'time':timestamp(),'full':len(full),'cohort':len(cohort),'ranking':len(ranking),'development':len(dev),'confirmation':len(confirm),'unique_videos':len(paths_by_hash),'full_sha256':fingerprint(full),'cohort_sha256':fingerprint(cohort),'ranking_sha256':fingerprint(ranking),'grounding_assumed_correct':True,'model_payload':'task and decoded pixels only; identity fields are bookkeeping','original_data_untouched':True}
    create_json(target/'manifest.json',manifest);print(json.dumps(manifest),flush=True)
if __name__=='__main__':main()
