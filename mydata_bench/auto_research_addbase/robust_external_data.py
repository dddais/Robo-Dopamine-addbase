"""Prepare the entire reserved official test without scoring or viewing labels.

Remote names never enter a model payload; all local media use opaque names.
Network attempts and decode failures are preserved. This is data preparation,
not permission to evaluate or tune on the reserved confirmation outcomes.
"""
import concurrent.futures
import hashlib
import json
import pathlib
import time
import urllib.parse
import urllib.request
import cv2
import numpy as np
from .common import BASE, create_json, append, read_rows, timestamp
from .robust_prepare import DEST, digest

OUT = DEST / 'external_roboreward_reserved_v1'


def fetch_video(name, revision):
    remote = pathlib.PurePosixPath(name)
    if remote.is_absolute() or '..' in remote.parts or remote.suffix != '.mp4':
        raise ValueError('Unexpected remote media path')
    opaque = hashlib.sha256(name.encode()).hexdigest()
    url = 'https://huggingface.co/datasets/teetone/RoboReward/resolve/' + revision + '/test/' + urllib.parse.quote(name, safe='/')
    attempts = []
    for attempt in range(1, 4):
        path = OUT / 'videos' / (opaque + f'_attempt{attempt}.mp4')
        if path.exists():
            raise FileExistsError('An attempt already exists; use its recorded status instead of overwriting')
        try:
            h = hashlib.sha256()
            size = 0
            with urllib.request.urlopen(url, timeout=60) as response, path.open('xb') as handle:
                while True:
                    chunk = response.read(1024*1024)
                    if not chunk:
                        break
                    handle.write(chunk)
                    h.update(chunk)
                    size += len(chunk)
            attempts.append(dict(attempt=attempt, status='ok', bytes=size))
            return dict(source_key=opaque, source_name=name, status='ok', path=str(path),
                        video_sha256=h.hexdigest(), bytes=size, attempts=attempts)
        except Exception as exc:
            attempts.append(dict(attempt=attempt, status='error', error=repr(exc)))
            time.sleep(attempt)
    return dict(source_key=opaque, source_name=name, status='error', attempts=attempts)


def decode_video(record):
    out = OUT / 'frames' / record['video_sha256']
    out.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(record['path'])
    try:
        if not cap.isOpened():
            raise ValueError('Video decode did not open')
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if count < 1 or not np.isfinite(fps) or fps <= 0:
            raise ValueError('Invalid frame count/fps')
        indices = np.linspace(0, count-1, 8).round().astype(int).tolist()
        wanted = set(indices)
        images, pixel_hashes = {}, {}
        seen = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if seen in wanted:
                path = out / f'frame_{seen:06d}.png'
                if path.exists():
                    raise FileExistsError(path)
                if not cv2.imwrite(str(path), frame):
                    raise IOError('Cannot save decoded frame')
                images[seen] = str(path)
                pixel_hashes[seen] = hashlib.sha256(frame.tobytes()).hexdigest()
            seen += 1
        if seen != count or set(images) != wanted:
            raise ValueError('Incomplete video decode')
        result = dict(status='ok', video_sha256=record['video_sha256'],
                      image_paths=[images[i] for i in indices], image_source_indices=indices,
                      image_sampling_record=dict(terminal_source_index=count-1,
                          selected_source_indices=indices, decoded_frame_count=count, source_fps=fps),
                      decoded_frame_pixel_sha256=[pixel_hashes[i] for i in indices])
        create_json(out / 'manifest_v1.json', result)
        return result
    except Exception as exc:
        return dict(status='error', video_sha256=record['video_sha256'], error=repr(exc))
    finally:
        cap.release()


def main():
    reservation = json.loads((OUT / 'reservation_v1.json').read_text())
    metadata = OUT / 'test_metadata_sealed.jsonl'
    if digest(metadata) != reservation['metadata_sha256']:
        raise ValueError('Reserved metadata changed')
    rows = list(read_rows(metadata))
    names = sorted({r['file_name'] for r in rows}, key=lambda name: hashlib.sha256(name.encode()).hexdigest())
    create_json(OUT / 'preparation_plan_v1.json', dict(time=timestamp(), source_script_sha256=digest(__file__),
        selected_samples='entire official test', selection_uses_reward=False,
        n_expected_rows=len(rows), n_video_paths=len(names), video_workers=6, decode_workers=4,
        inference_fields=['example_id', 'task', 'decoded pixels', 'label-free grounding'],
        label_and_check_fields_excluded_from_inference=True,
        media_failure_policy='retain expected denominator; no performance-based removal',
        overlap_policy='exact video SHA or exact eight decoded frames excluded from independent subset; all exclusions recorded',
        performance_evaluation_authorized_by_this_script=False))
    (OUT / 'videos').mkdir(exist_ok=True)
    videos = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        jobs = {pool.submit(fetch_video, name, reservation['revision']): name for name in names}
        for i, job in enumerate(concurrent.futures.as_completed(jobs), 1):
            result = job.result()
            videos[jobs[job]] = result
            append(OUT / 'download_records_v1.jsonl', dict(time=timestamp(), **result))
            if i % 50 == 0 or i == len(jobs):
                print(json.dumps(dict(time=timestamp(), downloaded=i, expected=len(jobs),
                    failures=sum(r['status'] != 'ok' for r in videos.values()))), flush=True)
    by_hash = {r['video_sha256']: r for r in videos.values() if r['status'] == 'ok'}
    decoded = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        jobs = [pool.submit(decode_video, row) for row in by_hash.values()]
        for i, job in enumerate(concurrent.futures.as_completed(jobs), 1):
            result = job.result()
            decoded[result['video_sha256']] = result
            append(OUT / 'decode_records_v1.jsonl', dict(time=timestamp(), **result))
            if i % 50 == 0 or i == len(jobs):
                print(json.dumps(dict(time=timestamp(), decoded=i, expected=len(jobs))), flush=True)
    old_rows = list(read_rows(BASE / 'inputs/full.jsonl'))
    old_by_hash = {row['video_sha256']: row for row in old_rows}
    old_pixels = set()
    for row in old_by_hash.values():
        values = []
        for path in row['image_paths']:
            image = cv2.imread(path)
            if image is None:
                raise ValueError('Old overlap reference unreadable')
            values.append(hashlib.sha256(image.tobytes()).hexdigest())
        old_pixels.add(tuple(values))
    overlap = set()
    for h, row in decoded.items():
        if h in old_by_hash or (row['status'] == 'ok' and tuple(row['decoded_frame_pixel_sha256']) in old_pixels):
            overlap.add(h)
    n_ready = 0
    # Labels are separated mechanically and never summarized or used for
    # selection, preparation, grounding, model execution, or output display.
    with (OUT / 'inputs_without_labels_v1.jsonl').open('x') as inputs, (OUT / 'labels_sealed_v1.jsonl').open('x') as labels:
        for index, row in enumerate(rows):
            eid = hashlib.sha256((str(index) + ':' + row['file_name'] + ':' + row['task']).encode()).hexdigest()
            video = videos[row['file_name']]
            h = video.get('video_sha256')
            frames = decoded.get(h, {})
            ready = video['status'] == 'ok' and frames.get('status') == 'ok'
            sample = dict(example_id=eid, task=row['task'], video_sha256=h,
                          video_path=video.get('path'), preparation_status='ok' if ready else 'error',
                          overlaps_old_data=h in overlap,
                          source_subset=row['file_name'].split('/')[0])
            if ready:
                sample.update({k: v for k, v in frames.items() if k not in ['status', 'video_sha256']})
                n_ready += 1
            inputs.write(json.dumps(sample, ensure_ascii=False) + '\n')
            labels.write(json.dumps(dict(example_id=eid, reward=row['reward']), ensure_ascii=False) + '\n')
    create_json(OUT / 'preparation_completion_v1.json', dict(time=timestamp(), expected_rows=len(rows),
        ready_rows=n_ready, unique_downloaded_video_hashes=len(by_hash),
        overlap_video_hashes=sorted(overlap), external_performance_not_evaluated=True,
        inputs_sha256=digest(OUT / 'inputs_without_labels_v1.jsonl'),
        labels_sealed_sha256=digest(OUT / 'labels_sealed_v1.jsonl')))


if __name__ == '__main__':
    main()
