"""Group reserved test source episodes without reading rewards or model scores.

Source filenames can contain score suffixes. Only source subset, original split
and leading episode index (or RoboArena UUID) identify a group. Suffix content
is ignored, never displayed, and never added to model inputs.
"""
import collections
import hashlib
import json
import pathlib
import re

from .common import create_json, append, read_rows, timestamp
from .robust_external_data import OUT
from .robust_prepare import digest


def source_episode(name):
    path = pathlib.PurePosixPath(name)
    subset = path.parts[0]
    stem = path.stem
    match = re.match(r'(.+_originalsplit_[^_]+_index_\d+)(?=_|$)', stem)
    if match:
        return subset + '/' + match.group(1), 'source_split_leading_episode_index'
    if subset == 'robo_arena':
        match = re.match(r'([0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12})(?=_|$)', stem)
        if match:
            return subset + '/' + match.group(1).lower(), 'roboarena_uuid'
    return None, 'unparsed_source_identity'


def group_components(rows, episode_by_id):
    parent = {r['example_id']: r['example_id'] for r in rows}

    def find(key):
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    first = {}
    for row in rows:
        eid = row['example_id']
        keys = [('video_bytes', row['video_sha256']),
                ('displayed_pixels', tuple(row['decoded_frame_pixel_sha256']))]
        if episode_by_id[eid] is not None:
            keys.append(('source_episode', episode_by_id[eid]))
        for key in keys:
            if key in first:
                a, b = find(eid), find(first[key])
                parent[max(a, b)] = min(a, b)
            else:
                first[key] = eid
    return {eid: hashlib.sha256(('external_episode_group_v1:' + find(eid)).encode()).hexdigest()
            for eid in parent}


def main():
    input_path = OUT / 'inputs_without_labels_v4.jsonl'
    done = json.loads((OUT / 'preparation_completion_v4.json').read_text())
    if digest(input_path) != done['inputs_sha256']:
        raise ValueError('Reserved inputs changed')
    downloads = OUT / 'download_records_v1.jsonl'
    create_json(OUT / 'episode_grouping_plan_v1.json', dict(time=timestamp(),
        source_sha256=digest(__file__), inputs_sha256=digest(input_path),
        download_records_sha256=digest(downloads),
        rule='connected components of source subset/split/leading episode index or RoboArena UUID, video byte SHA, and all8 displayed pixel SHA',
        filename_suffix_policy='discard everything following leading episode index/UUID; never parse score/attempt suffix values',
        unknown_identity_policy='retain; group using exact video bytes/pixels and explicitly report unresolved source identity',
        purpose='future cluster uncertainty estimates, not input features, reward assignment, selection or exclusion',
        uncertainty='leading source index may coarsely merge distinct suffixes; unknown original provenance remains possible',
        reward_labels_read=False, reward_model_performance_read=False))
    source_keys = {}
    for record in read_rows(downloads):
        expected = hashlib.sha256(record['source_name'].encode()).hexdigest()
        if expected != record['source_key']:
            raise ValueError('Download source key mismatch')
        if expected in source_keys:
            raise ValueError('Duplicate download source identity')
        source_keys[expected] = source_episode(record['source_name'])
    rows = list(read_rows(input_path))
    if len(rows) != 2831 or len({r['example_id'] for r in rows}) != len(rows):
        raise ValueError('Incomplete or duplicate reserved inputs')
    episodes, methods = {}, {}
    for row in rows:
        local_key = pathlib.Path(row['video_path']).name[:64]
        if not re.fullmatch('[0-9a-f]{64}', local_key) or local_key not in source_keys:
            raise ValueError('Cannot join opaque local video to download provenance')
        episode, method = source_keys[local_key]
        episodes[row['example_id']] = episode
        methods[row['example_id']] = method
    groups = group_components(rows, episodes)
    path = OUT / 'episode_grouping_records_v1.jsonl'
    if path.exists():
        raise FileExistsError('Preserve existing grouping records')
    for row in rows:
        eid = row['example_id']
        episode = episodes[eid]
        append(path, dict(example_id=eid, video_sha256=row['video_sha256'],
            source_episode_sha256=hashlib.sha256(episode.encode()).hexdigest() if episode else None,
            episode_group_sha256=groups[eid], source_identity_method=methods[eid]))
    sizes = collections.Counter(groups.values())
    create_json(OUT / 'episode_grouping_audit_v1.json', dict(time=timestamp(),
        examples=len(rows), unique_video_bytes=len({r['video_sha256'] for r in rows}),
        source_episode_identities=len({v for v in episodes.values() if v is not None}),
        connected_episode_groups=len(sizes), largest_group=max(sizes.values()),
        group_size_histogram=dict(sorted(collections.Counter(sizes.values()).items())),
        source_identity_methods=dict(collections.Counter(methods.values())),
        unresolved_source_identity_ids=[eid for eid, episode in episodes.items() if episode is None],
        no_samples_excluded=True, model_inputs_unchanged=True,
        records_sha256=digest(path), labels_read=False, model_scores_read=False,
        interpretation='conservative source-episode grouping for within-test dependence; not proof of train/test independence or absence of training contamination'))
    print(json.dumps(dict(examples=len(rows), groups=len(sizes), largest_group=max(sizes.values()),
        unresolved=sum(e is None for e in episodes.values()), source_methods=dict(collections.Counter(methods.values())))))


if __name__ == '__main__':
    main()
