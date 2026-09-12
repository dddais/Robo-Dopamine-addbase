"""Freeze episode-separated training data for prospective supervised head ranking.

Never open reserved test labels. Test source identities are used only to
exclude source-episode overlap, and remote filenames never become model text.
"""
import collections
import hashlib
import json
import pathlib
from .common import create_json, read_rows, timestamp
from .robust_external_episode_groups import source_episode
from .robust_prepare import DEST, digest

OUT = DEST / 'external_training_research_v1'


def key(text):
    return hashlib.sha256(('robust-head-training-v1:' + text).encode()).hexdigest()


def main():
    path = OUT / 'train_metadata_v1.jsonl'
    provenance = json.loads((OUT / 'training_metadata_provenance_v1.json').read_text())
    if digest(path) != provenance['sha256']:
        raise ValueError('Training source changed')
    test_path = DEST / 'external_roboreward_reserved_v1/download_records_v1.jsonl'
    test_episodes = {source_episode(r['source_name'])[0] for r in read_rows(test_path)}
    test_episodes.discard(None)
    groups = collections.defaultdict(list)
    exclusions = collections.Counter()
    for row in read_rows(path):
        episode, method = source_episode(row['file_name'])
        if episode is None:
            exclusions['unresolved_source_episode'] += 1
        elif episode in test_episodes:
            exclusions['reserved_test_source_episode'] += 1
        else:
            if row['reward'] not in [1, 2, 3, 4, 5]:
                raise ValueError('Invalid official training grade')
            groups[episode].append(row)
    subsets = collections.defaultdict(list)
    for episode in groups:
        subsets[episode.split('/')[0]].append(episode)
    assignments = {}
    # Fixed source-balanced sampling; no model outputs and no label stratification.
    for subset, episodes in sorted(subsets.items()):
        selected = sorted(episodes, key=key)[:20]
        n_validation = max(1, len(selected)//5)
        for index, episode in enumerate(selected):
            assignments[episode] = 'validation' if index < n_validation else 'fit'
    records, labels = [], []
    for episode in sorted(assignments, key=key):
        for index, row in enumerate(sorted(groups[episode], key=lambda r: key(r['file_name'] + '\n' + r['task']))):
            eid = key(episode + '\n' + str(index) + '\n' + row['file_name'] + '\n' + row['task'])
            records.append(dict(example_id=eid, source_name=row['file_name'],
                source_key=hashlib.sha256(row['file_name'].encode()).hexdigest(), task=row['task'],
                source_subset=episode.split('/')[0], source_episode_sha256=key(episode),
                training_partition=assignments[episode]))
            labels.append(dict(example_id=eid, reward=row['reward'], training_partition=assignments[episode]))
    if len({r['example_id'] for r in records}) != len(records):
        raise ValueError('Duplicate training IDs')
    create_json(OUT / 'training_selection_plan_v1.json', dict(time=timestamp(),
        source_sha256=digest(__file__), metadata_sha256=digest(path), reserved_test_downloads_sha256=digest(test_path),
        selected_source_episodes=len(assignments), selected_examples=len(records),
        per_source='first20 episodes by fixed salted SHA; first floor(n/5), minimum1, validation; remaining fit; retain all rows per episode',
        selection_uses_reward=False, selection_uses_model_performance=False,
        selected_episode_sha256={key(e): split for e, split in assignments.items()},
        training_grade_policy='all official1–5 ratings retained; never replace partial progress by endpoints',
        exclusions=dict(exclusions), reserved_test_labels_read=False,
        additional_media_audit_required='video bytes and sparse pixels against reserved test and old data; cross-partition duplicate components must be quarantined, never chosen by performance',
        training_fits_heads_only=True, held_out_training_validation_not_used_for_head_ranking=True))
    for name, rows in [('selected_training_metadata_v1.jsonl', records), ('training_labels_v1.jsonl', labels)]:
        with (OUT / name).open('x') as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False)+'\n')
    create_json(OUT / 'training_selection_completion_v1.json', dict(time=timestamp(),
        selected_examples=len(records), selected_episodes=len(assignments),
        example_partitions=dict(collections.Counter(r['training_partition'] for r in records)),
        episode_partitions=dict(collections.Counter(assignments.values())),
        source_subsets=len(subsets), source_overlap_with_reserved_test=0,
        selected_metadata_sha256=digest(OUT/'selected_training_metadata_v1.jsonl'),
        training_labels_sha256=digest(OUT/'training_labels_v1.jsonl'),
        training_labels_by_partition={part:dict(collections.Counter(r['reward'] for r in labels if r['training_partition']==part)) for part in ['fit','validation']},
        test_labels_and_test_model_performance_not_read=True))
    print('Training source frozen:',len(records),'examples;',len(assignments),'source episodes;',dict(collections.Counter(assignments.values())))


if __name__ == '__main__':
    main()
