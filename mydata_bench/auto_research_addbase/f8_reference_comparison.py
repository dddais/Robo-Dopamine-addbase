"""Load prospectively fixed F7 CE comparisons without changing F8 inference."""
import json
import pathlib
import yaml
from .common import read_rows
from .robust_prepare import digest


def f7_primary_rows(policy, case, stage, current_config, ids):
    item = policy['fixed_F7_comparator_configs'][stage][case]
    if digest(item['path']) != item['sha256']:
        raise ValueError('Prospectively frozen F7 comparator configuration changed')
    cfg = yaml.safe_load(pathlib.Path(item['path']).read_text())
    for field in ['model', 'protocol', 'model_path', 'processor_path', 'input_sha256', 'example_ids_file']:
        if cfg.get(field) != current_config.get(field):
            raise ValueError('F7/F8 comparison mismatch: '+field)
    out = pathlib.Path(cfg['output_dir'])/stage
    done = json.loads((out/'completion_v1.json').read_text())
    predictions = out/'predictions.jsonl'
    if digest(predictions) != done['predictions_sha256'] or json.loads((out/'config.json').read_text()) != cfg:
        raise ValueError('F7 completed comparator provenance mismatch')
    rows = list(read_rows(predictions))
    keys = [(r['example_id'], r['condition']) for r in rows]
    expected = {(eid, c['name']) for eid in ids for c in cfg['conditions']}
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError('F7 comparator is incomplete or duplicated')
    # Invalid predictions remain in the paired full denominator.
    return [r for r in rows if r['condition'] == 'reward_gradient_k48']
