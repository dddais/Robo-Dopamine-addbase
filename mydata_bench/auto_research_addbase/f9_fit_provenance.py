"""Validate F9 rankings against their original audited F8 gradient files."""
import json
import pathlib
from .robust_prepare import digest


def verify_fit_artifacts(done_path, config, expected_f8_gradient_sha, expected_config_sha):
    done_path=pathlib.Path(done_path)
    done=json.loads(done_path.read_text())
    if done['examples']!=1942 or done['fitting_episode_groups']!=446:
        raise ValueError('Full1942 fitting samples and446 episode groups required')
    if done['config_sha256']!=expected_config_sha:
        raise ValueError('F9 completion does not match frozen fitting configuration')
    ranking=done_path.parent/'ranking_v1.json'
    if digest(ranking)!=done['ranking_sha256']:
        raise ValueError('Frozen F9 ranking changed')
    source=pathlib.Path(done['source_gradients_file'])
    if str(source)!=config['source_gradients_file']:
        raise ValueError('F9 gradient source path mismatch')
    hashes=[done['source_gradients_sha256'],config['source_gradients_sha256'],expected_f8_gradient_sha]
    if any(digest(source)!=sha for sha in hashes):
        raise ValueError('F9 source gradients do not match the completed F8 audit')
    rank=json.loads(ranking.read_text())
    if rank['source_F8_gradients_file']!=str(source) or rank['source_F8_gradients_sha256']!=expected_f8_gradient_sha:
        raise ValueError('F9 ranking names a different original gradient source')
    for field in ['training_input','fit_ids','fit_labels']:
        if digest(config[field+'_file'])!=config[field+'_sha256']:
            raise ValueError('Frozen fitting dependency changed: '+field)
    return done
