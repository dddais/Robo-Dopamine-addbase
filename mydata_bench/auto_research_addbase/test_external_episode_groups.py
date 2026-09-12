"""Source variants and identical media must not inflate independent sample N."""
from mydata_bench.auto_research_addbase.robust_external_episode_groups import source_episode, group_components


def test_source_suffix_is_never_an_episode_or_reward_feature():
    base, method = source_episode('demo/demo_originalsplit_train_index_123.mp4')
    for suffix in ['_7', '_attempt_9_score_3', '_attempt_2_score_5']:
        assert source_episode('demo/demo_originalsplit_train_index_123' + suffix + '.mp4') == (base, method)
    assert source_episode('demo/demo_originalsplit_test_index_123.mp4')[0] != base
    assert source_episode('demo/demo_originalsplit_train_index_1234.mp4')[0] != base
    uuid = 'cbd9281c-e5d0-46cd-9836-b225f310128e'
    assert source_episode('robo_arena/' + uuid + '_F_right.mp4')[0] == source_episode('robo_arena/' + uuid + '_F_left.mp4')[0]
    assert source_episode('unknown/opaque.mp4')[0] is None


def test_union_is_transitive_and_order_independent():
    rows = [dict(example_id=n, video_sha256=b, decoded_frame_pixel_sha256=[p]*8)
            for n, b, p in [('a', 'v1', 'p1'), ('b', 'v2', 'p2'), ('c', 'v2', 'p3'), ('d', 'v4', 'p4')]]
    episodes = {'a': 'ep1', 'b': 'ep1', 'c': 'ep2', 'd': None}
    groups = group_components(rows, episodes)
    assert groups['a'] == groups['b'] == groups['c']
    assert groups['d'] != groups['a']
    assert groups == group_components(list(reversed(rows)), episodes)
