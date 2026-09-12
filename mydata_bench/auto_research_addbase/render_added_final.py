"""Render all predeclared mainline-1 target points, with no selection."""
import argparse
import csv
import hashlib
import json
import pathlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--report-root', required=True)
    args = parser.parse_args()
    root = pathlib.Path(args.report_root)
    manifest = json.loads((root / 'source_manifest.json').read_text())
    assert manifest['completion_checks_passed']
    source = root / 'all_target_effects.csv'
    assert hashlib.sha256(source.read_bytes()).hexdigest() == manifest['outputs'][source.name]['sha256']
    with source.open() as f:
        rows = list(csv.DictReader(f))
    names = list(dict.fromkeys(r['experiment'] for r in rows))
    assert len(names) == 15 and len(rows) == 45
    by = {(r['experiment'], r['condition']): r for r in rows}
    fig, axes = plt.subplots(1, 4, figsize=(14, 10), sharey=True)
    configs = [('delta_mae', -1, 'MAE improvement\n(native - target)'),
               ('delta_accuracy_pp', 1, 'Total accuracy gain (pp)'),
               ('delta_suc_accuracy_pp', 1, 'Success accuracy gain (pp)'),
               ('delta_fail_accuracy_pp', 1, 'Failure accuracy gain (pp)')]
    for ax, (metric, sign, title) in zip(axes, configs):
        a = np.array([[sign*float(by[(name, 'target_k'+str(k))][metric]) for k in [8,32,64]] for name in names])
        limit = max(abs(a).max(), 0.01)
        im = ax.imshow(a, cmap='RdBu_r', vmin=-limit, vmax=limit, aspect='auto')
        ax.set_xticks([0,1,2], ['k8','k32','k64'])
        ax.set_title(title, fontsize=11, pad=12)
        ax.set_yticks(range(15))
        for y in [4.5, 9.5]:
            ax.axhline(y, color='black', linewidth=1)
        for i in range(15):
            for j in range(3):
                ax.text(j, i, f'{a[i,j]:.2f}', ha='center', va='center', fontsize=8,
                        color='white' if abs(a[i,j]) > .6*limit else 'black')
        fig.colorbar(im, ax=ax, orientation='horizontal', pad=.035, fraction=.035)
    labels = [name.replace('_attention_residual_v1',' [full]').replace('_terminal_residual_v1',' [last]').replace('_v1','') for name in names]
    axes[0].set_yticklabels(labels, fontsize=9)
    fig.suptitle('Added baseline attention: every target k on the complete grounded cohort\nRed = improvement; blue = worsening. Descriptive results, no confidence intervals.', fontsize=13, y=.97)
    fig.text(.02,.025,'MAE: Meter continuous ordinal; SOLE five-bin adapter. Accuracy denominator = all 846 expected examples.\nMAE uses valid outputs; coverage and native-format errors are listed in report.md. Full = seven-step recursion; last = terminal-step intervention.', fontsize=9)
    fig.subplots_adjust(left=.255, right=.985, bottom=.13, top=.87, wspace=.25)
    out = root / 'figures'
    out.mkdir(exist_ok=False)
    for ext in ['png','svg']:
        fig.savefig(out / ('all_target_effects.'+ext), dpi=180)
    plt.close(fig)
    with (out / 'provenance.json').open('x') as f:
        json.dump({'source':str(source.resolve()), 'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
                   'implementation_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),
                   'all_45_target_points':True}, f, indent=2)
    print(out)


if __name__ == '__main__':
    main()
