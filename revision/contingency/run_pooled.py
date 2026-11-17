"""Run the activated C6 pooled cohort on the existing compute pod.

This wrapper only records the unchanged extraction/statistics commands as queue stages.
It never creates resources. The fixed input cohort is prepared before outcomes are inspected.
"""
import argparse
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    scripts = root / 'scripts'
    sys.path.insert(0, str(scripts))
    from run_revision import Stage, execute_stage

    cohort = root / 'data/papalexi/c6_pooled_rep1tx'
    output = root / 'results/phase8_thp1_pooled'
    atlas = root / 'runs/2b_atlas'
    common = ['--code-dir', root / 'code', '--run', atlas,
              '--model', 'vandijklab/C2S-Scale-Gemma-2-2B',
              '--trrust', root / 'ontologies_v2/trrust_human.tsv',
              '--design', cohort / 'design.json', '--layers', 8, 16, 22, '--out', output]
    stages = [
        Stage('phase8-thp1-pooled-extract',
              [sys.executable, scripts / 'rev_phase8_extract.py', *common],
              [output / 'rows.json', output / 'meta.json',
               *[output / f'feat_L{layer:02d}.npz' for layer in (8, 16, 22)]]),
        Stage('phase8-thp1-pooled-stats',
              [sys.executable, scripts / 'rev_phase8_stats.py',
               '--features-dir', output, '--design', cohort / 'design.json',
               '--catalog-dir', atlas, '--h5ad', cohort / 'counts_cleaned.h5ad',
               '--trrust', root / 'ontologies_v2/trrust_human.tsv',
               '--layers', 8, 16, 22, '--out', output / 'statistics'],
              [output / 'statistics/regulatory_statistics.json', output / 'statistics/manifest.json']),
        Stage('phase8-thp1-pooled-biology',
              [sys.executable, scripts / 'rev_response_biology.py',
               '--stats-dir', output / 'statistics', '--annotations-dir', root / 'results/annotations/2b',
               '--catalog-dir', atlas, '--trrust', root / 'ontologies_v2/trrust_human.tsv',
               '--layers', 8, 16, 22, '--out', output / 'statistics/response_biology.json'],
              [output / 'statistics/response_biology.json']),
        Stage('phase8-thp1-pooled-guide-concordance',
              [sys.executable, scripts / 'rev_guide_concordance.py',
               '--h5ad', cohort / 'counts_cleaned.h5ad', '--design', cohort / 'design.json',
               '--features-dir', output, '--catalog-dir', atlas,
               '--trrust', root / 'ontologies_v2/trrust_human.tsv', '--layers', 8, 16, 22,
               '--out', output / 'statistics/guide_concordance.json'],
              [output / 'statistics/guide_concordance.json']),
    ]
    for stage in stages:
        execute_stage(stage, root)


if __name__ == '__main__':
    main()
