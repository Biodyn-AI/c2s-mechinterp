#!/usr/bin/env python3
"""Independently check E4 presets and inactive-feature differences from saved counts."""
import argparse
import json
from pathlib import Path

import numpy as np

from audit_experimental_saved import Audit


def count_contrasts(original, projected):
    n = original['n_features']
    if projected['n_features'] != n: raise ValueError('E4 dictionary capacities differ')
    return {'dead_fraction_heldout_original': original['dead_features']/n,
            'dead_fraction_heldout_projected': projected['dead_features']/n,
            'dead_fraction_heldout_delta': (projected['dead_features']-original['dead_features'])/n,
            'dead_fraction_all_rows_original': original['dead_all_rows']/n,
            'dead_fraction_all_rows_projected': projected['dead_all_rows']/n,
            'dead_fraction_all_rows_delta': (projected['dead_all_rows']-original['dead_all_rows'])/n}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(); audit = Audit()
    saved = audit.read(args.root/'results/dropmassive_comparison.json')
    statistics_path = args.root/'results/scale-extract_residual_stats.npz'; audit.hash(statistics_path)
    records = []
    with np.load(statistics_path) as spectra:
        for record in saved['layers']:
            layer = record['layer']; audit.context = f'E4/L{layer}: '
            run = args.root/'runs/27b_matched'
            original = audit.read(run/'sae'/f'train_L{layer:02d}.json')
            projected = audit.read(run/'sae_dropmassive'/f'train_L{layer:02d}.json')
            preset_path = args.root/'results/dropmassive_dimensions'/f'dimensions_L{layer:02d}.json'
            preset = audit.read(preset_path)
            rms = spectra[f'rms_L{layer:02d}']; selected = np.flatnonzero(rms > 10*np.median(rms)).tolist()
            audit.equal('prespecified strict RMS dimension selection', preset['indices'], selected)
            audit.equal('RMS preset source SHA', preset['source_stats_sha256'], audit.sources[str(statistics_path)])
            if not selected:
                audit.equal('empty selection explicitly skipped', record['status'], 'skipped_no_dimensions')
                records.append({'layer': layer, 'status': 'skipped'}); continue
            for key in ['n_features', 'k', 'epochs', 'seed', 'n_rows', 'd_model', 'batch_size', 'lr']:
                audit.equal('matched recipe '+key, projected[key], original[key])
            audit.equal('zero projection dimensions', projected['input_projection']['removed_dimensions'], selected)
            audit.equal('projection preset checksum', projected['input_projection']['preset_sha256'], audit.sources[str(preset_path)])
            audit.equal('projection variance reference', projected['input_projection']['variance_reference'], 'projected input after zeroing selected columns')
            values = count_contrasts(original, projected)
            for key, value in values.items(): audit.equal('reported '+key, record[key], value)
            for side, metrics in [('original', original), ('projected', projected)]:
                audit.equal(side+' heldout fraction/count', metrics['dead_frac'], metrics['dead_features']/metrics['n_features'])
                audit.equal(side+' all-row fraction/count', metrics['dead_frac_all_rows'], metrics['dead_all_rows']/metrics['n_features'])
            audit.equal('original input VE copy', record['variance_explained_original_input'], original['variance_explained'])
            audit.equal('projected input VE copy', record['variance_explained_projected_input'], projected['variance_explained'])
            mean_path = run/'sae_dropmassive'/f'mu_L{layer:02d}.npy'; audit.hash(mean_path)
            mean = np.load(mean_path).reshape(-1)
            audit.check('zeroed input coordinates have exactly zero saved mean', np.all(mean[selected] == 0))
            records.append({'layer': layer, 'n_removed_dimensions': len(selected), **values,
                            'variance_reference_note': 'Original and projected VE are retained separately; no VE difference is interpreted.'})
    for path, checksum in saved['source_sha256'].items():
        if path not in audit.sources: audit.hash(path)
        audit.equal('producer source checksum '+path, audit.sources[path], checksum)
    audit.hash(Path(__file__))
    report = {'audit': 'Independent E4 inactive-feature and RMS-preset saved-data audit', 'passed': audit.passed,
              'n_checks': audit.n_checks, 'findings': audit.failed, 'layers': records, 'source_sha256': audit.sources,
              'scope': 'Same capacity/training recipe, projected inputs. No reconstruction forwards, raw activation writes or model fits are performed.'}
    args.out.parent.mkdir(parents=True, exist_ok=True); args.out.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({key: report[key] for key in ['passed', 'n_checks', 'findings', 'layers']}))
    raise SystemExit(0 if audit.passed else 1)


if __name__ == '__main__': main()
