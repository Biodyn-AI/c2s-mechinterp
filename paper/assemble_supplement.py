#!/usr/bin/env python3
"""Assemble small supplementary LaTeX fragments from saved artifacts only.

Run --draft to expose pending inputs. Without --draft every block, all S1--S6 PDFs,
and the figure provenance must be present and valid before any output is written.
This helper reads summaries/sidecars; it never fits a model or opens activation arrays.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re


def tex_escape(value):
    escapes = {'\\': r'\textbackslash{}', '&': r'\&', '%': r'\%', '$': r'\$',
               '#': r'\#', '_': r'\_', '{': r'\{', '}': r'\}',
               '~': r'\textasciitilde{}', '^': r'\textasciicircum{}'}
    return ''.join(escapes.get(character, character) for character in str(value))


def number(value):
    if value is None:
        return 'not evaluable'
    if not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f'Nonfinite or nonnumeric measurement: {value!r}')
    return f'{value:,}'.replace(',', ' ') if isinstance(value, int) else f'{value:.6g}'


def measurement(value):
    """Required measurements must never turn a missing field into final table prose."""
    if value is None:
        raise ValueError('Required saved measurement is missing')
    return number(value)


def require_source_hashes(record):
    hashes = record['source_sha256']
    if not hashes or any(not re.fullmatch(r'[0-9a-f]{64}', value) for value in hashes.values()):
        raise ValueError('Nonempty saved source SHA256 records are required')


def saved_interval(values):
    if len(values) != 2 or values[0] > values[1]:
        raise ValueError('Saved interval endpoints must be a pair in increasing order')
    return ' to '.join(number(value) for value in values)


def table_text(value):
    text = tex_escape(value).replace(r'\_', r'\_\allowbreak{}')
    # Keep a grouped integer and each scientific-notation token on one line;
    # an interval can still wrap between its two endpoints.
    text = re.sub(r'(?<=\d) (?=\d{3}(?:\D|$))', '~', text)
    return re.sub(r'(?<!\w)-?\d+(?:\.\d+)?e[+-]\d+\b', lambda match: r'\mbox{' + match[0] + '}', text)


def table(caption, headers, rows, widths):
    columns = '@{}' + ''.join(f'L{{{width}\\linewidth}}' for width in widths) + '@{}'
    heading = ' & '.join(table_text(header) for header in headers) + r' \\'
    body = '\n'.join(' & '.join(table_text(cell) for cell in row) + r' \\' for row in rows)
    return (f'\\begin{{longtable}}{{{columns}}}\n\\caption{{{table_text(caption)}}}\\\\\n'
            f'\\toprule\n{heading}\n\\midrule\n\\endfirsthead\n'
            f'\\toprule\n{heading}\n\\midrule\n\\endhead\n'
            '\\endfoot\n\\bottomrule\n\\endlastfoot\n'
            f'{body}\n\\end{{longtable}}\n')


FIGURE_UNITS = {
    '2': ('a: variance explained; b: inactive fraction; c: ontology rate/floor; d: decoder matching', 'dictionary features per layer; held-out versus all positions; target layers'),
    '3': ('a: category coverage; b: named features; c--e: line-specific firing', 'dictionary features and gene-token positions; example IDs saved in sidecar'),
    '4': ('a: stored edges by layer pair; b: shared-term baseline; c/e: K562/RPE1 specificity; d/f: K562 gene/feature AP',
          'stored capped graph edges; source-feature bootstrap for shared terms; screened TF families and TF bootstrap for retrieval'),
    '5': ('a: inactive features; b: annotation; c: covariance PR; d: dimension RMS', 'matched dictionary counts and relative depth; residual sampling in source metadata'),
    '6': ('a: functional geometry; b: steering path; c: frozen transfer; d: raw-coordinate PR', 'genes/permutations, paired source cells, target cells, gene vectors respectively'),
    'S1': ('a--f: reconstruction, firing, decoder overlap, annotation excess, expansion', 'each saved layer/sparsity/expansion setting; no cross-run confidence interval'),
    'S2': ('a: gene-similarity correlation; b: reconstruction; c: annotation/floor', 'same sentences and dictionary at each pooling rule'),
    'S3': ('a: database-library sensitivity; b: untrained dictionary comparison', 'features at each selected layer; same full test-family scorer'),
    'S4': ('a: edge overlap; b: conditional edge-rule sensitivity', 'same source-feature set; saved cell counts; simulated normal deltas for sensitivity'),
    'S5': ('a: covariance PR; b: TwoNN dimension', 'genes as rows; coordinate count and valid neighbor pairs in saved dimension rows'),
    'S6': ('a: wall time; b: device memory; c: host memory', 'twelve longest completed executable stages; device-wide peak, own process-tree RSS'),
}

PERTURBATION_COHORTS = (('k562', 'K562'), ('rpe1', 'RPE1'), ('thp1_pooled', 'THP1'))


class Assembler:
    def __init__(self, paper, results, figures, logs, versions, accounting):
        self.paper, self.results, self.figures, self.logs = map(Path, (paper, results, figures, logs))
        self.versions, self.accounting = Path(versions), Path(accounting)
        self.hashes = {}

    def read(self, path):
        path = Path(path)
        if path.stat().st_size > 32 * 1024 * 1024:
            raise ValueError(f'Supplement assembly accepts small summaries only: {path}')
        raw = path.read_bytes()
        payload = json.loads(raw, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
        if isinstance(payload, dict) and (payload.get('synthetic') or payload.get('synthetic_preview')):
            raise ValueError(f'Refusing synthetic artifact: {path}')
        self.hashes[str(path)] = hashlib.sha256(raw).hexdigest()
        return payload

    def figure(self, name):
        pdf = self.figures / f'Figure_{name}.pdf'
        with pdf.open('rb') as handle:
            if handle.read(5) != b'%PDF-':
                raise ValueError(f'Invalid figure PDF header: {pdf}')
        payload = self.read(self.figures / f'Figure_{name}.provenance.json')
        if payload['figure'] != name or not payload['plotted_values'] or not payload['source_sha256']:
            raise ValueError(f'Incomplete figure provenance: {name}')
        generator = self.paper / 'make_revision_figures.py'
        if payload['generator_sha256'] != hashlib.sha256(generator.read_bytes()).hexdigest():
            raise ValueError(f'Figure {name} generator hash differs from current source; render it from the current generator')
        self.hashes[str(pdf)] = hashlib.sha256(pdf.read_bytes()).hexdigest()
        return payload

    def tokenization(self):
        record = self.read(self.results / 'tokenization.json')
        rows = [('Cells', record['n_cells']), ('Gene occurrences', record['n_gene_occurrences']),
                ('Unique symbols', record['symbol_tokens']['n']),
                ('Tokens per occurrence, mean', record['occurrence_tokens']['mean']),
                ('Single-token occurrences, fraction', record['occurrence_tokens']['share_single_token']),
                ('Tokens per symbol, equal-symbol mean', record['symbol_tokens']['mean_tokens_equal_symbol_weight']),
                ('Always single-token symbols, fraction', record['symbol_tokens']['share_always_single_token']),
                ('Tokens per sentence, mean', record['sentence_tokens']['mean']),
                ('Truncated sentences', record['n_truncated_sentences'])]
        return table('Input tokenisation measurements. Occurrence- and symbol-weighted quantities have different denominators.',
                     ['Quantity', 'Saved value'], [(label, number(value)) for label, value in rows], [.63, .27])

    def databases(self):
        versions = self.read(self.versions)
        provenance = self.read(self.results / 'ontology_provenance.json')
        rows = []
        for source, record in versions['sources'].items():
            release = record.get('library', record.get('version'))
            count = record.get('n_terms', record.get('n_pairs', record.get('n_rows')))
            if release is None or count is None:
                raise KeyError(f'{source}: release and term/pair/row count')
            detail = f"info mapping {record['info_version']}; score >= {record['min_combined_score']}" if source == 'string' else 'source gene sets or relations'
            rows.append((source, release, number(count), detail))
        text = table('Database snapshots before restriction to the sentence-gene universe. Source URLs and checksums are retained in VERSIONS.json and ontology_provenance.json.',
                     ['Source', 'Library / release', 'Entries', 'Scope'], rows, [.13, .31, .12, .31])
        return text + '\\noindent Archived library provenance: ' + tex_escape(provenance['legacy_description']) + '\n\n'

    def software(self):
        environment = self.read(self.results / 'environment.json')
        rows = [('Python', environment['python']), ('CUDA', environment['cuda']), ('GPU', environment['gpu'])]
        rows += list(environment['packages'].items())
        return table('Software and device recorded in the compute environment.', ['Component', 'Version / device'], rows, [.26, .64])

    def regulatory(self):
        specificity_rows, ap_rows, auroc_rows = [], [], []
        for line, label in PERTURBATION_COHORTS:
            record = self.read(self.results / f'phase8_{line}/statistics/regulatory_statistics.json')
            for family in record['summary']:
                layer = '--' if family['layer'] is None else family['layer']
                specificity_rows.append((label, family['representation'], layer, f"{family['n_specific']}/{family['n_tfs']}"))
                levels = ['gene'] + (['feature'] if family['representation'] == 'model' else [])
                for level in levels:
                    retrieval = family[f'{level}_retrieval']
                    readout = family['representation'] + (f' L{layer}' if family['layer'] is not None else '') + f' / {level}'
                    ap_interval = retrieval['mean_average_precision_ci95']
                    auroc_interval = retrieval['mean_auroc_ci95']
                    ap_rows.append((label, readout, retrieval['n_tfs_evaluable'], number(retrieval['mean_average_precision']),
                        'not evaluable' if ap_interval is None else saved_interval(ap_interval),
                        number(retrieval['mean_prevalence']), number(retrieval['mean_random_label_expected_ap'])))
                    auroc_rows.append((label, readout, retrieval['n_tfs_evaluable'], number(retrieval['mean_auroc']),
                        'not evaluable' if auroc_interval is None else saved_interval(auroc_interval)))
        text = table('Unsigned target-set specificity. The denominator is the full screened TF family, including TFs with sparse or absent eligible targets. THP1 (pooled guides) groups cells by perturbation target.',
                     ['Line', 'Arm', 'Layer', 'Specific / tested TFs'], specificity_rows, [.12, .28, .12, .32])
        text += table('Gene- and feature-level target retrieval by average precision (AP). The TF count includes only evaluable factors. Target prevalence and the exact finite random-label AP expectation are distinct descriptive reference quantities. Intervals resample TFs, not cells.',
                      ['Line', 'Readout / level', 'TFs', 'Mean AP', '95% interval', 'Target prevalence', 'Exact random-label expected AP'],
                      ap_rows, [.06, .16, .09, .13, .15, .13, .13])
        text += table('Target retrieval by AUROC for the same evaluable TF families. Intervals resample evaluable TFs.',
                      ['Line', 'Readout / level', 'TFs', 'Mean AUROC', '95% interval'], auroc_rows, [.07, .31, .09, .15, .24])
        return text

    def reconstruction(self):
        rows = []
        for line, label in PERTURBATION_COHORTS:
            metadata = self.read(self.results / f'phase8_{line}/meta.json')
            records = metadata['reconstruction_by_group_layer']
            for layer in sorted({record['layer'] for record in records}):
                groups = [record for record in records if record['layer'] == layer]
                ve = [record['variance_explained'] for record in groups]
                dead = [record['dead_fraction'] for record in groups]
                rows.append((label, layer, number(metadata['n_cells']), number(len(groups)),
                             number(min(ve)) + ' to ' + number(max(ve)), number(min(dead)) + ' to ' + number(max(dead))))
        return table('Applying the K562 dictionaries to perturbation cohorts, including THP1 (pooled guides). Ranges are minima and maxima across saved perturbation/control groups, without pooling their denominators.',
                     ['Line', 'Layer', 'Cells', 'Groups', 'Variance explained range', 'Inactive fraction range'], rows, [.10, .07, .09, .09, .26, .26])

    def text_deletion(self):
        report = self.read(self.results / 'text_deletion_reporting.json')
        if report['schema'] != 'text-name-deletion-report-v1' or report['status'] != 'complete':
            raise ValueError('Complete text-name deletion reporting artifact required')
        rows, reference_rows, metadata_rows = [], [], []
        for line, label in (('k562', 'K562'), ('rpe1', 'RPE1')):
            relative = f'phase8_{line}/statistics/insilico_retrieval.json'
            source = self.read(self.results / relative)
            if self.hashes[str(self.results / relative)] != report['source_sha256'][relative]:
                raise ValueError(f'Text-deletion source hash differs: {relative}')
            entry = report['lines'][line]
            if entry['n_screened_named_tfs'] != len(source['meta']['tfs']) or entry['n_pairs'] != source['meta']['n_pairs']:
                raise ValueError(f'Text-deletion coverage differs from saved source: {line}')
            metadata_rows.append((label, number(entry['n_screened_named_tfs']), number(entry['n_pairs']),
                number(entry['n_named_gene_universe']), number(entry['max_pre_deletion_cosdist_mean']),
                number(entry['max_pre_deletion_cosdist_max'])))
            for layer in entry['layers']:
                def estimate(metric):
                    return number(layer[f'mean_{metric}']) + ' (' + saved_interval(layer[f'mean_{metric}_ci95']) + ')'
                rows.append((label, layer['layer'], f"{layer['n_tfs_evaluable']}/{layer['n_tfs_screened']}",
                             estimate('average_precision'), estimate('auroc')))
                reference_rows.append((label, layer['layer'], estimate('prevalence'), estimate('random_label_expected_ap'),
                    number(layer['evaluable_downstream_genes_min']) + ' to ' + number(layer['evaluable_downstream_genes_max'])))
        text = table(f"Text-name deletion target retrieval. Points are unweighted means across evaluable factors; parentheses show 95% intervals from {number(report['bootstrap'])} TF bootstrap draws with per-factor retrieval fixed. Screened TF names come from control sentences and differ from the measured-knockdown family. This is a model-text assay, not biological regulatory validation.",
                     ['Line', 'Layer', 'Evaluable / named TFs', 'Mean AP (95% interval)', 'Mean AUROC (95% interval)'],
                     rows, [.08, .06, .13, .29, .29])
        text += table('Text-deletion retrieval references and downstream coverage. Each reference is an unweighted mean over the same evaluable factors, with a TF bootstrap interval. Finite random-label expected AP preserves the saved score ties and is distinct from prevalence. Downstream gene universes differ across factors and cell lines. The nearby deletion control is not matched for subword token length.',
                      ['Line', 'Layer', 'Target prevalence (95% interval)', 'Finite expected AP (95% interval)', 'Downstream genes, range'],
                      reference_rows, [.08, .06, .27, .29, .15])
        text += table('Text-deletion cohort and numerical noise. Pairs are TF-cell pairs. The final columns are the mean and maximum of per-pair maximum pre-deletion cosine distances at the first sampled layer (layer eight), restricted to deletions after rank five. They measure numerical agreement before the deleted name, not a biological response.',
                      ['Line', 'Named TFs', 'TF-cell pairs', 'Named gene universe', 'Mean pre-deletion maximum', 'Maximum pre-deletion distance'],
                      metadata_rows, [.08, .10, .13, .14, .19, .20])
        return text

    def surface_forms(self):
        record = self.read(self.results / 'surface_form_retrieval.json')
        labels = [('human', 'Human symbols'), ('nearmiss', 'Near-miss strings'), ('lower', 'Lowercase symbols'),
                  ('mouse', 'Mouse-style symbols'), ('anagram', 'Anagrams')]
        rows = []
        for key, label in labels:
            form = record['forms'][key]
            rows.append((label, f"{form['n_positive']}/{form['n']}",
                number(form['average_precision']) + ' (' + saved_interval(form['average_precision_ci95']) + ')',
                number(form['auroc']) + ' (' + saved_interval(form['auroc_ci95']) + ')',
                number(form['prevalence']), number(form['random_label_expected_ap'])))
        return table('Name-only phase retrieval. The positive class is G2/M; all surface forms use the same saved gene identities and labels. AP denotes stepwise average precision. Intervals are saved stratified gene-item bootstrap intervals and do not account for dependence within gene families. Prevalence and exact finite random-label expected AP are separate descriptive references.',
                     ['Surface form', 'G2/M / genes', 'AP (95% interval)', 'AUROC (95% interval)', 'Prevalence', 'Finite expected AP'],
                     rows, [.15, .09, .20, .20, .11, .11])

    def dimensions(self):
        rows, cohort_rows = [], []
        representations = ['expression_log1p_CP10k', 'expression_cells_shuffled_within_gene',
                           'expression_shape_iid_normal']
        representations += [f'model_L{layer:02d}{suffix}' for layer in (9, 13, 17)
                            for suffix in ('', '_shape_iid_normal')]
        labels = dict(zip(representations[:3], ['Log1p counts per 10 000', 'Within-gene cell shuffle',
                                               'Expression-shape Gaussian']))
        labels.update({f'model_L{layer:02d}{suffix}': f'Model L{layer}' + (' Gaussian' if suffix else '')
                       for layer in (9, 13, 17) for suffix in ('', '_shape_iid_normal')})
        for cohort, filename in [('CD34+', 'cd34_dimensions.json'), ('K562 matched-cell replication', 'k562_replication_dimensions.json')]:
            payload = self.read(self.results / filename)
            genes, cells, records = payload['genes'], payload['cell_indices'], payload['rows']
            if not genes or not cells or len(set(genes)) != len(genes) or len(set(cells)) != len(cells):
                raise ValueError('Nonempty, unique common gene identities and cell indices are required')
            if len(records) != len(representations) or {row['representation'] for row in records} != set(representations):
                raise ValueError('Complete expression/model and shape-control dimension rows are required')
            by_name = {row['representation']: row for row in records}
            for layer in (9, 13, 17):
                name = f'model_L{layer:02d}'
                if by_name[name]['n_coordinates'] != by_name[name+'_shape_iid_normal']['n_coordinates']:
                    raise ValueError('Model and Gaussian control coordinate shapes must match')
            cohort_rows.append((cohort, number(len(cells)), number(len(genes))))
            for record in records:
                if (not isinstance(record['n_coordinates'], int) or
                        isinstance(record['n_coordinates'], bool) or record['n_coordinates'] < 2):
                    raise ValueError('Each completed dimension row requires an integer coordinate count')
                if record['n_genes'] != len(genes):
                    raise ValueError('Dimension rows must retain the same common gene identities')
                if record['representation'].startswith('expression') and record['n_coordinates'] != len(cells):
                    raise ValueError('Expression coordinate count must equal the sampled cell count')
                valid, excluded = record['twonn_valid_rows'], record['twonn_excluded_rows']
                if min(valid, excluded) < 0 or valid + excluded != len(genes):
                    raise ValueError('TwoNN valid and excluded row counts must sum to common genes')
                if record['twonn_id'] is None and valid:
                    raise ValueError('TwoNN estimate is missing despite positive valid-row coverage')
                twonn = measurement(record['twonn_id']) if valid else 'not evaluable (zero valid rows)'
                short_cohort = 'K562' if cohort.startswith('K562') else cohort
                rows.append((short_cohort, labels[record['representation']], number(record['n_coordinates']),
                             measurement(record['participation_ratio']), twonn,
                             f'{number(valid)} / {number(excluded)}'))
        text = table('Dimensionality cohorts. Every expression, model and random-control row within a cohort uses these same common gene identities. Genes, rather than cells, are the objects whose dimensionality is estimated.',
                     ['Cohort', 'Sampled cells', 'Common genes'], cohort_rows, [.48, .19, .20])
        return text + table('Raw-coordinate gene-vector dimensionality and shape-matched controls. Participation ratio centres coordinate columns. TwoNN uses original Euclidean distances and excludes zero or tied nearest-neighbour distances. Valid/excluded counts describe genes, not biological replicates.',
                     ['Cohort', 'Representation', 'Coordinates', 'PR', 'TwoNN', 'Valid / excluded genes'],
                     rows, [.14, .27, .12, .10, .10, .15])

    def scaling(self):
        report = self.read(self.results / 'scale_reporting.json')
        require_source_hashes(report)
        expected = list(zip((0, 4, 8, 13, 17, 21, 25), (0, 7, 14, 24, 31, 38, 45)))
        pairs = report['pairs']
        if (report['schema'] != 'c2s-scale-reporting-v1' or report['passed'] is not True or
                report['n_pairs'] != 7 or len(pairs) != 7 or
                [(pair['2b']['layer'], pair['27b_matched']['layer']) for pair in pairs] != expected):
            raise ValueError('The complete audited seven matched-depth pairs are required')
        metric_rows, annotation_rows = [], []
        for pair in pairs:
            taps = f"{pair['2b']['layer']} / {pair['27b_matched']['layer']}"
            for key, label in [('2b', '2B'), ('27b_matched', '27B')]:
                row = pair[key]
                total, active, annotated = row['n_features'], row['n_catalogue_active'], row['n_ontology']
                if total != 9216 or not 0 < active <= total or row['n_catalogue_inactive'] != total-active:
                    raise ValueError('Matched dictionary capacity and active denominator are inconsistent')
                comparisons = [(row['ontology_all_features'], annotated/total),
                    (row['ontology_catalogue_active'], annotated/active),
                    (row['ontology_frequency_floor_all_features'],
                     row['ontology_frequency_floor_catalogue_active']*active/total),
                    (row['all_row_inactive_fraction'], (total-active)/total)]
                if any(not math.isclose(saved, derived, abs_tol=1e-12) for saved, derived in comparisons):
                    raise ValueError('Saved scaling annotation/inactivity denominator differs')
                metric_rows.append((taps, label, measurement(100*row['heldout_variance_explained']),
                    measurement(100*row['heldout_inactive_fraction']), measurement(100*row['all_row_inactive_fraction']),
                    f'{number(active)}/{number(total)}'))
                annotation_rows.append((taps, label, measurement(100*row['ontology_all_features']),
                    measurement(100*row['ontology_frequency_floor_all_features']),
                    measurement(100*row['ontology_catalogue_active']),
                    measurement(100*row['ontology_frequency_floor_catalogue_active'])))
        text = table('Seven matched-capacity layer pairs. Each dictionary has 9 216 features, k=32 and six saved training epochs. Held-out variance explained (VE) uses each model\'s residual-activation variance; 2B values are fresh re-evaluations. Held-out inactive features and features inactive across all catalogue positions have separate definitions.',
                     ['2B / 27B taps', 'Model', 'Held-out VE (%)', 'Held-out inactive (%)', 'All-position inactive (%)', 'Catalogue active / features'],
                     metric_rows, [.14, .08, .15, .16, .16, .20])
        return text + table('Named ontology/regulon annotation and its frequency-weighted random-set floor. Rates among all dictionary features retain inactive features in the denominator; catalogue-active rates use features firing anywhere. Floors have the corresponding denominator. Pair/network membership evidence is excluded.',
                     ['2B / 27B taps', 'Model', 'Named / all (%)', 'Floor / all (%)', 'Named / active (%)', 'Floor / active (%)'],
                     annotation_rows, [.14, .08, .16, .16, .17, .17])

    def residuals(self):
        report = self.read(self.results / 'residual_summary.json')
        require_source_hashes(report)
        rows, positions = [], set()
        for key, label, count, width in [('2b', '2B', 26, 2304), ('27b_matched', '27B', 46, 4608)]:
            records = report[key]['layers']
            if len(records) != count or sorted(row['layer'] for row in records) != list(range(count)):
                raise ValueError('Complete residual tap coverage is required for both models')
            for row in sorted(records, key=lambda item: item['layer']):
                if row['width'] != width or not math.isclose(row['centered_covariance_pr']/width,
                        row['centered_covariance_pr_fraction_width'], abs_tol=1e-12):
                    raise ValueError('Residual width/participation-ratio normalisation differs')
                positions.add(row['n_gene_positions'])
                rows.append((f"{label} L{row['layer']}", measurement(row['centered_covariance_pr']),
                    measurement(row['centered_covariance_pr_fraction_width']), measurement(row['top1_variance_share']),
                    measurement(row['top10_variance_share']), measurement(row['rms_max_over_median']),
                    measurement(row['mean_vector_norm_over_mean_row_norm'])))
        if len(positions) != 1 or min(positions) <= 0:
            raise ValueError('Residual taps must retain the same positive gene-position count')
        return table(f'All 72 residual taps, with {number(next(iter(positions)))} sampled gene positions per tap. Widths are 2 304 for 2B and 4 608 for 27B. PR and top variance shares use centred covariance; the mean-vector norm divided by mean row norm measures a separate mean component. RMS ratios use coordinate root-mean-square values. Final taps include final RMS normalisation. These descriptive quantities do not establish a cause of SAE inactivity.',
                     ['Model / tap', 'PR', 'PR / width', 'Top-1 share', 'Top-10 share', 'Max / median RMS', 'Mean / row norm'],
                     rows, [.13, .11, .13, .13, .13, .13, .13])

    def dimension_ablation(self):
        comparison = self.read(self.results / 'dropmassive_comparison.json')
        audited = self.read(self.results / 'scale_reporting.json')
        require_source_hashes(comparison)
        if audited['passed'] is not True or audited['optional_e4']['producer_comparison'] != comparison:
            raise ValueError('Current dimension-ablation output differs from the audited saved comparison')
        records = comparison['layers']
        if comparison['experiment'] != 'E4' or sorted(row['layer'] for row in records) != [24, 38]:
            raise ValueError('Both completed dimension-removal layers 24 and 38 are required')
        rows, variance_rows = [], []
        for row in sorted(records, key=lambda item: item['layer']):
            projection = row['input_projection']
            removed = projection['removed_dimensions']
            if (row['status'] != 'compared' or projection['mode'] != 'zero_selected_columns' or
                    not removed or len(removed) != len(set(removed)) or
                    projection['retained_input_dimensions'] != projection['original_width']-len(removed)):
                raise ValueError('Complete, consistent original-width input projection metadata required')
            for family in ('heldout', 'all_rows'):
                if not math.isclose(row[f'dead_fraction_{family}_projected']-row[f'dead_fraction_{family}_original'],
                                    row[f'dead_fraction_{family}_delta'], abs_tol=1e-12):
                    raise ValueError('Projected-minus-original inactivity delta differs')
            rows.append((row['layer'], number(len(removed)), measurement(100*row['dead_fraction_heldout_original']),
                measurement(100*row['dead_fraction_heldout_projected']), measurement(100*row['dead_fraction_all_rows_original']),
                measurement(100*row['dead_fraction_all_rows_projected'])))
            variance_rows.append((row['layer'], number(projection['retained_input_dimensions']),
                measurement(100*row['variance_explained_original_input']),
                measurement(100*row['variance_explained_projected_input'])))
        text = table('27B high-RMS-coordinate removal. Selected columns are zeroed while stored width remains unchanged. These rows distinguish held-out inactivity from inactivity anywhere in the full catalogue. The all-position inactive fraction can increase even when held-out inactivity falls.',
                     ['Layer', 'Columns zeroed', 'Held-out original (%)', 'Held-out projected (%)', 'All-position original (%)', 'All-position projected (%)'],
                     rows, [.07, .13, .17, .17, .17, .17])
        return text + table('Variance explained under different variance references. Original-input VE and projected-input VE use their respective input variances; they are not a like-for-like reconstruction contrast. Retained input columns exclude zeroed columns, while stored width remains 4 608.',
                     ['Layer', 'Retained input columns', 'Original-input VE (%)', 'Projected-input VE (%)'],
                     variance_rows, [.10, .23, .27, .27])

    def circuit_stability(self):
        report = self.read(self.results / 'circuit_stability_full.json')
        require_source_hashes(report)
        baseline = self.read(self.results / 'circuit_stability.json')
        expected_hashes = [value for path, value in report['source_sha256'].items()
                           if Path(path).name == 'circuit_stability.json']
        if expected_hashes != [self.hashes[str(self.results / 'circuit_stability.json')]]:
            raise ValueError('Circuit stability source hash differs from the audited full-family record')
        sources = report['per_source']
        if (report['passed'] is not True or report['status'] != 'complete' or report['source_layers'] != [7] or
                len(sources) != report['n_source_features'] or not sources or
                sum(bool(row['common_eligible']) for row in sources) != report['n_common_eligible_source_features']):
            raise ValueError('Complete audited source-layer-7 stability records are required')
        all_tested = report['pooled_all_tested']
        selected = report['selection_conditioned_shared_passing']
        comparison = selected['edge_comparison']
        if all_tested['n_effect_pairs'] != sum(row['all_tested']['n_effect_pairs'] for row in sources):
            raise ValueError('All-tested effect count differs from the per-source total')
        for key in ('n_first', 'n_second', 'n_shared', 'jaccard'):
            if not math.isclose(comparison[key], baseline['comparison'][key], abs_tol=1e-12):
                raise ValueError('Saved stability edge comparison differs from audited source')
        for key, field in [('n_first', 'n_first_passing'), ('n_second', 'n_second_passing'), ('n_shared', 'n_shared_passing')]:
            if comparison[key] != sum(row[field] for row in sources):
                raise ValueError('Passing-effect count differs from per-source summaries')
        if selected['correlations']['n_effect_pairs'] != comparison['n_shared']:
            raise ValueError('Shared-passing correlation count differs from shared effects')
        overall_rows = [(label, number(row['n_effect_pairs']), measurement(row['pearson']), measurement(row['spearman']))
            for label, row in [('All common tested effects', all_tested),
                               ('Shared passing effects only', selected['correlations'])]]
        count_rows, correlation_rows = [], []
        for row in sorted(sources, key=lambda item: item['source_feature']):
            if (not row['common_eligible'] or min(row['n_active_first'], row['n_active_second']) < 2 or
                    row['all_tested']['n_effect_pairs'] != row['n_target_layers']*row['n_target_features_per_layer']):
                raise ValueError('Common-eligible full tested-family coverage differs')
            union = row['n_first_passing']+row['n_second_passing']-row['n_shared_passing']
            if (union <= 0 or row['n_shared_passing'] > min(row['n_first_passing'], row['n_second_passing']) or
                    not math.isclose(row['passing_jaccard'], row['n_shared_passing']/union, abs_tol=1e-12)):
                raise ValueError('Per-source passing-set Jaccard differs from saved counts')
            count_rows.append((row['source_feature'], f"{number(row['n_active_first'])} / {number(row['n_active_second'])}",
                number(row['n_first_passing']), number(row['n_second_passing']), number(row['n_shared_passing']),
                measurement(row['passing_jaccard']), number(row['all_tested']['n_effect_pairs'])))
            correlation_rows.append((row['source_feature'], measurement(row['all_tested']['pearson']),
                measurement(row['all_tested']['spearman']), measurement(row['shared_passing_only']['pearson']),
                measurement(row['shared_passing_only']['spearman'])))
        text = table(f"Circuit stability for the same source-layer-7 feature panel at 50 and 100 sampled cells. All-tested correlations include every common tested downstream layer/feature effect, including zeros and nonpassing effects. The shared-passing comparison conditions on passing the rule in both runs. Passing sets contain {number(comparison['n_first'])} and {number(comparison['n_second'])} effects, with {number(comparison['n_shared'])} shared and Jaccard {number(comparison['jaccard'])}. These are rule-passing effects, not counts after a graph storage cap or FDR discoveries.",
                     ['Effect family', 'Effect pairs', 'Pearson', 'Spearman'], overall_rows, [.40, .16, .15, .15])
        text += table('Per-source activity and passing-effect counts. Active cells are source-active observations, not the nominal sampled cell count. Tested pairs enumerate the full common downstream family for each source.',
                     ['Source feature', 'Active cells 50 / 100', 'Passing 50', 'Passing 100', 'Shared passing', 'Jaccard', 'All tested pairs'],
                     count_rows, [.10, .17, .12, .12, .13, .12, .13])
        return text + table('Per-source effect correlations. All-tested and shared-passing columns have different selection conditions; shared-passing correlations are not substitutes for full-family stability. Stored float32 effects are used without recomputing model interventions.',
                     ['Source feature', 'All-tested Pearson', 'All-tested Spearman', 'Shared-passing Pearson', 'Shared-passing Spearman'],
                     correlation_rows, [.11, .19, .19, .19, .19])

    def phase_probes(self):
        within = self.read(self.results / 'within_k562_cellcycle.json')
        transfer = self.read(self.results / 'selection_matched_transfer.json')
        primary = 'model_minus_expression_topk_rank_fullpanel'
        if (within['schema'] != 'within-k562-input-ladder-v1' or
                transfer['schema'] != 'selection-matched-frozen-transfer-v1' or transfer['primary_contrast'] != primary):
            raise ValueError('Expected saved within-cohort and full-panel-selection-matched transfer schemas and primary contrast')
        k = within['k']
        arms = [('expression_full', 'Full expression'), ('expression_topk_magnitude', f'Top {k}: magnitudes'),
                ('expression_topk_binary', f'Top {k}: binary membership'), ('expression_topk_rank', f'Top {k}: ranks'),
                ('model', f"Model L{within['layer']}"), ('constant', 'Constant phase reference'), ('random', 'Random phase reference')]
        rows = [(label, number(within['point'][arm]['circular_r2']), saved_interval(within['point'][arm]['ci95']),
                 within['pca_dimensions'].get(arm, '--')) for arm, label in arms]
        text = table(f"Within-K562 circular R-squared on {number(within['n_cells'])} cells. Intervals use {number(within['n_boot'])} paired cell draws with fixed out-of-fold predictions and the resampled circular-loss denominator. PCA and scaling use the whole cohort before five-fold Ridge (alpha=10).",
                     ['Input / representation', 'circular R-squared', '95% interval', 'PCA dimensions'], rows, [.34, .15, .26, .12])
        comparisons = [('model_minus_expression_topk_rank', 'Model minus rank input'),
                       ('model_minus_expression_topk_binary', 'Model minus binary input'),
                       ('model_minus_expression_full', 'Model minus full expression')]
        interpretations = {'unresolved_difference': 'unresolved difference', 'first_arm_lower': 'model lower', 'first_arm_higher': 'model higher'}
        rows = [(label, number(within['contrasts'][key]['point_delta']), saved_interval(within['contrasts'][key]['ci95']),
                 interpretations[within['contrasts'][key]['interpretation']]) for key, label in comparisons]
        text += table('Paired within-K562 contrasts in circular R-squared. Intervals spanning zero do not establish equivalence.',
                      ['Contrast', 'Point difference', '95% interval', 'Saved interpretation'], rows, [.31, .15, .25, .16])
        k = transfer['k']
        arms = [('expression_full_shared', 'Full shared expression'), ('expression_topk_magnitude_fullpanel', f'Full-panel top {k}: magnitudes'),
                ('expression_topk_rank_fullpanel', f'Full-panel top {k}: ranks'), ('model', f"Model L{transfer['layer']}"),
                ('constant', 'Constant phase reference'), ('random', 'Random phase reference')]
        rows = [(label, number(transfer['point'][arm]['R_diff'])) for arm, label in arms]
        text += table(f"Frozen K562-to-RPE1 transfer R-diff: {number(transfer['n_source'])} source cells, {number(transfer['n_target'])} target cells and {number(transfer['n_shared_genes'])} shared gene identities for expression predictors. Source-only scaling and Ridge (alpha={number(transfer['ridge_alpha'])}) use no PCA. Individual-arm confidence intervals are not supplied by this artifact.",
                      ['Input / representation', 'R-diff'], rows, [.63, .24])
        contrast = transfer['contrasts'][primary]
        text += table(f"Primary paired transfer contrast in R-diff. The {number(transfer['n_boot'])} bootstrap draws resample target cells with source fits fixed; this is not uncertainty over source fitting or independent experiments.",
                      ['Contrast', 'Point difference', '95% interval', 'Saved interpretation'],
                      [('Model minus full-panel rank input', number(contrast['point_delta']), saved_interval(contrast['ci95']), interpretations[contrast['interpretation']])],
                      [.31, .15, .25, .16])
        return text

    def sampled_cache(self):
        rows = []
        for line in ('k562', 'rpe1'):
            record = self.read(self.results / f'cache_validation_{line}.json')
            criteria = record['criteria']
            rows.append((line.upper(), number(record['n_cached_rows']), number(record['n_sampled']), number(record['n_passed']),
                         'yes' if record['all_sampled_rows_pass'] else 'no', number(criteria['minimum_cosine']),
                         number(criteria['maximum_relative_l2']), 'required' if criteria['expected_row_must_be_nearest_or_tied'] else 'not required'))
        return table('Sampled fresh extraction checks of cached cell-summary states. Counts describe sampled rows only; identity or correctness of unsampled rows is not established. Source h5ad, state and row-mapping hashes are retained in the saved validation records.',
                     ['Line', 'Cached rows', 'Sampled', 'Passed', 'All sampled pass', 'Minimum cosine', 'Maximum relative L2', 'Expected row nearest / tied'],
                     rows, [.07, .09, .09, .07, .12, .12, .12, .13])

    def guide_coverage(self):
        record = self.read(self.results / 'audits/phase8_independent_checks.json')['designs']['k562_original48']
        rows = [('Selected TFs', record['n_tfs']), ('TF groups with one observed guide', record['n_tf_groups_with_single_guide']),
                ('TF groups with two observed guides', record['n_tf_groups_with_two_guides']),
                ('Selected cells, all groups', record['n_cells_total']), ('Non-targeting cells', record['n_controls']),
                ('Observed non-targeting guides', record['control_guide_count'])]
        return table('Guide coverage in the selected K562 perturbation cohort. Cell-level uncertainty is conditional on the observed guides; cell counts do not establish independent guide replication.',
                     ['Design quantity', 'Saved count'], [(label, number(value)) for label, value in rows], [.65, .22])

    def resources(self):
        paths = sorted(self.logs.glob('*.complete.json'))
        if not paths:
            raise FileNotFoundError(f'No completed stage records: {self.logs}')
        records = [self.read(path) for path in paths]
        records = [record for record in records if record['command']]
        if not records:
            raise KeyError('No executable stages in completion records')
        rows = []
        for record in sorted(records, key=lambda item: item['name']):
            if record['returncode'] != 0:
                raise ValueError(f"Non-successful completion record: {record['name']}")
            rows.append((record['name'], number(record['seconds']/60),
                         number(record['peak_device_memory_used_mib']/1024),
                         number(record['peak_process_tree_rss_bytes']/1e9), number(record['disk_used_delta_bytes']/1e9)))
        return table('Executable-stage resource measurements. GPU memory is the device-wide peak, including concurrent jobs; host RAM is the stage process tree. Disk change is a filesystem delta, not exclusive stage output size. A zero monitor value may mean the monitor was unavailable.',
                     ['Stage', 'Minutes', 'Device GiB', 'Host GB', 'Disk change GB'], rows, [.39, .11, .12, .12, .13])

    def costs(self):
        record = self.read(self.accounting)
        if not record['allocations'] or record['total_cost_usd'] is None or not record['basis']:
            raise KeyError('Compute accounting requires populated allocations, total cost and a recorded charging basis')
        if any(item.get(key) is None for item in record['allocations']
               for key in ('gpu', 'hourly_usd', 'billed_seconds', 'cost_usd')):
            raise KeyError('Compute accounting has unrecorded allocation fields')
        rows = [(item['gpu'], number(item['hourly_usd']), number(item['billed_seconds']/3600), number(item['cost_usd']))
                for item in record['allocations']]
        rows.append(('Reported total, all allocations', '--', '--', number(record['total_cost_usd'])))
        if 'conservative_combined_lifetime_estimate_usd' in record:
            rows.append(('Conservative full-lifetime estimate', '--', '--',
                         measurement(record['conservative_combined_lifetime_estimate_usd'])))
        return table('Compute allocation accounting. Basis: ' + record['basis'],
                     ['Device', 'USD / hour', 'GPU-billed hours', 'USD'], rows, [.40, .16, .16, .15])

    def provenance(self):
        rows = [('1', 'Conceptual workflow schematic', 'No statistical sample or test; vector artwork', 'make_design_schematic.py')]
        for name, (quantity, unit) in FIGURE_UNITS.items():
            record = self.figure(name)
            names = sorted({Path(path).name for path in record['source_sha256']})
            # A longtable can break between rows, never within a source-list cell.
            # Keep every filename while limiting the height of each continuation row.
            for offset in range(0, len(names), 6):
                rows.append((name, quantity if offset == 0 else '', unit if offset == 0 else '',
                             '; '.join(names[offset:offset+6])))
        return table('Figure provenance. All data panels are generated by make_revision_figures.py. The named source artifacts and their full SHA256 values are retained in each Figure_NAME.provenance.json sidecar; plotted_values contains the exact denominators and measurements. The assembly manifest also hashes every included figure and sidecar.',
                     ['Fig.', 'Panels / quantities', 'Sample unit / uncertainty', 'Source artifact names'], rows, [.05, .28, .26, .30])

    def write(self, out, draft=False):
        out = Path(out)
        fragments, pending = {}, {}
        builders = {'tokenization': self.tokenization, 'databases': self.databases, 'software': self.software,
                    'regulatory': self.regulatory, 'reconstruction': self.reconstruction, 'dimensions': self.dimensions,
                    'scaling': self.scaling, 'residuals': self.residuals,
                    'dimension_ablation': self.dimension_ablation, 'circuit_stability': self.circuit_stability,
                    'text_deletion': self.text_deletion, 'surface_forms': self.surface_forms,
                    'phase_probes': self.phase_probes, 'sampled_cache': self.sampled_cache, 'guide_coverage': self.guide_coverage,
                    'resources': self.resources, 'costs': self.costs, 'provenance': self.provenance}
        for name, builder in builders.items():
            try:
                fragments[name] = builder()
            except (FileNotFoundError, KeyError) as error:
                pending[name] = str(error)
                fragments[name] = '\\SupplementPending{' + tex_escape(name + ': required saved input unavailable; details are in the assembly manifest.') + '}\n'
        for index in range(1, 7):
            name = f'S{index}'
            try:
                self.figure(name)
                relative_pdf = Path(os.path.relpath(self.figures / f'Figure_{name}.pdf', self.paper)).as_posix()
                fragments[f'figure_{name}'] = f'\\includegraphics[width=\\linewidth]{{{relative_pdf}}}\n'
            except (FileNotFoundError, KeyError) as error:
                pending[f'figure_{name}'] = str(error)
                fragments[f'figure_{name}'] = '\\SupplementPending{' + tex_escape(f'Figure {name}: validated PDF and provenance are required; details are in the assembly manifest.') + '}\n'
        if pending and not draft:
            raise ValueError('Incomplete supplementary assembly; no output written:\n' + '\n'.join(f'{name}: {reason}' for name, reason in pending.items()))
        manifest = {'schema': 'supplement-assembly-v1', 'status': 'draft_incomplete' if pending else 'complete',
                    'assembled_utc': datetime.now(timezone.utc).isoformat(),
                    'complete_blocks': sorted(set(builders)-set(pending)), 'pending_blocks': pending,
                    'input_sha256': self.hashes, 'figure_units': FIGURE_UNITS,
                    'limits': ['Assembly reads saved summary measurements; it does not rerun or independently validate statistical analyses.',
                               'Figure sidecars record source hashes; this helper does not re-read raw activation/checkpoint arrays.']}
        fragments['status'] = '\\SupplementCompletefalse\n' if pending else '\\SupplementCompletetrue\n'
        manifest['output_sha256'] = {f'{name}.tex': hashlib.sha256(fragment.encode()).hexdigest()
                                     for name, fragment in fragments.items()}
        out.mkdir(parents=True, exist_ok=True)
        for name, fragment in fragments.items():
            (out / f'{name}.tex').write_text(fragment)
        (out / 'assembly_manifest.json').write_text(json.dumps(manifest, indent=2, allow_nan=False) + '\n')
        return manifest


def main():
    paper = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir', type=Path, default=paper/'experiments/results')
    parser.add_argument('--figures-dir', type=Path, default=paper/'figures')
    parser.add_argument('--stage-logs-dir', type=Path, default=paper/'experiments/stage_logs')
    parser.add_argument('--ontology-versions', type=Path, default=paper/'experiments/ontologies_v2/VERSIONS.json')
    parser.add_argument('--compute-accounting', type=Path, default=paper/'experiments/results/compute_accounting.json')
    parser.add_argument('--out-dir', type=Path, default=paper/'supplement_generated')
    parser.add_argument('--draft', action='store_true', help='Write explicit pending placeholders; not suitable for submission')
    args = parser.parse_args()
    try:
        manifest = Assembler(paper, args.results_dir, args.figures_dir, args.stage_logs_dir,
                             args.ontology_versions, args.compute_accounting).write(args.out_dir, args.draft)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps({'status': manifest['status'], 'pending_blocks': manifest['pending_blocks'], 'out': str(args.out_dir)}, indent=2))


if __name__ == '__main__':
    main()
