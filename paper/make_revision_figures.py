#!/usr/bin/env python3
"""Artifact-driven publication figures for CBAC-D-26-05545.

Examples:
  python make_revision_figures.py --list-required
  python make_revision_figures.py --figures 2 3 4 5 6 --results-dir experiments/results
  python make_revision_figures.py --figures S1 S2 S3 S4 S5 S6
  python make_revision_figures.py --synthetic-preview /tmp/c2s-figure-layout

Every final plot requires saved measurements; missing fields/files fail before that figure is
written. No old annotation percentages or benchmark constants are embedded here. Each figure has
a JSON provenance sidecar containing plotted values and input SHA256s. Synthetic layout previews
are prominently labelled and must live outside the final figures directory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import textwrap
import warnings

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np


BLUE, ORANGE, GREEN, PURPLE, GREY = '#0072B2', '#D55E00', '#009E73', '#CC79A7', '#777777'
MODEL_COLOURS = {'model': BLUE, 'expression': ORANGE, 'input': GREEN}
MAIN = ['2', '3', '4', '5', '6']
SUPPLEMENT = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6']
LAYERS = list(range(26))
MATCH_2B, MATCH_27B = [0, 4, 8, 13, 17, 21, 25], [0, 7, 14, 24, 31, 38, 45]
EXAMPLE_CATEGORIES = ['haem and erythroid', 'translation and ribosome', 'cell cycle and mitosis']


def numeric(values, label):
    array = np.asarray(values, dtype=float)
    if not np.isfinite(array).all():
        raise ValueError(f'{label}: missing or nonfinite measurements')
    return array


def circuit_edge_profile(graph, source_layers=(0, 7, 16), storage_cap=50):
    """Count stored graph records, retaining target-layer zeros and verifying the saved total.

    Counts describe the capped panel graph, not every effect passing the tracing rule.
    Each source feature stores at most ``storage_cap`` targets per downstream layer.
    """
    edges, saved_count = graph.get('edges'), graph.get('n_edges')
    if not isinstance(edges, list) or not edges or type(saved_count) is not int or saved_count!=len(edges):
        raise ValueError('Circuit graph must contain nonempty edges matching its saved n_edges')
    counts = {str(layer): [0]*len(LAYERS) for layer in source_layers}
    unique_edges, source_features, target_features, block_counts = set(), set(), set(), {}
    for edge in edges:
        identity = tuple(edge.get(key) for key in ('source_layer', 'source_feature', 'target_layer', 'target_feature'))
        if any(type(value) is not int or value<0 for value in identity):
            raise ValueError('Circuit source/target layer and feature IDs must be nonnegative integers')
        source, source_feature, target, target_feature = identity
        if source not in source_layers or target not in LAYERS or target<=source:
            raise ValueError('Circuit edge has an unexpected source or a non-downstream target layer')
        if identity in unique_edges:
            raise ValueError('Circuit graph contains duplicate source-target edges')
        unique_edges.add(identity)
        source_features.add((source, source_feature)); target_features.add((target, target_feature))
        block = (source, source_feature, target)
        block_counts[block] = block_counts.get(block, 0)+1
        if block_counts[block]>storage_cap:
            raise ValueError('Circuit graph exceeds the documented per-source target-layer storage cap')
        counts[str(source)][target] += 1
    totals = {source:sum(row) for source,row in counts.items()}
    if sum(totals.values())!=saved_count:
        raise ValueError('Circuit edge profile does not reconcile to the saved graph total')
    return {'n_edges':saved_count, 'target_layers':LAYERS.copy(), 'source_layers':list(source_layers),
            'counts_by_source_layer':counts, 'total_by_source_layer':totals,
            'n_source_features_counted':len(source_features), 'n_target_features_counted':len(target_features),
            'storage_cap_per_source_feature_per_target_layer':storage_cap,
            'interpretation':'Stored capped-panel edges; not uncapped counts of all rule-passing effects'}


def labelled_title(axis, letter, title):
    wrapped = '\n'.join(textwrap.fill(line, 28) for line in title.split('\n'))
    axis.set_title(f'{letter}  {wrapped}', loc='left', fontsize=8.2, fontweight='bold', pad=8)


def error_points(axis, positions, estimates, intervals, colour, label=None, marker='o'):
    """Plot saved interval endpoints directly; do not assume that the estimate lies inside them."""
    estimates = numeric(estimates, label or 'estimates')
    intervals = numeric(intervals, label or 'intervals')
    if intervals.shape != (len(estimates), 2) or np.any(intervals[:, 1] < intervals[:, 0]):
        raise ValueError('interval endpoints are not aligned or ordered')
    axis.vlines(positions, intervals[:, 0], intervals[:, 1], color=colour, lw=1.1)
    axis.plot(positions, estimates, marker, color=colour, ms=4, label=label)


class Figures:
    def __init__(self, args):
        self.args = args
        self.results, self.runs, self.route, self.geometry = (args.results_dir, args.runs_dir,
                                                            args.route_dir, args.geometry_dir)
        self.sources, self.values = {}, {}

    def read(self, path):
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f'Missing required artifact: {path}')
        raw = path.read_bytes()
        document = json.loads(raw, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f'nonfinite JSON value {value}')))
        if isinstance(document, dict) and document.get('synthetic') and not self.args.synthetic_preview:
            raise ValueError(f'Refusing synthetic artifact for a final figure: {path}')
        self.sources[str(path)] = hashlib.sha256(raw).hexdigest()
        return document

    def arrays(self, path):
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f'Missing required artifact: {path}')
        self.sources[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        with np.load(path, allow_pickle=False) as archive:
            return {key: archive[key] for key in archive.files}

    def annotation(self, group, layer):
        path = self.results / 'annotations' / group / f'ann_v2_L{layer:02d}.json'
        record = self.read(path)['summary']
        try:
            rate, floor = record['ontology_annotation_rate_alive'], record['floors']['freq']['ontology_mean']
            numeric([rate, floor], f'{path}: ontology rate and frequency-matched ontology floor')
        except KeyError as error:
            raise ValueError(f'{path}: missing ontology-only rate/floor field {error}; rerun the current scorer') from error
        return {'layer': layer, 'rate': rate, 'floor': floor,
                'n_alive': record['n_alive'], 'n_total': record['n_features_total']}

    def evaluated(self, layer):
        return self.read(self.results / f'2b_eval_L{layer:02d}.json')

    def train(self, run, layer):
        return self.read(self.runs / run / 'sae' / f'train_L{layer:02d}.json')

    def save(self, figure, name):
        if self.args.synthetic_preview:
            figure.suptitle('SYNTHETIC LAYOUT PREVIEW - NO STUDY RESULTS', color='#A00000', fontsize=10)
            figure.text(.5, .47, 'SYNTHETIC PREVIEW', ha='center', va='center', rotation=18,
                        color='#A00000', alpha=.13, fontsize=25, weight='bold')
        self.args.out_dir.mkdir(parents=True, exist_ok=True)
        stem = ('synthetic_' if self.args.synthetic_preview else '') + f'Figure_{name}'
        for extension in ('pdf', 'svg', 'png'):
            figure.savefig(self.args.out_dir / f'{stem}.{extension}', dpi=300, bbox_inches='tight', pad_inches=.07)
        sidecar = {'figure': name, 'synthetic_preview': bool(self.args.synthetic_preview),
                   'source_sha256': self.sources, 'plotted_values': self.values,
                   'generator_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
        (self.args.out_dir / f'{stem}.provenance.json').write_text(json.dumps(sidecar, indent=2, allow_nan=False))
        plt.close(figure)
        print(f'Wrote {stem}: PDF, SVG, 300-dpi PNG and provenance', flush=True)

    def figure2(self):
        evaluation = [self.evaluated(layer) for layer in LAYERS]
        annotation = [self.annotation('2b', layer) for layer in LAYERS]
        persistence = self.read(self.runs / '2b_atlas/cross_layer_tracking.json')
        self.values.update(evaluation=evaluation, annotation=annotation,
                           persistence=persistence['long_range_persistence'],
                           decoder_matching_threshold=persistence['config']['match_threshold'])
        fig, axes = plt.subplots(2, 2, figsize=(7.5, 4.8), layout='constrained')
        axis = axes[0, 0]
        axis.plot(LAYERS, numeric([r['variance_explained_reextracted'] for r in evaluation], 'reconstruction'), 'o-', color=BLUE, ms=2.7)
        axis.set(xlabel='Layer', ylabel='Held-out variance explained')
        labelled_title(axis, 'a', 'Dictionary reconstruction')
        axis = axes[0, 1]
        dead = [100*r['dead_val_reextracted']/r['n_features'] for r in evaluation]
        all_dead = [100*r['dead_frac_all_rows'] for r in evaluation]
        axis.plot(LAYERS, dead, 'o-', color=ORANGE, ms=2.7, label='Held-out positions')
        axis.plot(LAYERS, all_dead, '--', color=GREY, label='All atlas positions')
        axis.set(xlabel='Layer', ylabel='Features that never fire (%)')
        axis.legend(frameon=False, fontsize=6.5)
        labelled_title(axis, 'b', 'Firing depends on the evaluation sample')
        axis = axes[1, 0]
        axis.plot(LAYERS, [100*r['rate'] for r in annotation], 'o-', color=GREEN, ms=2.7, label='Ontology / regulon')
        axis.plot(LAYERS, [100*r['floor'] for r in annotation], '--', color=GREY, label='Frequency-matched random sets')
        axis.set(xlabel='Layer', ylabel='Live features annotated (%)', ylim=(0, None))
        axis.legend(frameon=False, fontsize=6.3)
        labelled_title(axis, 'c', 'Named biological terms against chance')
        axis = axes[1, 1]
        rows = persistence['long_range_persistence']
        axis.plot([r['target_layer'] for r in rows], [100*r['persistence_rate'] for r in rows], 'o-', color=BLUE, ms=2.7)
        axis.set(xlabel='Target layer', ylabel='Layer-0 decoder directions matched (%)', ylim=(0, 105))
        axis.text(.03, .06, f'Absolute cosine >= {persistence["config"]["match_threshold"]:g}', transform=axis.transAxes, fontsize=6.5)
        labelled_title(axis, 'd', 'Decoder persistence across depth')
        self.save(fig, '2')

    def figure3(self):
        biology = self.read(self.results / 'biology.json')
        line_metadata = self.read(self.results / 'lines2.json')
        frequencies = self.arrays(self.results / 'lines2_freq.npz')
        layer_rows = sorted(biology['layers'], key=lambda row: row['layer'])
        categories = biology['categories']
        example_layer = next((r for r in layer_rows if r['layer'] == self.args.example_layer), None)
        if example_layer is None:
            raise ValueError(f'No biology summary for example layer {self.args.example_layer}')
        examples = []
        for category in self.args.example_categories:
            if category not in example_layer['examples']:
                raise ValueError(f'No saved example for category {category!r} at layer {self.args.example_layer}; choose --example-categories explicitly')
            examples.append((category, example_layer['examples'][category]))
        heatmap = np.array([[100*r['category_share_all_features'][category] for r in layer_rows] for category in categories])
        self.values.update(category_order=categories, layers=[r['layer'] for r in layer_rows],
                           category_share_all_features_percent=heatmap.tolist(), examples=dict(examples), line_activity={})
        fig = plt.figure(figsize=(7.5, 5.6), layout='constrained')
        grid = fig.add_gridspec(2, 3, height_ratios=(1.7, 1))
        axis = fig.add_subplot(grid[0, :2])
        image = axis.pcolormesh(np.arange(len(layer_rows)+1), np.arange(len(categories)+1), heatmap,
                               cmap=LinearSegmentedColormap.from_list('biological_coverage', ['#FFFFFF', '#91D0BC', '#00694F']), vmin=0, shading='flat', rasterized=False)
        ticks = list(range(0, len(layer_rows), 5))
        axis.set_xticks(np.array(ticks)+.5, [layer_rows[i]['layer'] for i in ticks])
        axis.set_yticks(np.arange(len(categories))+.5, [textwrap.fill(c, 24) for c in categories], fontsize=6.2)
        axis.invert_yaxis()
        axis.set(xlabel='Layer')
        fig.colorbar(image, ax=axis, fraction=.035, pad=.02, label='All dictionary features (%)')
        labelled_title(axis, 'a', 'Biological categories across depth')
        axis = fig.add_subplot(grid[0, 2]); axis.axis('off')
        labelled_title(axis, 'b', f'Examples at layer {self.args.example_layer}')
        for index, (category, record) in enumerate(examples):
            term = record['term']
            if 'fold' not in term:
                raise ValueError('Named feature examples require saved fold enrichment and q-value')
            genes = [g['gene_name'] if isinstance(g, dict) else g for g in record['top_genes'][:6]]
            y = .98-index*.33
            axis.text(0, y, f'F{record["feature_idx"]}: {category}', transform=axis.transAxes, fontsize=6.7, weight='bold', va='top', wrap=True)
            body = textwrap.fill(term['term'], 32)+'\n'+textwrap.fill(', '.join(genes), 33)+f'\nq={term["q"]:.1e}; fold={term["fold"]:.1f}'
            axis.text(0, y-.055, body, transform=axis.transAxes, fontsize=6.0, va='top', linespacing=1.4)
        lines = line_metadata['lines']
        for index, (category, record) in enumerate(examples):
            axis = fig.add_subplot(grid[1, index])
            values = []
            for line in lines:
                key = f'L{self.args.example_layer:02d}_{line}'
                if key not in frequencies:
                    raise ValueError(f'Missing line activity array {key}')
                values.append(float(100*frequencies[key][record['feature_idx']]))
            self.values['line_activity'][category] = {'feature': record['feature_idx'], 'lines': lines, 'firing_percent': values}
            axis.bar(range(len(lines)), values, color=[BLUE if line.lower()=='k562' else GREEN if line.lower()=='rpe1' else GREY for line in lines], width=.65)
            axis.set_xticks(range(len(lines)), [line.upper() for line in lines], rotation=25, ha='right', fontsize=6.5)
            axis.set_ylabel('Gene positions with feature active (%)', fontsize=6.7)
            labelled_title(axis, chr(ord('c')+index), f'F{record["feature_idx"]}: {textwrap.fill(category, 22)}')
        self.save(fig, '3')

    def regulatory(self, line):
        return self.read(self.results / f'phase8_{line}/statistics/regulatory_statistics.json')

    @staticmethod
    def family(document, representation, layer):
        matching = [r for r in document['summary'] if r['representation']==representation and r['layer']==layer]
        if len(matching)!=1:
            raise ValueError(f'Expected one regulatory family: {representation}, layer {layer}')
        return matching[0]

    def specificity_panel(self, axis, document, letter, title):
        records = [self.family(document, 'expression', None), self.family(document, 'input', None), self.family(document, 'model', 16)]
        if any(r['n_tfs']<=0 for r in records):
            raise ValueError('Cannot plot a specificity percentage without tested TFs')
        values = [100*r['n_specific']/r['n_tfs'] for r in records]
        axis.bar(range(3), values, color=[ORANGE, GREEN, BLUE], width=.6)
        for index, record in enumerate(records):
            axis.text(index, values[index], f'{record["n_specific"]}/{record["n_tfs"]}', va='bottom', ha='center', fontsize=6.5)
        axis.set_xticks(range(3), ['Expression\n(gene test)', 'Input ranks\n(gene test)', 'Model L16\n(feature test)'], fontsize=6.2)
        axis.set_ylabel('Tested TFs with specific response (%)')
        axis.margins(y=.18)
        labelled_title(axis, letter, title)
        return records

    def retrieval_panel(self, axis, records, field, metric, labels, letter, title):
        estimates = [r[field][f'mean_{metric}'] for r in records]
        intervals = [r[field][f'mean_{metric}_ci95'] for r in records]
        if any(value is None for value in estimates+intervals):
            raise ValueError(f'Missing saved retrieval statistic or TF-bootstrap CI: {field}/{metric}')
        for index, (point, interval) in enumerate(zip(estimates, intervals)):
            error_points(axis, [index], [point], [interval], MODEL_COLOURS[records[index]['representation']])
        if metric=='average_precision':
            # Rare-target AP has a finite-sample null above prevalence. Show both;
            # prevalence alone can make a below-null score appear informative.
            prevalence = numeric([r[field]['mean_prevalence'] for r in records], 'target prevalence')
            expected_ap = numeric([r[field]['mean_random_label_expected_ap'] for r in records], 'finite random-label AP')
            axis.plot(range(len(records)), prevalence, 'x', color=GREY, ms=5, label='Target prevalence')
            axis.plot(range(len(records)), expected_ap, '+', color=PURPLE, ms=6, label='Finite random-label AP')
            axis.set_ylabel('Average precision (stepwise AUPRC)')
        else:
            axis.axhline(.5, ls='--', color=GREY, lw=.8, label='Chance AUROC')
            axis.set_ylabel('AUROC')
        axis.set_xticks(range(len(records)), labels, fontsize=6.2)
        axis.set_ylim(bottom=0)
        axis.legend(frameon=False, fontsize=5.8)
        labelled_title(axis, letter, title)

    def figure4(self):
        graph = self.read(self.runs / '2b_atlas/circuit_graph.json')
        edge_profile = circuit_edge_profile(graph)
        circuits = self.read(self.results / 'circuit_baseline.json')
        k562, rpe1 = self.regulatory('k562'), self.regulatory('rpe1')
        if not k562.get('summary') or not rpe1.get('summary'):
            raise ValueError('Figure4 requires nonempty K562 and RPE1 statistical summaries')
        self.values.update(circuit_edge_profile=edge_profile, circuits=circuits,
                           k562=k562['summary'], rpe1=rpe1['summary'])
        fig, axes = plt.subplots(2, 3, figsize=(7.5, 5.0), layout='constrained')
        axis = axes[0, 0]
        for source, colour in zip(edge_profile['source_layers'], (BLUE, ORANGE, GREEN)):
            targets = np.asarray(edge_profile['target_layers'])
            downstream = targets>source
            counts = np.asarray(edge_profile['counts_by_source_layer'][str(source)])
            axis.plot(targets[downstream], counts[downstream], 'o-', ms=2.3, color=colour, label=f'Source L{source}')
        maximum_count = max(max(row) for row in edge_profile['counts_by_source_layer'].values())
        axis.set(xlabel='Target layer', ylabel='Stored circuit edges', xlim=(1, 25), ylim=(0, maximum_count*1.35))
        axis.set_xticks([1, 5, 10, 15, 20, 25])
        axis.legend(frameon=False, fontsize=5.8)
        axis.text(.98, .96, 'At most 50 targets per source\nfeature and target layer',
                  ha='right', va='top', transform=axis.transAxes, fontsize=5.5)
        labelled_title(axis, 'a', 'Stored circuit edges by target layer')
        axis = axes[0, 1]
        bootstrap = circuits['source_bootstrap']
        for index, (name, colour) in enumerate((('observed', BLUE), ('baseline', GREY))):
            record = bootstrap[name]
            error_points(axis, [index], [100*record['estimate']], [[100*x for x in record['ci95']]], colour)
        axis.set_xticks([0, 1], ['Observed', 'Same source,\nrandom target'], fontsize=6.3)
        axis.set(ylabel='Annotated edges sharing a term (%)', ylim=(0, 105))
        excess = bootstrap['excess']
        axis.text(.04, .05, f'Excess: {100*excess["estimate"]:.1f} points\n95% CI {100*excess["ci95"][0]:.1f} to {100*excess["ci95"][1]:.1f}', transform=axis.transAxes, fontsize=6.3)
        labelled_title(axis, 'b', 'Circuit biology against chance')
        primary = self.specificity_panel(axes[0, 2], k562, 'c', 'K562 factor specificity')
        self.retrieval_panel(axes[1, 0], primary, 'gene_retrieval', 'average_precision',
                             ['Expression', 'Input', 'Model L16'], 'd', 'K562 gene-level target retrieval')
        self.specificity_panel(axes[1, 1], rpe1, 'e', 'RPE1 factor specificity')
        model_rows = [self.family(k562, 'model', layer) for layer in (8, 16, 22)]
        self.retrieval_panel(axes[1, 2], model_rows, 'feature_retrieval', 'average_precision',
                             ['L8', 'L16', 'L22'], 'f', 'K562 feature-level target retrieval')
        # Prevalence is a class-balance reference; finite random-label expected AP can differ.
        for axis in (axes[1, 0], axes[1, 2]):
            handles, labels = axis.get_legend_handles_labels()
            axis.set_ylim(0, axis.get_ylim()[1]*1.25)
            axis.legend(handles, labels, loc='upper right', frameon=False, fontsize=5.8)
        self.save(fig, '4')

    def figure5(self):
        residual = self.read(self.results / 'residual_summary.json')
        annotation = {'2b': [self.annotation('2b', layer) for layer in MATCH_2B],
                      '27b_matched': [self.annotation('27b_matched', layer) for layer in MATCH_27B]}
        self.values.update(residual=residual, matched_annotation=annotation)
        fig, axes = plt.subplots(2, 2, figsize=(7.5, 4.8), layout='constrained')
        for model, layers, colour, label, width_depth in [('2b', MATCH_2B, BLUE, '2B', 25),
                                                        ('27b_matched', MATCH_27B, ORANGE, '27B matched', 45)]:
            rows = residual[model]['layers']
            chosen = [next(r for r in rows if r['layer']==layer) for layer in layers]
            x = [r['layer']/width_depth for r in chosen]
            axes[0, 0].plot(x, numeric([100*r['dead_fraction_heldout'] for r in chosen], 'matched held-out dead fraction'), 'o-', color=colour, ms=3, label=label)
            axes[0, 1].plot(x, [100*r['rate'] for r in annotation[model]], 'o-', color=colour, ms=3, label=label)
            axes[0, 1].plot(x, [100*r['floor'] for r in annotation[model]], '--', color=colour, lw=1, label=f'{label} random floor')
            axes[1, 0].plot([r['layer']/width_depth for r in rows], [r['centered_covariance_pr_fraction_width'] for r in rows], '-', color=colour, label=label)
            points = [r for r in rows if r['dead_fraction_heldout'] is not None]
            axes[1, 1].scatter(numeric([r['rms_max_over_median'] for r in points], 'RMS shape'),
                               [100*r['dead_fraction_heldout'] for r in points], s=17, color=colour, alpha=.8, label=label)
        panels = [('a', 'Equal dictionary capacity', 'Relative depth', 'Held-out dead features (%)'),
                  ('b', 'Biological terms at matched depth', 'Relative depth', 'Live features annotated (%)'),
                  ('c', 'Residual-stream dimensionality', 'Relative depth', 'Centered covariance PR / width'),
                  ('d', 'Activation shape and dead features', 'Maximum / median dimension RMS', 'Held-out dead features (%)')]
        for axis, (letter, title, xlabel, ylabel) in zip(axes.flat, panels):
            axis.set(xlabel=xlabel, ylabel=ylabel)
            axis.legend(frameon=False, fontsize=6)
            labelled_title(axis, letter, title)
        axes[0, 1].set_ylim(0, 105)
        axes[1, 1].set_xscale('log')
        axes[1, 1].legend(loc='upper left', frameon=False, fontsize=6)
        self.save(fig, '5')

    def figure6(self):
        geometry = self.read(self.geometry / 'gene_manifold_null.json')['layers']
        steering = self.read(self.route / 'manifold_steer_L21.json')
        ladder = self.read(self.results / 'selection_matched_transfer.json')
        if ladder['schema']!='selection-matched-frozen-transfer-v1':
            raise ValueError('Figure6 requires the full-panel-selection-matched transfer analysis')
        dimension_path = self.results / 'cd34_dimensions.json'
        dimensions = self.read(dimension_path) if dimension_path.exists() else None
        fig, axes = plt.subplots(2, 2, figsize=(7.5, 4.8), layout='constrained') if dimensions else plt.subplots(1, 3, figsize=(7.5, 2.8), layout='constrained')
        axes = np.asarray(axes).ravel()
        x = np.arange(len(geometry))
        axes[0].plot(x, [r['coexpr_matched']['within_bin_mean_cos_delta'] for r in geometry], 'o', color=GREEN, label='Observed')
        axes[0].errorbar(x+.12, [r['coexpr_matched']['coexpr_null_mean'] for r in geometry],
                         yerr=[2*r['coexpr_matched']['coexpr_null_std'] for r in geometry], fmt='x', color=GREY, capsize=3, label='Null mean +/- 2 SD')
        axes[0].set_xticks(x, [f'L{r["layer"]}' for r in geometry])
        axes[0].axhline(0, color='#333333', lw=.5)
        axes[0].set_ylabel('Same- vs different-function cosine gap')
        axes[0].legend(frameon=False, fontsize=5.7)
        labelled_title(axes[0], 'a', 'Gene geometry beyond co-expression')
        paths = {}
        for arm in ('manifold', 'linear', 'random'):
            paths[arm] = [float(np.abs(np.diff(np.unwrap(numeric([point['phase'] for point in row['traj'][arm]], 'steering phases')))).sum()) for row in steering['rows']]
        axes[1].boxplot([paths[arm] for arm in paths], tick_labels=['Manifold', 'Linear', 'Random'], widths=.55, patch_artist=True,
                        boxprops={'facecolor': '#DDEBF3'}, medianprops={'color': BLUE}, flierprops={'markersize': 2})
        axes[1].set_ylabel('Readout phase path length (radians)')
        axes[1].tick_params(axis='x', labelsize=6.2)
        labelled_title(axes[1], 'b', f'Paired steering, L{steering["layer"]}')
        names = [('expression_full_shared', 'Full shared expression'), ('expression_topk_magnitude_fullpanel', 'Top-512 + values'),
                 ('expression_topk_rank_fullpanel', 'Top-512 + rank'), ('model', f'Model L{ladder["layer"]}'),
                 ('constant', 'Constant'), ('random', 'Random')]
        y = np.arange(len(names))[::-1]
        values = [ladder['point'][key]['R_diff'] for key, _ in names]
        axes[2].barh(y, values, color=[BLUE if key=='model' else GREY if key in ('constant', 'random') else ORANGE for key, _ in names], height=.65)
        axes[2].set_yticks(y, [label for _, label in names], fontsize=6.2)
        axes[2].set(xlabel=r'Phase agreement ($R_{\rm diff}$)', xlim=(0, 1))
        labelled_title(axes[2], 'c', 'Frozen K562 to RPE1 transfer')
        contrast = ladder['contrasts']['model_minus_expression_topk_rank_fullpanel']
        axes[2].text(.34, .015, f'Model - rank: {contrast["point_delta"]:+.3f}\n95% paired CI:\n{contrast["ci95"][0]:+.3f} to {contrast["ci95"][1]:+.3f}',
                     transform=axes[2].transAxes, fontsize=6.0, va='bottom')
        if dimensions:
            self.dimension_panel(axes[3], dimensions, 'd', 'CD34+ dimensionality controls')
        self.values.update(gene_geometry=geometry, steering_path_lengths=paths,
                           steering_definition='sum of absolute differences of unwrapped per-cell readout phases',
                           transfer_ladder=ladder, cd34_dimensions=dimensions,
                           cd34_dimensions_included=dimensions is not None)
        self.save(fig, '6')

    def variant(self, layer, k=32, expansion=4):
        if k==32 and expansion==4:
            trained, evaluated = self.train('2b_atlas', layer), self.evaluated(layer)
            return {'k': k, 'expansion': expansion, 'layer': layer,
                    've': evaluated['variance_explained_reextracted'],
                    'dead': evaluated['dead_val_reextracted']/evaluated['n_features'],
                    'median_firing': evaluated['firing_freq_quantiles_1_10_50_90_99'][2],
                    'decoder_overlap': trained['mean_decoder_cos'], **self.annotation('2b', layer)}
        run = f'sweep_L{layer:02d}_k{k}_x{expansion}'
        trained = self.train(run, layer)
        return {'k': k, 'expansion': expansion, 'layer': layer, 've': trained['variance_explained'],
                'dead': trained['dead_frac'], 'median_firing': trained['firing_freq_quantiles_1_10_50_90_99'][2],
                'decoder_overlap': trained['mean_decoder_cos'], **self.annotation(run, layer)}

    def supplement1(self):
        rows = {layer: [self.variant(layer, k) for k in (8, 16, 32, 64, 128)] for layer in (0, 13)}
        expansions = [self.variant(13, expansion=x) for x in (2, 4, 8)]
        fig, axes = plt.subplots(2, 3, figsize=(7.5, 4.8), layout='constrained')
        metrics = [('ve', 'Held-out variance explained'), ('dead', 'Held-out dead fraction'),
                   ('median_firing', 'Median firing fraction (live features)'),
                   ('decoder_overlap', 'Sampled decoder mean absolute cosine')]
        for axis, (key, ylabel) in zip(axes.flat, metrics):
            for layer, colour in ((0, BLUE), (13, ORANGE)):
                axis.plot([r['k'] for r in rows[layer]], numeric([r[key] for r in rows[layer]], key), 'o-', ms=3, color=colour, label=f'L{layer}')
            axis.set(xlabel='TopK sparsity k', ylabel=ylabel, xscale='log', xticks=[8, 16, 32, 64, 128])
            axis.set_xticklabels(['8', '16', '32', '64', '128'])
            axis.legend(frameon=False, fontsize=6)
        for layer, colour in ((0, BLUE), (13, ORANGE)):
            axes[1, 1].plot([r['k'] for r in rows[layer]], [100*(r['rate']-r['floor']) for r in rows[layer]], 'o-', ms=3, color=colour, label=f'L{layer}')
        axes[1, 1].set(xlabel='TopK sparsity k', ylabel='Ontology annotation excess (points)', xscale='log')
        axes[1, 1].set_xticks([8, 16, 32, 64, 128], ['8', '16', '32', '64', '128'])
        axes[1, 1].legend(frameon=False, fontsize=6)
        axes[1, 2].plot([r['expansion'] for r in expansions], [r['ve'] for r in expansions], 'o-', color=BLUE)
        axes[1, 2].set(xlabel='Expansion at L13, k=32', ylabel='Held-out variance explained', xticks=[2, 4, 8])
        for index, axis in enumerate(axes.flat):
            labelled_title(axis, chr(ord('a')+index), ['Reconstruction', 'Feature firing', 'Firing frequency', 'Decoder overlap', 'Biological terms against chance', 'Dictionary expansion'][index])
        self.values.update(sparsity_sweep=rows, expansion_sweep=expansions)
        self.save(fig, 'S1')

    def supplement2(self):
        pooling = self.read(self.results / 'pool_comparison.json')
        annotation = [self.annotation(group, 13) for group in ('2b', 'pool_first', 'pool_last')]
        fig, axes = plt.subplots(1, 3, figsize=(7.5, 2.7), layout='constrained')
        axes[0].bar([0, 1], [pooling[k]['gene_similarity_matrix_correlation'] for k in ('first', 'last')], color=[BLUE, ORANGE])
        axes[0].set_xticks([0, 1], ['First token', 'Last token']); axes[0].set_ylabel('Correlation with mean-pooled gene similarity')
        axes[1].bar(range(3), [pooling[k]['sae_variance_explained'] for k in ('mean', 'first', 'last')], color=[GREEN, BLUE, ORANGE])
        axes[1].set_xticks(range(3), ['Mean', 'First', 'Last']); axes[1].set_ylabel('Variance explained by mean-trained SAE')
        axes[2].bar(range(3), [100*r['rate'] for r in annotation], color=[GREEN, BLUE, ORANGE])
        axes[2].plot(range(3), [100*r['floor'] for r in annotation], 'x', color=GREY, label='Matching random floor')
        axes[2].set_xticks(range(3), ['Mean', 'First', 'Last']); axes[2].set_ylabel('Live features with ontology term (%)')
        # Keep the floor legend above all bars so its marker cannot look like
        # an extra measured floor superimposed on the first-token bar.
        axes[2].set_ylim(0, 100)
        axes[2].legend(frameon=False, fontsize=6)
        for axis, letter, title in zip(axes, 'abc', ['Gene geometry at L13', 'Reconstruction across pooling rules', 'Biological annotation across pooling rules']):
            labelled_title(axis, letter, textwrap.fill(title, 29))
        self.values.update(pooling=pooling, annotation=annotation)
        self.save(fig, 'S2')

    def supplement3(self):
        layers = [0, 13, 22]
        groups = {'Full libraries': '2b', 'Legacy libraries': '2b_legacy'}
        annotations = {name: [self.annotation(group, layer) for layer in layers] for name, group in groups.items()}
        untrained = [self.annotation(f'untrained_L{layer:02d}', layer) for layer in layers]
        fig, axes = plt.subplots(1, 2, figsize=(7.5, 2.8), layout='constrained')
        for (label, rows), colour in zip(annotations.items(), (BLUE, ORANGE)):
            axes[0].plot(layers, [100*r['rate'] for r in rows], 'o-', color=colour, label=label)
            axes[0].plot(layers, [100*r['floor'] for r in rows], '--', color=colour, lw=.9, label=f'{label}: random floor')
        for label, rows, colour in [('Trained', annotations['Full libraries'], BLUE), ('Untrained', untrained, GREY)]:
            axes[1].plot(layers, [100*r['rate'] for r in rows], 'o-', color=colour, label=label)
            axes[1].plot(layers, [100*r['floor'] for r in rows], '--', color=colour, lw=.9, label=f'{label}: random floor')
        for axis, letter, title in zip(axes, 'ab', ['Database-library sensitivity', 'Untrained dictionary control']):
            axis.set(xlabel='Layer', ylabel='Live features with ontology term (%)', xticks=layers)
            axis.legend(frameon=False, fontsize=5.9)
            labelled_title(axis, letter, title)
        self.values.update(library_annotations=annotations, untrained_annotations=untrained)
        self.save(fig, 'S3')

    def supplement4(self):
        record = self.read(self.results / 'circuit_stability.json')
        comparison = record['comparison']
        fig, axes = plt.subplots(1, 2, figsize=(7.5, 2.8), layout='constrained')
        axes[0].bar(range(3), [comparison[key] for key in ('n_first', 'n_second', 'n_shared')], color=[BLUE, ORANGE, GREEN])
        axes[0].set_ylim(0, 1.25*max(comparison[key] for key in ('n_first', 'n_second', 'n_shared')))
        axes[0].set_xticks(range(3), ['50 cells', '100 cells', 'Shared effects']); axes[0].set_ylabel('Effects passing the edge rule')
        axes[0].text(.02, .95, f'Jaccard: {comparison["jaccard"]:.3f}', transform=axes[0].transAxes, va='top', fontsize=6.5)
        for power in record['edge_rule_sensitivity']:
            axes[1].plot([r['population_standardized_effect'] for r in power['curve']], [r['call_probability'] for r in power['curve']], label=f'{power["n_active_cells"]} active cells')
        axes[1].set(xlabel='Population mean / SD (normal deltas)', ylabel='Probability edge rule is passed', ylim=(0, 1.02))
        axes[1].legend(frameon=False, fontsize=6)
        labelled_title(axes[0], 'a', 'Circuit edge stability')
        labelled_title(axes[1], 'b', 'Conditional edge-rule sensitivity')
        self.values.update(stability=record)
        self.save(fig, 'S4')

    @staticmethod
    def dimension_panel(axis, document, letter, title, metric='participation_ratio'):
        rows = document['rows']
        values = numeric([row[metric] for row in rows], metric)
        labels = [row['representation'].replace('expression_', 'Expr: ').replace('model_', 'Model ').replace('_shape_iid_normal', ' iid').replace('_', ' ') for row in rows]
        positions = np.arange(len(rows))[::-1]
        colours = [BLUE if row['representation'].startswith('model_') and 'iid' not in row['representation'] else ORANGE if 'shuffled' not in row['representation'] and 'iid' not in row['representation'] else GREY for row in rows]
        # Points on a logarithmic axis keep the measured and Gaussian controls
        # readable across orders of magnitude without implying a zero bar base.
        if np.any(values <= 0):
            raise ValueError('Logarithmic dimension panels require positive saved estimates')
        axis.scatter(values, positions, s=20, color=colours)
        axis.set_xscale('log')
        axis.set_yticks(positions, [textwrap.fill(label, 29) for label in labels], fontsize=6.3)
        axis.set_ylim(-.7, len(rows)-.3)
        axis.set_xlabel(('Centered covariance participation ratio' if metric=='participation_ratio' else 'TwoNN intrinsic dimension')+' (log scale)')
        labelled_title(axis, letter, title)

    def supplement5(self):
        document = self.read(self.results / 'cd34_dimensions.json')
        fig, axes = plt.subplots(1, 2, figsize=(7.5, 4.5), layout='constrained')
        self.dimension_panel(axes[0], document, 'a', 'CD34+ covariance dimensionality')
        self.dimension_panel(axes[1], document, 'b', 'CD34+ local dimensionality', 'twonn_id')
        self.values.update(dimensions=document)
        self.save(fig, 'S5')

    def supplement6(self):
        paths = sorted(self.args.stage_logs_dir.glob('*.complete.json'))
        if not paths:
            raise FileNotFoundError(f'No completed-stage measurements in {self.args.stage_logs_dir}')
        records = [self.read(path) for path in paths]
        records = [record for record in records if record['command']]
        if not records:
            raise ValueError('Completed stage records contain no measured executable stages')
        # Show the twelve longest measured stages; never reconstruct missing timing from clock dates.
        records = sorted(records, key=lambda record: -record['seconds'])[:12]
        fig, axes = plt.subplots(1, 3, figsize=(7.5, 4.1), layout='constrained')
        labels = [textwrap.fill(record['name'], 23) for record in records]
        y = np.arange(len(records))[::-1]
        for axis, key, scale, xlabel, letter in zip(axes, ['seconds', 'peak_device_memory_used_mib', 'peak_process_tree_rss_bytes'],
                                                   [60, 1024, 1e9], ['Wall time (minutes)', 'Peak GPU memory used (GiB)', 'Peak process-tree RAM (GB)'], 'abc'):
            axis.barh(y, numeric([record[key]/scale for record in records], key), color=BLUE, height=.7)
            axis.set_yticks(y, labels if letter=='a' else ['']*len(y), fontsize=5.8)
            axis.set_xlabel(xlabel, fontsize=6.5)
            labelled_title(axis, letter, ['Measured stage time', 'Device memory', 'Host memory'][ord(letter)-ord('a')])
        self.values.update(longest_completed_stages=records, selection='12 longest seconds among completed executable stages')
        self.save(fig, 'S6')


def required_paths(args, name):
    r, runs = args.results_dir, args.runs_dir
    ann = lambda group, layers: [r / 'annotations' / group / f'ann_v2_L{layer:02d}.json' for layer in layers]
    if name=='2':
        return [*[r/f'2b_eval_L{layer:02d}.json' for layer in LAYERS], *ann('2b', LAYERS), runs/'2b_atlas/cross_layer_tracking.json']
    if name=='3': return [r/'biology.json', r/'lines2.json', r/'lines2_freq.npz']
    if name=='4': return [runs/'2b_atlas/circuit_graph.json', r/'circuit_baseline.json', *[r/f'phase8_{line}/statistics/regulatory_statistics.json' for line in ('k562', 'rpe1')]]
    if name=='5': return [r/'residual_summary.json', *ann('2b', MATCH_2B), *ann('27b_matched', MATCH_27B)]
    if name=='6': return [args.geometry_dir/'gene_manifold_null.json', args.route_dir/'manifold_steer_L21.json', r/'selection_matched_transfer.json']
    if name=='S1':
        paths = [*[r/f'2b_eval_L{layer:02d}.json' for layer in (0, 13)], *[runs/f'2b_atlas/sae/train_L{layer:02d}.json' for layer in (0, 13)], *ann('2b', [0, 13])]
        settings = [(layer, k, 4) for layer in (0, 13) for k in (8, 16, 64, 128)]+[(13, 32, x) for x in (2, 8)]
        for layer, k, x in settings:
            group = f'sweep_L{layer:02d}_k{k}_x{x}'
            paths += [runs/group/'sae'/f'train_L{layer:02d}.json', *ann(group, [layer])]
        return paths
    if name=='S2': return [r/'pool_comparison.json', *[path for group in ('2b', 'pool_first', 'pool_last') for path in ann(group, [13])]]
    if name=='S3': return [*[path for group in ('2b', '2b_legacy') for path in ann(group, [0, 13, 22])], *[path for layer in (0, 13, 22) for path in ann(f'untrained_L{layer:02d}', [layer])]]
    if name=='S4': return [r/'circuit_stability.json']
    if name=='S5': return [r/'cd34_dimensions.json']
    if name=='S6': return sorted(args.stage_logs_dir.glob('*.complete.json')) or [args.stage_logs_dir/'<stage>.complete.json']
    raise ValueError(f'Unknown figure {name}')


def synthetic_inputs(args):
    """Construct isolated, explicitly marked layout fixtures, never submission artifacts."""
    base = args.synthetic_preview.resolve()
    final = Path(__file__).resolve().parent/'figures'
    if base==final or base.is_relative_to(final):
        raise ValueError('Synthetic previews must be outside the final figures directory')
    args.results_dir, args.runs_dir = base/'inputs/results', base/'inputs/runs'
    args.route_dir, args.geometry_dir, args.stage_logs_dir = base/'inputs/route', base/'inputs/geometry', base/'inputs/stage_logs'
    args.out_dir = base/'figures'
    def write(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({'synthetic': True, **value}))
    def annotation(group, layer, rate):
        write(args.results_dir/'annotations'/group/f'ann_v2_L{layer:02d}.json', {'summary': {'n_alive': 9000, 'n_features_total': 9216, 'ontology_annotation_rate_alive': rate, 'floors': {'freq': {'ontology_mean': .014}}}})
    for layer in LAYERS:
        write(args.results_dir/f'2b_eval_L{layer:02d}.json', {'variance_explained_reextracted': .99-.004*layer,
              'dead_val_reextracted': 20+layer*3, 'dead_frac_all_rows': 0., 'n_features': 9216,
              'firing_freq_quantiles_1_10_50_90_99': [.0001, .001, .003, .01, .03]})
        write(args.runs_dir/'2b_atlas/sae'/f'train_L{layer:02d}.json', {'mean_decoder_cos': .05})
        annotation('2b', layer, .18-.003*layer)
    write(args.runs_dir/'2b_atlas/cross_layer_tracking.json', {'config': {'match_threshold': .7}, 'long_range_persistence': [{'target_layer': layer, 'persistence_rate': 1-.018*layer} for layer in LAYERS]})
    categories = EXAMPLE_CATEGORIES+['RNA processing', 'DNA replication and repair', 'other']
    biology, line_arrays = [], {}
    lines = ['k562', 'rpe1']
    for layer in LAYERS:
        examples = {category: {'feature_idx': index, 'term': {'term': f'Example {category} process', 'q': .0001, 'fold': 12.}, 'top_genes': ['GENEA', 'GENEB', 'GENEC', 'GENED']} for index, category in enumerate(EXAMPLE_CATEGORIES)}
        biology.append({'layer': layer, 'category_share_all_features': {category: .015+.008*((index+layer)%4) for index, category in enumerate(categories)}, 'examples': examples})
    write(args.results_dir/'biology.json', {'layers': biology, 'categories': categories})
    write(args.results_dir/'lines2.json', {'lines': lines})
    for index, line in enumerate(lines): line_arrays[f'L13_{line}'] = np.array([.02, .04, .03])*(index+1)/3
    np.savez(args.results_dir/'lines2_freq.npz', **line_arrays)
    bootstrap = {'observed': {'estimate': .65, 'ci95': [.58, .72]}, 'baseline': {'estimate': .42, 'ci95': [.36, .48]}, 'excess': {'estimate': .23, 'ci95': [.17, .29]}}
    write(args.results_dir/'circuit_baseline.json', {'source_bootstrap': bootstrap})
    # Explicitly synthetic capped graph: plotting code derives counts from these records.
    synthetic_edges = [{'source_layer': source, 'source_feature': feature, 'target_layer': target,
                        'target_feature': source*1000+target*50+feature}
                       for source in (0, 7, 16) for target in range(source+1, 26)
                       for feature in range(max(2, 20-(target-source)//2))]
    write(args.runs_dir/'2b_atlas/circuit_graph.json', {'n_edges':len(synthetic_edges), 'edges':synthetic_edges})
    def retrieval(level):
        return {'mean_average_precision': level, 'mean_average_precision_ci95': [level*.8, level*1.2], 'mean_prevalence': level/3,
                'mean_random_label_expected_ap': level/2,
                'mean_auroc': .62, 'mean_auroc_ci95': [.56, .68]}
    families = [{'representation': rep, 'layer': layer, 'n_tfs': 48, 'n_specific': n,
                 'gene_retrieval': retrieval(.045), 'feature_retrieval': retrieval(.18)}
                for rep, layer, n in [('expression', None, 12), ('input', None, 8), ('model', 8, 2), ('model', 16, 3), ('model', 22, 2)]]
    for line in ('k562', 'rpe1'): write(args.results_dir/f'phase8_{line}/statistics/regulatory_statistics.json', {'summary': families})
    residual = {}
    for model, count in [('2b', 26), ('27b_matched', 46)]:
        residual[model] = {'layers': [{'layer': layer, 'dead_fraction_heldout': (.005+.001*layer if model=='2b' else .2+.004*layer),
                                      'centered_covariance_pr_fraction_width': .2/(1+layer/10), 'rms_max_over_median': 2+layer/3} for layer in range(count)]}
    write(args.results_dir/'residual_summary.json', residual)
    for layer in MATCH_27B: annotation('27b_matched', layer, .11-layer*.0008)
    write(args.geometry_dir/'gene_manifold_null.json', {'layers': [{'layer': layer, 'coexpr_matched': {'within_bin_mean_cos_delta': .04+index*.01, 'coexpr_null_mean': .001, 'coexpr_null_std': .008}} for index, layer in enumerate((9, 13, 17))]})
    write(args.route_dir/'manifold_steer_L21.json', {'layer': 21, 'rows': [{'traj': {arm: [{'phase': value} for value in np.linspace(0, scale*(1+index/30), 9)] for arm, scale in [('manifold', 1.7), ('linear', .9), ('random', .2)]}} for index in range(24)]})
    write(args.results_dir/'selection_matched_transfer.json', {'schema': 'selection-matched-frozen-transfer-v1', 'layer': 21,
          'point': {key: {'R_diff': value} for key, value in [('expression_full_shared', .88), ('expression_topk_magnitude_fullpanel', .81), ('expression_topk_rank_fullpanel', .77), ('model', .79), ('constant', .04), ('random', .02)]},
          'contrasts': {'model_minus_expression_topk_rank_fullpanel': {'point_delta': .02, 'ci95': [-.01, .05]}}})
    settings = [(layer, k, 4) for layer in (0, 13) for k in (8, 16, 64, 128)]+[(13, 32, x) for x in (2, 8)]
    for layer, k, expansion in settings:
        group = f'sweep_L{layer:02d}_k{k}_x{expansion}'
        write(args.runs_dir/group/'sae'/f'train_L{layer:02d}.json', {'variance_explained': .7+np.log2(k)*.025, 'dead_frac': .01, 'firing_freq_quantiles_1_10_50_90_99': [.0001, .001, .003, .01, .03], 'mean_decoder_cos': .05})
        annotation(group, layer, .14)
    for layer in (0, 13, 22):
        annotation('2b_legacy', layer, .13)
        annotation(f'untrained_L{layer:02d}', layer, .04)
    pooling = {'mean': {'sae_variance_explained': .94}, 'first': {'sae_variance_explained': .91, 'gene_similarity_matrix_correlation': .95}, 'last': {'sae_variance_explained': .92, 'gene_similarity_matrix_correlation': .96}}
    write(args.results_dir/'pool_comparison.json', pooling)
    for group in ('pool_first', 'pool_last'): annotation(group, 13, .15)
    write(args.results_dir/'circuit_stability.json', {'comparison': {'n_first': 2000, 'n_second': 2600, 'n_shared': 1800, 'jaccard': 1800/2800},
          'edge_rule_sensitivity': [{'n_active_cells': n, 'curve': [{'population_standardized_effect': float(effect), 'call_probability': float(1/(1+np.exp(-(effect-.7)*n/8)))} for effect in np.linspace(0, 1.5, 31)]} for n in (50, 100)]})
    write(args.results_dir/'cd34_dimensions.json', {'rows': [{'representation': label, 'participation_ratio': pr, 'twonn_id': twonn} for label, pr, twonn in [('expression_log1p_CP10k', 30, 5), ('expression_cells_shuffled_within_gene', 150, 20), ('expression_shape_iid_normal', 250, 35), ('model_L09', 90, 7), ('model_L09_shape_iid_normal', 300, 40)]]})
    for index in range(12): write(args.stage_logs_dir/f'example-{index}.complete.json', {'name': f'example-stage-{index}', 'command': ['synthetic'], 'seconds': 60*(index+1), 'peak_device_memory_used_mib': 1024*(index+1), 'peak_process_tree_rss_bytes': 1e9*(index+1)})


def main():
    paper = Path(__file__).resolve().parent
    project = paper.parent
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--figures', nargs='+', default=MAIN, help='2 3 4 5 6 S1 S2 S3 S4 S5 S6, or all')
    parser.add_argument('--results-dir', type=Path, default=paper/'experiments/results')
    parser.add_argument('--runs-dir', type=Path, default=project/'runs')
    parser.add_argument('--route-dir', type=Path, default=project/'route_genemanifold_c2s/results')
    parser.add_argument('--geometry-dir', type=Path, default=project/'geometry')
    parser.add_argument('--stage-logs-dir', type=Path, default=paper/'experiments/stage_logs')
    parser.add_argument('--out-dir', type=Path, default=paper/'figures')
    parser.add_argument('--example-layer', type=int, default=13)
    parser.add_argument('--example-categories', nargs=3, default=EXAMPLE_CATEGORIES)
    parser.add_argument('--list-required', action='store_true', help='List required artifacts and optional CD34+ panel without plotting')
    parser.add_argument('--synthetic-preview', type=Path, help='Create labelled synthetic fixtures and figures outside final figures')
    args = parser.parse_args()
    if args.figures==['all']: args.figures = MAIN+SUPPLEMENT
    if any(name not in MAIN+SUPPLEMENT for name in args.figures): parser.error('Unknown figure name; use 2-6 or S1-S6')
    if args.synthetic_preview: synthetic_inputs(args)
    if args.list_required:
        for name in args.figures:
            print(f'Figure {name}:')
            for path in required_paths(args, name): print(f'  {path}')
            if name=='6': print(f'  OPTIONAL panel: {args.results_dir / "cd34_dimensions.json"}')
        return
    missing = sorted({str(path) for name in args.figures for path in required_paths(args, name) if not path.is_file()})
    if missing:
        parser.error('Missing saved artifacts; no requested figures were generated:\n'+ '\n'.join(missing))
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 7.3, 'axes.labelsize': 7,
                         'axes.titlesize': 8.2, 'xtick.labelsize': 6.5, 'ytick.labelsize': 6.5,
                         'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': .65,
                         'lines.linewidth': 1.3, 'pdf.fonttype': 42, 'ps.fonttype': 42,
                         'svg.fonttype': 'none', 'axes.unicode_minus': False, 'savefig.facecolor': 'white'})
    warnings.filterwarnings('error', message='Glyph .* missing from font')
    methods = {str(number): f'figure{number}' for number in range(2, 7)}
    methods.update({f'S{number}': f'supplement{number}' for number in range(1, 7)})
    for name in args.figures:
        generator = Figures(args)
        try:
            getattr(generator, methods[name])()
        except (KeyError, ValueError, StopIteration) as error:
            raise SystemExit(f'Figure {name}: required artifact schema/value is missing or invalid: {error}') from error


if __name__=='__main__':
    main()
