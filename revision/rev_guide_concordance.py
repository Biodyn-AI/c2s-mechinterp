"""Descriptive guide concordance for the fixed, staged C6 THP1 cohort; no model inference.

Cells within a guide are observations for the existing response rule, not independent
perturbation replicates. Pairwise guide summaries share the same control sample and are
not independent. No guide-level confidence interval or regulatory-direction claim is made.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
from itertools import combinations
import json
from pathlib import Path

import numpy as np
from rev_phase8_stats import (response_statistics, downstream_target_labels, feature_gene_scores,
                             retrieval_metrics, load_targets)


def align_selected_rows(design, rows, source_targets, source_batches):
    """Require the whole fixed design; preserve extraction order when indexing features."""
    expected = {}
    for group in design['groups']:
        for cell in group['cells']:
            if cell in expected: raise ValueError('Design groups share cells')
            expected[cell] = (group['group'], group['kind'])
    if len(rows)!=len(expected) or len({row['cell_index'] for row in rows})!=len(rows):
        raise ValueError('Extraction must contain each selected design cell exactly once')
    groups, batches = {}, set()
    for index, row in enumerate(rows):
        cell = row['cell_index']
        if cell not in expected or expected[cell]!=(row['group'],row['kind']):
            raise ValueError('Extraction group/cell/kind does not match fixed design')
        if not 0<=cell<len(source_targets): raise ValueError('Cell index outside staged h5ad')
        target = 'control' if row['kind'] in ('control','pseudo') else row['group']
        if str(source_targets[cell]).upper()!=target.upper():
            raise ValueError('Staged h5ad target label does not match fixed design')
        batches.add(str(source_batches[cell]))
        groups.setdefault(row['group'],[]).append(index)
    if len(batches)!=1: raise ValueError('C6 guide audit requires one recorded batch partition')
    return {group:np.asarray(indices,int) for group,indices in groups.items()}


def compare_guide_effects(first, second, effect_threshold=.5):
    """Compare feature-response signs, never TF activation/repression signs."""
    left, right = np.asarray(first['effect'],float), np.asarray(second['effect'],float)
    a, b = np.asarray(first['responding'],bool), np.asarray(second['responding'],bool)
    if left.ndim!=1 or any(value.shape!=left.shape for value in (right,a,b)) or not np.isfinite([left,right]).all():
        raise ValueError('Guide feature statistics must be finite aligned vectors')
    both, either = a & b, a | b
    supported = (np.abs(left)>effect_threshold) & (np.abs(right)>effect_threshold)
    correlation = float(np.corrcoef(left,right)[0,1]) if left.std()>0 and right.std()>0 else None
    return {'n_features':len(left), 'effect_pearson':correlation,
            'n_joint_responses':int(both.sum()), 'n_union_responses':int(either.sum()),
            'response_jaccard':float(both.sum()/either.sum()) if either.any() else None,
            'joint_response_sign_agreement':float((np.sign(left[both])==np.sign(right[both])).mean()) if both.any() else None,
            'n_both_abs_effect_above_threshold':int(supported.sum()),
            'effect_supported_sign_agreement':float((np.sign(left[supported])==np.sign(right[supported])).mean()) if supported.any() else None}


def guide_profiles(features, indices, controls, row_guides, factor, known_targets, catalogue, genes,
                   min_cells=3, alpha=.05, effect_threshold=.5):
    if min_cells<3: raise ValueError('At least three cells are needed by the existing response rule')
    known, labels, candidates = downstream_target_labels(genes,known_targets,factor)
    guides, arrays = [], []
    for guide in sorted(set(row_guides[indices])):
        selected = indices[row_guides[indices]==guide]
        record = {'guide':str(guide),'n_cells':len(selected),'n_control_cells':len(controls),
                  'status':'insufficient cells' if len(selected)<min_cells else 'evaluable'}
        if len(selected)>=min_cells:
            statistics = response_statistics(features[selected],features[controls],alpha,effect_threshold)
            gene_scores,_ = feature_gene_scores(np.abs(statistics['effect']),catalogue,genes)
            record.update(n_responding=int(statistics['responding'].sum()),
                          gene_retrieval=retrieval_metrics(labels[candidates],gene_scores[candidates],bootstrap=0))
            arrays.append({'guide':str(guide),**statistics})
        guides.append(record)
    pairs = [{'first_guide':first['guide'],'second_guide':second['guide'],
              **compare_guide_effects(first,second,effect_threshold)} for first,second in combinations(arrays,2)]
    return {'group':factor,'n_cells':len(indices),'n_guides_selected':len(guides),
            'n_guides_evaluable':len(arrays),'n_known_targets_excluding_self':len(known),
            'guides':guides,'pairs':pairs},arrays


def file_hash(path):
    digest = hashlib.sha256()
    with open(path,'rb') as handle:
        for chunk in iter(lambda:handle.read(4<<20),b''): digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--h5ad',type=Path,required=True)
    parser.add_argument('--design',type=Path,required=True)
    parser.add_argument('--features-dir',type=Path,required=True)
    parser.add_argument('--catalog-dir',type=Path,required=True)
    parser.add_argument('--trrust',type=Path,required=True)
    parser.add_argument('--layers',type=int,nargs='+',default=[8,16,22])
    parser.add_argument('--min-cells-per-guide',type=int,default=3)
    parser.add_argument('--out',type=Path,required=True)
    args = parser.parse_args()
    if args.min_cells_per_guide<3: parser.error('--min-cells-per-guide must be at least 3')
    import anndata
    design = json.loads(args.design.read_text())
    rows_path = args.features_dir/'rows.json'
    rows = json.loads(rows_path.read_text())
    adata = anndata.read_h5ad(args.h5ad,backed='r')
    for column in ('guide_id','c6_gene','c6_recorded_batch'):
        if column not in adata.obs: raise ValueError(f'Staged C6 h5ad lacks {column}')
    targets = adata.obs['c6_gene'].astype(str).to_numpy()
    batches = adata.obs['c6_recorded_batch'].astype(str).to_numpy()
    guides = adata.obs['guide_id'].astype(str).to_numpy()
    genes = [str(gene).upper() for gene in adata.var_names]
    if len(set(genes))!=len(genes): raise ValueError('Resolve all uppercase gene collisions before guide retrieval')
    group_indices = align_selected_rows(design,rows,targets,batches)
    row_guides = guides[[row['cell_index'] for row in rows]]
    controls = [group for group in design['groups'] if group['kind']=='control']
    if len(controls)!=1: raise ValueError('Exactly one fixed control group is required')
    control_indices = group_indices[controls[0]['group']]
    factors = [group for group in design['groups'] if group['kind']=='tf']
    if not factors or len(control_indices)<3: raise ValueError('Nonempty fixed TF/control groups are required')
    # Verify retained guide identities against the pre-outcome sampling manifest.
    for group in design['groups']:
        selected_counts = dict(Counter(row_guides[group_indices[group['group']]]))
        if selected_counts!=group['guide_counts_selected']:
            raise ValueError(f'Guide counts differ from selected design: {group["group"]}')
    known_targets = load_targets(args.trrust)
    source_paths = [args.h5ad,args.design,rows_path,args.trrust,Path(__file__),Path(__file__).with_name('rev_phase8_stats.py')]
    results = []
    args.out.parent.mkdir(parents=True,exist_ok=True)
    for layer in args.layers:
        features_path = args.features_dir/f'feat_L{layer:02d}.npz'
        catalogue_path = args.catalog_dir/f'feature_catalog_L{layer:02d}.json'
        source_paths.extend([features_path,catalogue_path])
        with np.load(features_path,allow_pickle=False) as archive: features = archive['X']
        if features.ndim!=2 or len(features)!=len(rows) or not np.isfinite(features).all():
            raise ValueError('Feature matrix must be finite and aligned to extracted rows')
        catalogue = json.loads(catalogue_path.read_text())['features']
        if sorted(int(row['feature_idx']) for row in catalogue)!=list(range(features.shape[1])):
            raise ValueError('Catalogue must cover each feature column exactly once')
        saved = []
        for group in factors:
            profile,arrays = guide_profiles(features,group_indices[group['group']],control_indices,row_guides,
                  group['group'],known_targets.get(group['group'].upper(),set()),catalogue,genes,args.min_cells_per_guide)
            results.append({'layer':layer,**profile})
            saved.extend([(group['group'],record) for record in arrays])
        if saved:
            np.savez_compressed(args.out.parent/f'guide_statistics_L{layer:02d}.npz',
                  groups=np.asarray([group for group,_ in saved]), guides=np.asarray([row['guide'] for _,row in saved]),
                  effect=np.asarray([row['effect'] for _,row in saved],np.float32),
                  responding=np.asarray([row['responding'] for _,row in saved],bool))
    adata.file.close()
    output = {'experiment':'C6-guide-concordance','minimum_cells_per_guide':args.min_cells_per_guide,
              'n_fixed_control_cells':len(control_indices),'batch':sorted(set(batches.tolist())),
              'control_extracted_row_indices':control_indices.tolist(),
              'control_h5ad_cell_indices':[rows[index]['cell_index'] for index in control_indices],
              'records':results,'source_sha256':{str(path):file_hash(path) for path in source_paths},
              'interpretation':{
                  'selection':'All prespecified TF groups/guides retained; insufficient-cell guides explicitly unevaluable',
                  'response':'Existing per-guide cell-level Mann-Whitney/BH across all features and absolute d>0.5',
                  'sign':'Feature response change sign only; not regulatory activation/repression',
                  'retrieval':'Absolute feature d mapped through fixed catalogue to the same h5ad gene universe; direct factor excluded',
                  'uncertainty':'Descriptive only; no guide-level CI, no resampling of two/few guides, no paired-significance claim',
                  'dependence':'All guide pairs share fixed controls; pairs and feature columns are not independent replicates',
                  'low_power':'Few cells per guide can yield no shared significant responses; empty-union Jaccard is undefined',
                  'scope':'One recorded HTO partition; not independent culture/batch replication'}}
    args.out.write_text(json.dumps(output,indent=2,allow_nan=False))
    print(f'{args.out}: {len(results)} fixed TF/layer guide profiles',flush=True)


if __name__=='__main__':main()
