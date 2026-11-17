"""Package completed analysis artifacts, with new SAE weights kept separate.

Only explicit completed groups are selected. Activation memmaps, count matrices,
foundation-model weights, access material and in-progress compute runs
are never selected. Archive members use paths relative to --root.
"""
import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import re
import tarfile
import tempfile
import time

MAX_ASSET_BYTES = 2_000_000_000
PAYLOAD_BYTES = 1_800_000_000
MANIFEST_NAME = 'COLLECTION_MANIFEST.json'
SECRET = re.compile(rb'(?:hf_|rpa_)[A-Za-z0-9]{24,}|(?i:authorization:\s*bearer\s+)[A-Za-z0-9._-]{24,}')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(4*1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def signature(path):
    stat = Path(path).stat()
    return [stat.st_size, stat.st_mtime_ns, stat.st_ino]


def stage_ready(root, name):
    path = root/'stage_logs'/f'{name}.complete.json'
    if not path.is_file():
        return False
    record = json.loads(path.read_text())
    if record.get('returncode') != 0 or not record.get('end_utc'):
        return False
    attempt = root/'stage_logs'/f'{name}.attempt.json'
    if attempt.is_file():
        latest = json.loads(attempt.read_text())
        if latest.get('fingerprint') != record.get('fingerprint') or latest['start_utc'] > record['start_utc']:
            return False
    return all(Path(output).exists() for output in record.get('outputs', []))


def validate_path(root, path, category):
    path = Path(path).absolute()
    require(not path.is_symlink() and path.resolve() == path, f'symlink must be recorded as a dependency: {path}')
    require(path.is_file() and path.is_relative_to(root), f'not a regular root-contained file: {path}')
    name = path.name.lower()
    require(not any(token in name for token in ('credential', 'password', 'api_key', 'runpod_access')) and not name.startswith('.env'),
            f'access material is forbidden: {path}')
    require(path.suffix.lower() not in ('.h5ad', '.safetensors', '.bin') and '_activations.npy' not in name
            and not name.endswith(('.mtx', '.mtx.gz')), f'raw counts/model weights/activations forbidden: {path}')
    if category == 'local_only_weights':
        require(path.suffix == '.pt' and name.startswith('sae_l'), f'only new SAE checkpoints belong in weights archive: {path}')
    else:
        require(path.suffix != '.pt', f'weights cannot enter a scientific archive: {path}')
    return path


def snapshot_files(root, paths, prior, category='scientific'):
    original_root = Path(root).absolute()
    root = original_root.resolve()
    selected, repeated = {}, []
    for path in sorted(set(map(Path, paths))):
        # macOS /var is itself a system symlink; reject links below the supplied
        # root without rejecting that harmless spelling of the root directory.
        path = root / path.absolute().relative_to(original_root)
        path = validate_path(root, path, category)
        before = signature(path)
        digest = hashlib.sha256()
        tail = b''
        text = path.suffix.lower() in ('.json', '.log', '.txt', '.tsv')
        with path.open('rb') as handle:
            for block in iter(lambda: handle.read(4*1024*1024), b''):
                digest.update(block)
                if text:
                    require(not SECRET.search(tail + block), f'possible credential value in {path}; not collected')
                    tail = block[-256:]
        require(signature(path) == before, f'source changed during snapshot: {path}')
        relative = str(path.relative_to(root))
        hexdigest = digest.hexdigest()
        if relative in prior and prior[relative]['sha256'] == hexdigest:
            repeated.append(relative)
            continue
        selected[relative] = {'bytes': before[0], 'sha256': hexdigest, 'source_stat': before,
                              'category': category}
        if relative in prior:
            selected[relative]['supersedes_prior_sha256'] = prior[relative]['sha256']
    return selected, repeated


def partition_files(files, max_payload_bytes=PAYLOAD_BYTES):
    parts, current, size = [], {}, 0
    for name, record in sorted(files.items()):
        require(record['bytes'] <= max_payload_bytes, f'single source exceeds archive payload budget: {name}')
        if current and size + record['bytes'] > max_payload_bytes:
            parts.append(current)
            current, size = {}, 0
        current[name] = record
        size += record['bytes']
    if current:
        parts.append(current)
    return parts


def verify_archive(path, expected_sha256):
    require(sha_file(path) == expected_sha256, f'archive SHA256 mismatch: {path}')
    with tarfile.open(path, 'r|gz') as archive:
        first = archive.next()
        require(first is not None and first.name == MANIFEST_NAME and first.isfile(), 'manifest must be first archive member')
        manifest = json.load(archive.extractfile(first))
        seen, n_bytes = set(), 0
        for member in archive:
            if member.name == MANIFEST_NAME:
                continue
            require(member.isfile() and member.name in manifest['files'] and member.name not in seen,
                    f'unexpected/nonregular/duplicate archive member: {member.name}')
            expected = manifest['files'][member.name]
            require(member.size == expected['bytes'], f'archive member size mismatch: {member.name}')
            digest = hashlib.sha256()
            with archive.extractfile(member) as handle:
                for block in iter(lambda: handle.read(4*1024*1024), b''):
                    digest.update(block)
            require(digest.hexdigest() == expected['sha256'], f'archive member SHA256 mismatch: {member.name}')
            seen.add(member.name)
            n_bytes += member.size
        require(seen == set(manifest['files']), 'archive is missing manifested members')
    return {'n_files': len(seen), 'bytes': n_bytes, 'manifest': manifest}


def write_archive(root, files, out, category, metadata):
    root = Path(root).resolve()
    require(not out.exists(), f'archive already exists: {out}')
    temporary = out.with_suffix(out.suffix + '.partial')
    require(not temporary.exists(), f'partial archive already exists: {temporary}')
    manifest = {'schema': 'completed-artifact-archive-v1', 'category': category,
        'files': files, 'metadata': metadata}
    raw_manifest = (json.dumps(manifest, sort_keys=True, indent=2) + '\n').encode()
    with temporary.open('wb') as raw:
        with gzip.GzipFile(fileobj=raw, filename='', mode='wb', compresslevel=3, mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode='w|', format=tarfile.PAX_FORMAT) as archive:
                info = tarfile.TarInfo(MANIFEST_NAME)
                info.size = len(raw_manifest)
                archive.addfile(info, io.BytesIO(raw_manifest))
                for name, record in sorted(files.items()):
                    path = root/name
                    require(signature(path) == record['source_stat'], f'source changed before archive: {name}')
                    info = tarfile.TarInfo(name)
                    info.size = record['bytes']
                    info.mtime = path.stat().st_mtime
                    info.mode = 0o644
                    with path.open('rb') as handle:
                        archive.addfile(info, handle)
                    require(signature(path) == record['source_stat'], f'source changed during archive: {name}')
    require(temporary.stat().st_size < MAX_ASSET_BYTES, f'archive reaches 2GB: {out}')
    digest = sha_file(temporary)
    verified = verify_archive(temporary, digest)
    temporary.rename(out)
    return {'name': out.name, 'bytes': out.stat().st_size, 'sha256': digest,
        'n_files': verified['n_files'], 'expanded_bytes': verified['bytes'], 'category': category}


def install_scientific(path, expected_sha256, destinations, conflict_root):
    """Verify first, then atomically install new science files without overwrites.

    Existing different files remain untouched; incoming versions go to a separate
    conflict directory. Weight archives are verified but never accepted here.
    """
    verified = verify_archive(path, expected_sha256)
    require(verified['manifest']['category'] == 'scientific', 'local-only weights must never be extracted')
    installed, matched, conflicts = 0, 0, []
    with tarfile.open(path, 'r|gz') as archive:
        for member in archive:
            if member.name == MANIFEST_NAME:
                continue
            relative = Path(member.name)
            require(not relative.is_absolute() and '..' not in relative.parts, 'unsafe member path')
            require(relative.parts[0] in destinations, f'no canonical destination for {member.name}')
            expected = verified['manifest']['files'][member.name]
            base = Path(destinations[relative.parts[0]]).resolve()
            target = base.joinpath(*relative.parts[1:])
            require(target.resolve().is_relative_to(base), f'destination escapes mapped root: {member.name}')
            if target.exists() and sha_file(target) == expected['sha256']:
                matched += 1
                continue
            if target.exists():
                original = target
                target = Path(conflict_root).resolve()/relative
                conflicts.append({'source': member.name, 'existing': str(original), 'incoming': str(target)})
                if target.exists():
                    require(sha_file(target) == expected['sha256'], f'existing conflict differs: {target}')
                    matched += 1
                    continue
            target.parent.mkdir(parents=True, exist_ok=True)
            digest = hashlib.sha256()
            with tempfile.NamedTemporaryFile(dir=target.parent, prefix='.collection-', delete=False) as output:
                temporary = Path(output.name)
                with archive.extractfile(member) as source:
                    for block in iter(lambda: source.read(4*1024*1024), b''):
                        digest.update(block)
                        output.write(block)
            require(digest.hexdigest() == expected['sha256'] and temporary.stat().st_size == expected['bytes'],
                    f'extracted member differs: {member.name}')
            try:
                # Linking a verified temporary file fails if a concurrent writer
                # has created the final path; it can never replace that file.
                os.link(temporary, target)
            except FileExistsError:
                require(sha_file(target) == expected['sha256'], f'concurrent destination change: {target}')
            finally:
                temporary.unlink()
            installed += 1
    return {'n_verified': verified['n_files'], 'expanded_bytes': verified['bytes'],
        'installed': installed, 'already_exact': matched, 'conflicts': conflicts}


def audit_ready(path):
    """Only explicitly passed, complete audit reports are frozen inputs."""
    try:
        before = signature(path)
        record = json.loads(path.read_text())
        flags = [record[key] for key in ('passed','all_checks_passed','all_arithmetic_checks_passed') if key in record]
        checks = record.get('checks')
        explicit_checks_pass = isinstance(checks, dict) and bool(checks) and all(value is True for value in checks.values())
        positive = any(value is True for value in flags) or explicit_checks_pass
        return (signature(path) == before and positive and not any(value is False for value in flags)
                and record.get('complete', True) is True
                and not record.get('pending_files') and not record.get('findings')
                and not record.get('unaudited_derived_ledger_entries'))
    except (OSError, ValueError, AttributeError):
        return False


def science_audit_ready(path):
    prefixes = ('annotation_', 'experimental_saved', 'phase8_', 'phb2_worked',
                'c6_', 'circuit', 'biology_lines', 'sensitivity_saved', 'e4_saved',
                'dimensions_', 'retained_saved')
    return path.name.startswith(prefixes) and 'partial' not in path.name and audit_ready(path)


def collect_plan(root, prior, scope='middle', required_audits=()):
    paths, weights, completed, pending, dependencies = set(), set(), {}, {}, {}

    def add(name, stages, patterns, weight_patterns=()):
        missing = [stage for stage in stages if not stage_ready(root, stage)]
        if missing:
            pending[name] = missing
            return
        completed[name] = stages
        for stage in stages:
            for suffix in ('.complete.json', '.attempt.json', '.log'):
                path = root/'stage_logs'/f'{stage}{suffix}'
                if path.is_file():
                    paths.add(path)
        for pattern in patterns:
            for path in root.glob(pattern):
                if path.is_symlink() or path.resolve() != path.absolute():
                    dependencies[str(path.relative_to(root))] = str(path.resolve().relative_to(root))
                    continue
                if not path.is_file():
                    continue
                if '_activations.npy' not in path.name and path.suffix != '.pt':
                    paths.add(path)
        for pattern in weight_patterns:
            weights.update(path for path in root.glob(pattern) if path.is_file())

    def finish():
        scientific, repeated = snapshot_files(root, paths, prior)
        checkpoints, _ = snapshot_files(root, weights, prior, category='local_only_weights')
        return {'scientific': scientific, 'local_only_weights': checkpoints, 'completed_groups': completed,
            'pending_groups': pending, 'omitted_exact_prior_repeats': repeated, 'symlink_dependencies': dependencies}

    layers = (0, 7, 14, 24, 31, 38, 45)
    if scope == 'annotations':
        add('matched_scale_annotations', [f'scale-annotate-L{layer:02d}' for layer in layers],
            ['results/annotations/27b_matched/*.json'])
        return finish()
    if scope == 'final':
        scale_stages = ['scale-extract', 'scale-residual-summary'] + [
            f'scale-{kind}-L{layer:02d}' for layer in layers for kind in ('train','catalog','annotate')]
        add('matched_scale', scale_stages, [
            'runs/27b_matched/*.json', 'runs/27b_matched/row_cell_ids.npy',
            'runs/27b_matched/svd_axes_L*.npy', 'runs/27b_matched/sae/*.json',
            'runs/27b_matched/sae/*.npy', 'results/annotations/27b_matched/*.json',
            'results/residual_summary.json', 'results/scale_reporting.json'], ['runs/27b_matched/sae/*.pt'])
        add('massive_dimensions', ['scale-dropmassive-presets', 'scale-dropmassive-train-L24',
            'scale-dropmassive-train-L38', 'scale-dropmassive-comparison'],
            ['results/dropmassive_dimensions/*', 'results/dropmassive_comparison.json',
             'runs/27b_matched/sae_dropmassive/*.json', 'runs/27b_matched/sae_dropmassive/*.npy'],
            ['runs/27b_matched/sae_dropmassive/*.pt'])
        add('circuit_replication', ['circuit-rerun-inputs','circuit-rerun-50','circuit-rerun-100','circuit-rerun-comparison'],
            ['results/circuit_50/**/*','results/circuit_100/**/*','results/circuit_stability.json','results/circuit_stability_full.json',
             'designs/circuit_original_features.json','designs/circuit_k562.json'])
        add('geometry_cd34', ['geometry-inputs','geometry-extract','geometry-gene-means','geometry-dimensions'],
            ['designs/geometry_cd34/*.json','designs/geometry_cd34/row_cell_ids.npy',
             'runs/geometry_cd34/*.json','runs/geometry_cd34/row_cell_ids.npy',
             'results/cd34_gene_means.npz','results/cd34_dimensions.json'])
        add('geometry_k562_replication', ['geometry-k562-extract','geometry-k562-gene-means','geometry-k562-dimensions'],
            ['designs/atlas_k562/*.json','designs/atlas_k562/row_cell_ids.npy',
             'runs/geometry_k562_replication/*.json','runs/geometry_k562_replication/row_cell_ids.npy',
             'results/k562_replication_gene_means.npz','results/k562_replication_dimensions.json'])
        for relative in required_audits:
            path = root/relative
            if not audit_ready(path):
                pending['audit:'+relative] = ['passed complete audit required']
            else:
                paths.add(path)
        # Existing final audits are copied only after the entire selected compute
        # scope completes. Partial/failed audit JSONs never become final assets.
        if not pending:
            for path in (root/'results/audits').glob('*.json'):
                if science_audit_ready(path):
                    paths.add(path)
            completed['passed_final_audits'] = [str(path.relative_to(root)) for path in sorted(paths)
                if path.parent == root/'results/audits']
        return finish()
    require(scope == 'middle', f'unknown collection scope: {scope}')

    for layer in (0, 13):
        for k in (8, 16, 64, 128):
            name = f'sweep_L{layer:02d}_k{k}_x4'
            prefix = name.replace('_', '-')
            add(name, [f'{prefix}-inputs', f'{prefix}-train', f'{prefix}-catalog-L{layer:02d}', f'{prefix}-annotate-L{layer:02d}'],
                [f'runs/{name}/**/*', f'results/annotations/{name}/*.json'], [f'runs/{name}/sae/*.pt'])
    for expansion in (2, 8):
        name, prefix = f'sweep_L13_k32_x{expansion}', f'sweep-L13-k32-x{expansion}'
        add(name, [prefix+'-inputs', prefix+'-train', prefix+'-catalog-L13', prefix+'-annotate-L13'],
            [f'runs/{name}/**/*', f'results/annotations/{name}/*.json'], [f'runs/{name}/sae/*.pt'])
    for layer in (0, 13, 22):
        name, prefix = f'untrained_L{layer:02d}', f'untrained-L{layer:02d}'
        add(name, [prefix+'-inputs', prefix+'-init', prefix+f'-catalog-L{layer:02d}', prefix+f'-annotate-L{layer:02d}'],
            [f'runs/{name}/**/*', f'results/annotations/{name}/*.json'], [f'runs/{name}/sae/*.pt'])
    for pool in ('first', 'last'):
        add(f'pool_{pool}', [f'pool-{pool}-extract', f'pool-{pool}-checkpoint', f'pool-{pool}-catalog-L13', f'pool-{pool}-annotate-L13'],
            [f'runs/pool_{pool}/**/*', f'results/annotations/pool_{pool}/*.json'])
    add('pool_comparison', ['pool-comparison'], ['results/pool_comparison.json'])
    for line in ('k562', 'rpe1'):
        add('phase8_'+line, [f'phase8-{line}-extract', f'phase8-{line}-stats', f'phase8-{line}-response-biology'],
            [f'results/phase8_{line}/**/*'])
    add('phase8_thp1_pooled', ['phase8-thp1-pooled-extract', 'phase8-thp1-pooled-stats',
        'phase8-thp1-pooled-biology', 'phase8-thp1-pooled-guide-concordance'],
        ['results/phase8_thp1_pooled/**/*', 'data/papalexi/c6_pooled_rep1tx/design.json',
         'data/papalexi/c6_pooled_rep1tx/provenance.json'])
    add('source_designs', ['phase8-original-design', 'phase8-rpe1-extract', 'lines-eval'],
        ['designs/*.json', 'designs/make_designs.log', 'designs/*/*.json', 'designs/*/row_cell_ids.npy',
         'runs/2b_atlas/row_gene_names.json', 'runs/2b_atlas/row_cell_ids.npy',
         'runs/2b_atlas/sae/train_L*.json', 'runs/2b_atlas/sae/freq_L*.npy', 'runs/2b_atlas/sae/mu_L*.npy'])
    add('legacy_libraries', [f'legacy-annotate-L{layer:02d}' for layer in (0, 13, 22)],
        ['results/annotations/2b_legacy/*.json', 'ontologies_v2/*', 'data/ontologies/*'])
    add('lines2', ['lines-extract', 'lines-labels', 'lines-eval'],
        ['runs/lines2/*.json', 'runs/lines2/row_cell_ids.npy', 'results/lines2.json', 'results/lines2_freq.npz'])
    add('biology', ['biology-summary'], ['results/biology.json', 'results/biology.json.matches.npz'])
    add('gene_means', ['atlas-gene-means'], ['results/atlas_gene_means.npz'])
    add('original_circuit_baseline', ['circuit-baseline'],
        ['results/circuit_baseline.json', 'runs/2b_atlas/circuit_graph.json'])
    add('finished_reporting', ['untrained-L22-annotate-L22', 'phase8-rpe1-response-biology'],
        ['results/atlas_reporting.json', 'results/sensitivity_reporting.json',
         'results/audits/phb2_worked_audit.json', 'results/audits/phb2_worked_claims.json',
         'results/audits/sensitivity_saved_v1.json', 'results/audits/phase8_k562_saved_v1.json',
         'results/audits/phase8_rpe1_saved_v1.json', 'results/audits/phase8_thp1_pooled_saved_v1.json',
         'results/audits/c6_guide_concordance_v1.json', 'results/audits/c6_pooled_metadata_v1.json'])
    return finish()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--prior-manifest', type=Path, action='append', required=True)
    parser.add_argument('--scope', choices=('middle','annotations','final'), default='middle')
    parser.add_argument('--label', help='Asset/manifest prefix; defaults to stage2, stage3_annotations or stage3')
    parser.add_argument('--required-audit', action='append', default=[], help='Root-relative audit JSON; must explicitly pass')
    parser.add_argument('--plan-only', action='store_true')
    args = parser.parse_args()
    started = time.monotonic()
    root = args.root.resolve()
    prior = {}
    for path in args.prior_manifest:
        prior.update(json.loads(path.read_text())['files'])
    label = args.label or {'middle':'stage2','annotations':'stage3_annotations','final':'stage3'}[args.scope]
    require(re.fullmatch(r'[A-Za-z0-9_-]+', label), 'safe asset label required')
    plan = collect_plan(root, prior, args.scope, args.required_audit)
    metadata = {key: value for key, value in plan.items() if key not in ('scientific', 'local_only_weights')}
    metadata.update(created_utc=datetime.now(timezone.utc).isoformat(),
        prior_manifests_sha256={str(path):sha_file(path) for path in args.prior_manifest},
        collector_sha256=sha_file(__file__), scope=args.scope,
        exclusions=['activation memmaps', 'primary count matrices', 'foundation-model weights', 'credentials/access/tools',
                    'atlas v3/v4 delivery payloads', 'in-progress stages and incomplete audits'])
    summary = {category: {'n_files': len(plan[category]), 'expanded_bytes': sum(x['bytes'] for x in plan[category].values()),
                         'n_parts': len(partition_files(plan[category]))}
               for category in ('scientific', 'local_only_weights')}
    print(json.dumps({'plan': summary, 'pending_groups': plan['pending_groups']}, indent=2), flush=True)
    if args.plan_only:
        return
    require(not plan['pending_groups'], 'Selected collection scope is incomplete; no archives written: '+str(plan['pending_groups']))
    require(not args.out.exists(), f'collection destination must be new: {args.out}')
    args.out.mkdir(parents=True)
    assets = []
    for category in ('scientific', 'local_only_weights'):
        for index, files in enumerate(partition_files(plan[category]), 1):
            prefix = label+'_scientific' if category == 'scientific' else label+'_new_sae_weights_LOCAL_ONLY'
            out = args.out/f'{prefix}_{index:02d}.tar.gz'
            asset = write_archive(root, files, out, category, metadata)
            assets.append(asset)
            print(json.dumps({'verified_asset': asset}), flush=True)
    manifest = {'schema': 'completed-artifact-collection-v1', 'status': 'complete', 'metadata': metadata,
        'assets': assets, 'files': {**plan['scientific'], **plan['local_only_weights']},
        'summary': summary, 'runtime_seconds': time.monotonic()-started}
    (args.out/f'collection_{label}_manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True)+'\n')
    print(json.dumps({'status': 'complete', 'out': str(args.out), 'runtime_seconds': manifest['runtime_seconds']}), flush=True)


if __name__ == '__main__':
    main()
