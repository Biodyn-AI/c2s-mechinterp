"""Independent v4 transport audit against preserved v3 scientific JSON; standard library only.

Reads one v3 annotation layer and one decoded v4 chunk at a time. Never imports the
exporter or writes experimental inputs. Temporary corruption fixtures are isolated.
"""
import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import resource
import sys
import tempfile
import time


def require(condition,message):
    if not condition:raise ValueError(message)


def sha(path):
    digest=hashlib.sha256()
    with open(path,'rb') as handle:
        for block in iter(lambda:handle.read(4<<20),b''):digest.update(block)
    return digest.hexdigest()


def unique_object(pairs):
    result={}
    for key,value in pairs:
        require(key not in result,f'Duplicate JSON key: {key}')
        result[key]=value
    return result


def parse_json(value):
    return json.loads(value,object_pairs_hook=unique_object,
                      parse_constant=lambda value:(_ for _ in ()).throw(ValueError(f'Nonfinite JSON: {value}')))


def read_json(path):
    with open(path,encoding='utf-8') as handle:
        return json.load(handle,object_pairs_hook=unique_object,
                         parse_constant=lambda value:(_ for _ in ()).throw(ValueError(f'Nonfinite JSON: {value}')))


def read_resource(root,descriptor):
    relative=descriptor['path']
    path=Path(relative)
    require(not path.is_absolute() and '..' not in path.parts and path.as_posix()==relative,f'Unsafe resource path: {relative}')
    encoded=(root/path).read_bytes()
    require(len(encoded)==descriptor['encoded_bytes'],f'Encoded length mismatch: {relative}')
    require(hashlib.sha256(encoded).hexdigest()==descriptor['encoded_sha256'],f'Encoded SHA mismatch: {relative}')
    if descriptor['encoding']=='gzip':
        require(encoded[:2]==b'\x1f\x8b',f'Not gzip: {relative}')
        decoded=gzip.decompress(encoded)
    else:
        require(descriptor['encoding']=='identity',f'Unsupported encoding: {relative}')
        decoded=encoded
    require(len(decoded)==descriptor['decoded_bytes'],f'Decoded length mismatch: {relative}')
    require(hashlib.sha256(decoded).hexdigest()==descriptor['decoded_sha256'],f'Decoded SHA mismatch: {relative}')
    return parse_json(decoded.decode('utf-8',errors='strict'))


def compare_layer(root,transport,layer,original,compact):
    layout=transport['annotations'][str(layer)]
    expected=set(original)
    require(layout['n_features']==len(expected)==len(compact),f'Layer {layer}: feature count mismatch')
    chunk_size=layout['chunk_size']
    require(type(chunk_size) is int and 1<=chunk_size<=512,f'Layer {layer}: invalid chunk size')
    expected_bins={str(int(key)//chunk_size) for key in expected}
    require(set(layout['chunks'])==expected_bins,f'Layer {layer}: missing or extra chunk descriptor')
    seen=set();used=set();counts=Counter()
    for bin_id,chunk in sorted(layout['chunks'].items(),key=lambda item:int(item[0])):
        first,last=int(bin_id)*chunk_size,(int(bin_id)+1)*chunk_size-1
        require(chunk['first_feature_id']==first and chunk['last_feature_id']==last,f'Layer {layer}: bin bounds mismatch')
        logical=chunk['resource'];require(logical in transport['files'],f'Missing chunk resource: {logical}')
        require(transport['files'][logical]['encoding']=='gzip',f'Chunk is not gzip: {logical}')
        records=read_resource(root,transport['files'][logical])
        require(isinstance(records,dict) and len(records)==chunk['n_features'],f'Layer {layer}: decoded chunk count mismatch')
        for key,value in records.items():
            require(str(int(key))==key and first<=int(key)<=last,f'Layer {layer}: noncanonical/out-of-bin ID {key}')
            require(key in expected and key not in seen,f'Layer {layer}: unknown/duplicate ID {key}')
            # Full recursive JSON object/list equality, including every field and ordering of lists.
            require(value==original[key],f'Layer {layer}, feature {key}: full detail object changed')
            require(type(value) is type(original[key]),f'Layer {layer}, feature {key}: record type changed')
            seen.add(key)
            counts['features']+=1;counts['gene_records']+=len(value['genes'])
            counts['ontology_records']+=len(value['anns']);counts['significant_term_ids']+=len(value['significant_term_ids'])
            counts['network_evidence_records']+=len(value['network_evidence'])
        used.add(logical)
    require(seen==expected,f'Layer {layer}: incomplete feature ID coverage')
    compact_ids={str(record['i']) for record in compact}
    require(seen==compact_ids,f'Layer {layer}: compact/detail IDs differ')
    require(seen=={str(index) for index in range(len(compact))},f'Layer {layer}: expected complete 0..N-1 actual atlas IDs')
    counts['dead_features']=sum(bool(record['d']) for record in compact)
    return {'layer':layer,**dict(counts),'n_chunks':len(used),'first_id':0,'last_id':len(compact)-1,
            'last_feature_deep_equal':str(len(compact)-1) in seen,'complete_id_coverage':True},used


def verify_manifest(root,manifest):
    actual={path.relative_to(root).as_posix() for path in root.rglob('*') if path.is_file()}
    require(set(manifest['output_files'])==actual-{'export_manifest.json'},'Manifest file inventory mismatch')
    for name,descriptor in manifest['output_files'].items():
        path=root/name
        require(path.stat().st_size==descriptor['bytes'],f'Manifest byte mismatch: {name}')
        require(sha(path)==descriptor['sha256'],f'Manifest SHA mismatch: {name}')
    return len(manifest['output_files'])


def corruption_tests():
    passed=[]
    with tempfile.TemporaryDirectory(prefix='atlas_v4_audit_fixture_') as temporary:
        root=Path(temporary)
        payload=b'{"0":{"genes":[],"anns":[],"significant_term_ids":[],"network_evidence":[]}}'
        encoded=gzip.compress(payload,mtime=0)
        path=root/'fixture.json.gz';path.write_bytes(encoded)
        descriptor={'path':path.name,'encoding':'gzip','encoded_bytes':len(encoded),'decoded_bytes':len(payload),
                    'encoded_sha256':hashlib.sha256(encoded).hexdigest(),'decoded_sha256':hashlib.sha256(payload).hexdigest()}
        require(read_resource(root,descriptor)['0']['genes']==[],'Fixture roundtrip failed');passed.append('valid gzip roundtrip')
        layout={'annotations':{'0':{'chunk_size':1,'n_features':1,'chunks':{'0':{'resource':'chunk','first_feature_id':0,'last_feature_id':0,'n_features':1}}}},'files':{'chunk':descriptor}}
        row,_=compare_layer(root,layout,0,parse_json(payload),[{'i':0,'d':True}])
        require(row['dead_features']==1 and row['last_feature_deep_equal'],'Dead/last fixture coverage failed');passed.append('dead and last feature detail preserved')
        corrupt=bytearray(encoded);corrupt[-1]^=1;path.write_bytes(corrupt)
        try:read_resource(root,descriptor)
        except ValueError:passed.append('encoded file corruption rejected')
        else:raise AssertionError('Corrupt gzip file accepted')
        path.write_bytes(encoded)
        wrong=dict(descriptor,decoded_sha256='0'*64)
        try:read_resource(root,wrong)
        except ValueError:passed.append('decoded checksum corruption rejected')
        else:raise AssertionError('Wrong decoded checksum accepted')
        path.unlink()
        try:read_resource(root,descriptor)
        except FileNotFoundError:passed.append('missing chunk file rejected')
        else:raise AssertionError('Missing chunk file accepted')
        layout={'annotations':{'0':{'chunk_size':1,'n_features':1,'chunks':{}}},'files':{}}
        try:compare_layer(root,layout,0,{'0':{}},[{'i':0,'d':True}])
        except ValueError:passed.append('missing chunk descriptor rejected')
        else:raise AssertionError('Missing chunk descriptor accepted')
    return passed


def audit(v3,v4,asset_record):
    started=time.time()
    test_results=corruption_tests()
    old_manifest=read_json(v3/'export_manifest.json');manifest=read_json(v4/'export_manifest.json')
    require(old_manifest['schema_version']=='c2s-atlas-v3','Expected preserved v3 source')
    require(manifest['schema_version']=='c2s-atlas-v4' and manifest['status']=='complete','Expected complete v4 export')
    transport=read_json(v4/'data_transport.json')
    require(transport['schema']=='c2s-atlas-transport-v1' and transport['atlas_schema']=='c2s-atlas-v4','Transport schema mismatch')
    files=transport['files'];physical=[descriptor['path'] for descriptor in files.values()]
    require(len(physical)==len(set(physical)),'Aliased physical resource paths')
    actual={path.relative_to(v4).as_posix():path.stat().st_size for path in v4.rglob('*') if path.is_file()}
    require(set(actual)==set(physical)|{'data_transport.json','export_manifest.json'},'Undeclared or missing v4 physical files')
    require(max(actual.values())<100*1024*1024,'100 MiB strict per-file budget exceeded')
    data_bytes=sum(actual.values());require(data_bytes<950_000_000,'950 MB strict published-data budget exceeded')
    total_site_bytes=data_bytes+asset_record['bytes']
    require(total_site_bytes<1_000_000_000,'1 GB strict full candidate-site budget exceeded')
    for root,doc in [(v3,old_manifest),(v4,manifest)]:verify_manifest(root,doc)
    ordinary={name for name in files if not name.startswith('annotations/')}
    expected={path.relative_to(v3).as_posix() for path in v3.rglob('*.json')
              if path.name!='export_manifest.json' and not (path.name.startswith('layer_') and path.name.endswith('_annotations.json'))}
    require(ordinary==expected,'Ordinary resource logical inventory changed')
    summary=None
    for index,name in enumerate(sorted(ordinary),1):
        decoded=read_resource(v4,files[name]);original=read_json(v3/name)
        require(decoded==original,f'Scientific whole-resource JSON changed: {name}')
        if name=='global_summary.json':summary=decoded
        del decoded,original
    require(summary['total_features']==239616 and summary['n_layers']==26,'Unexpected actual atlas size')
    require(set(transport['annotations'])=={str(layer) for layer in manifest['layers']},'Annotation layers do not match manifest')
    rows=[];used=set();totals=Counter()
    for layer in manifest['layers']:
        original=read_json(v3/f'layer_{layer:02d}_annotations.json')
        compact=read_json(v3/f'layer_{layer:02d}_features.json')
        row,resources=compare_layer(v4,transport,layer,original,compact)
        rows.append(row);used.update(resources)
        for key in ['features','gene_records','ontology_records','significant_term_ids','network_evidence_records','dead_features']:totals[key]+=row.get(key,0)
        del original,compact
        print(f"Verified layer {layer:02d}: {row['features']} complete feature objects",flush=True)
    require(used|ordinary==set(files),'Some declared resource was not decoded and verified')
    require(totals['features']==summary['total_features'],'Global feature count mismatch')
    common_hashes=set(old_manifest['inputs_sha256']) & set(manifest['inputs_sha256'])
    require(all(old_manifest['inputs_sha256'][key]==manifest['inputs_sha256'][key] for key in common_hashes),'Existing input provenance hash changed')
    require(set(old_manifest['inputs_sha256'])<=set(manifest['inputs_sha256']),'Input provenance hash removed')
    expected_metadata={'schema_version','output_files','transport','requires_ui_updates','inputs_sha256','positions'}
    differing=[key for key in set(old_manifest)|set(manifest) if old_manifest.get(key)!=manifest.get(key)]
    require(set(differing)<=expected_metadata,'Unexpected non-transport manifest differences')
    position_provenance_changes=[]
    for layer,position in manifest['positions'].items():
        old=old_manifest['positions'][layer]
        source_name=f'layer_{int(layer):02d}_positions.json'
        expected_position={'kind':'provided_dictionary_positions','source':source_name,
                           'sha256':sha(v3/source_name),
                           'requirement':'caller supplies positions from these same saved dictionaries'}
        require(position==expected_position,f'Unexpected copied coordinate provenance at layer {layer}')
        require(old.get('kind') in ('provided_dictionary_positions','UMAP_decoder_directions'),f'Unexpected v3 coordinate origin at layer {layer}')
        if old['kind']=='provided_dictionary_positions':
            require({key:value for key,value in position.items() if key!='sha256'}==old,f'Provided coordinate origin changed at layer {layer}')
        else:
            require(old=={'kind':'UMAP_decoder_directions','seed':42,'metric':'cosine','n_neighbors':15,'min_dist':.1},
                    f'Unexpected original UMAP metadata at layer {layer}')
        position_provenance_changes.append({'layer':int(layer),'v3_origin':old,'v4_copy_provenance':position,
            'changed_fields':{key:{'v3':old.get(key),'v4':position.get(key)} for key in sorted(set(old)|set(position)) if old.get(key)!=position.get(key)},
            'data_json_deep_equal':True})
    require(manifest['requires_ui_updates']==['v4 gzip transport and on-demand feature-ID chunks']+old_manifest['requires_ui_updates'],
            'Unexpected UI-requirement metadata changes')
    peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_bytes=peak if sys.platform=='darwin' else peak*1024
    require(peak_bytes<2_000_000_000,'Audit exceeded intended 2 GB peak RSS')
    return {'audit':'Independent actual v4-v3 full transport/scientific roundtrip','passed':True,
            'self_tests':test_results,'v3':str(v3),'v4':str(v4),
            'v3_manifest_sha256':sha(v3/'export_manifest.json'),'v4_manifest_sha256':sha(v4/'export_manifest.json'),
            'transport_index_sha256':sha(v4/'data_transport.json'),
            'n_decoded_resources_verified':len(files),'n_annotation_chunks':len(used),
            'ordinary_scientific_resources_deep_equal':sorted(ordinary),'totals':dict(totals),'layers':rows,
            'published_data_bytes':data_bytes,'published_data_files':len(actual),
            'largest_data_file_bytes':max(actual.values()),'strict_data_budget_bytes':950_000_000,
            'site_asset_record':asset_record,'candidate_site_bytes_data_plus_assets':total_site_bytes,
            'strict_site_budget_bytes':1_000_000_000,'strict_file_budget_bytes':100*1024*1024,
            'encoded_resource_bytes':sum(row['encoded_bytes'] for row in files.values()),
            'decoded_resource_bytes':sum(row['decoded_bytes'] for row in files.values()),
            'manifest_difference_keys':sorted(differing),
            'newly_hashed_input_files':sorted(set(manifest['inputs_sha256'])-set(old_manifest['inputs_sha256'])),
            'expected_metadata_changes':'v4 transport/schema/chunk inventory, added input checksums, reuse of all preserved v3 coordinate files with added copy checksums, and explicit UI transport requirement; scientific JSON unchanged',
            'position_origin_checkpoint_manifest_sha256':sha(v3/'export_manifest.json'),
            'position_provenance_changes':position_provenance_changes,
            'source_input_hashes_common_unchanged':len(common_hashes),
            'runtime_seconds':round(time.time()-started,3),'peak_rss_bytes':peak_bytes,
            'scope':'Full data-integrity audit only; no browser visual QA or independent revalidation of biological claims'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--v3',type=Path);parser.add_argument('--v4',type=Path)
    parser.add_argument('--site-assets',type=Path);parser.add_argument('--out',type=Path)
    parser.add_argument('--self-test',action='store_true')
    args=parser.parse_args()
    if args.self_test:
        print(json.dumps(corruption_tests(),indent=2));return
    if not all([args.v3,args.v4,args.site_assets,args.out]):parser.error('v3/v4/site-assets/out are required')
    try:
        record=audit(args.v3,args.v4,read_json(args.site_assets))
        record['audit_helper_sha256']=sha(Path(__file__))
    except Exception as error:
        record={'audit':'Independent actual v4-v3 full transport/scientific roundtrip','passed':False,'error':str(error),
                'audit_helper_sha256':sha(Path(__file__))}
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps({key:value for key,value in record.items() if key not in ['layers','ordinary_scientific_resources_deep_equal','site_asset_record']},indent=2))
    if not record['passed']:raise SystemExit(1)


if __name__=='__main__':main()
