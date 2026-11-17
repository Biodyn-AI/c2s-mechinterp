"""Synthetic checks for guide alignment, fixed controls and descriptive concordance."""
import unittest
import json
from pathlib import Path
import tempfile
from unittest.mock import patch
import numpy as np
from rev_guide_concordance import align_selected_rows, compare_guide_effects, guide_profiles


class GuideConcordanceTests(unittest.TestCase):
    def test_row_alignment_retains_shuffled_extraction_order_and_fixed_controls(self):
        design={'groups':[{'group':'control','kind':'control','cells':[0,1,2]},
                          {'group':'TF','kind':'tf','cells':[3,4,5]}]}
        rows=[{'cell_index':cell,'group':('TF' if cell>=3 else 'control'),
               'kind':('tf' if cell>=3 else 'control')} for cell in [5,0,4,1,3,2]]
        indices=align_selected_rows(design,rows,np.array(['control']*3+['TF']*3),np.array(['one']*6))
        np.testing.assert_array_equal(indices['control'],[1,3,5])
        np.testing.assert_array_equal(indices['TF'],[0,2,4])
        with self.assertRaises(ValueError):align_selected_rows(design,rows[:-1],np.array(['control']*3+['TF']*3),np.array(['one']*6))
        with self.assertRaises(ValueError):align_selected_rows(design,rows,np.array(['control']*3+['OTHER']*3),np.array(['one']*6))

    def test_pair_metrics_are_descriptive_and_empty_response_union_is_not_perfect(self):
        first={'effect':np.array([2.,-2.,0.,1.]),'responding':np.array([1,1,0,0],bool)}
        same=compare_guide_effects(first,first)
        self.assertAlmostEqual(same['effect_pearson'],1.)
        self.assertEqual(same['joint_response_sign_agreement'],1.)
        opposite=compare_guide_effects(first,{'effect':-first['effect'],'responding':first['responding']})
        self.assertAlmostEqual(opposite['effect_pearson'],-1.)
        self.assertEqual(opposite['joint_response_sign_agreement'],0.)
        empty={'effect':np.zeros(4),'responding':np.zeros(4,bool)}
        record=compare_guide_effects(empty,empty)
        self.assertIsNone(record['response_jaccard'])
        self.assertIsNone(record['effect_pearson'])
        self.assertIsNone(record['joint_response_sign_agreement'])

    def test_profiles_use_all_selected_guides_and_the_same_control_rows(self):
        rng=np.random.default_rng(5)
        controls=rng.normal(size=(20,4))
        first=rng.normal(size=(8,4)); first[:,0]+=3
        second=rng.normal(size=(8,4)); second[:,0]-=3
        features=np.vstack([controls,first,second,[[1,2,3,4]]])
        row_guides=np.array(['NT']*20+['TFg1']*8+['TFg2']*8+['TFg3'])
        catalogue=[{'feature_idx':index,'top_genes':[gene]} for index,gene in enumerate(['A','B','C','TF'])]
        summary,arrays=guide_profiles(features,np.arange(20,37),np.arange(20),row_guides,
                                       'TF',{'TF','A'},catalogue,['A','B','C','TF'],min_cells=3)
        self.assertEqual(summary['n_guides_selected'],3)
        self.assertEqual(summary['n_guides_evaluable'],2)
        self.assertEqual([record['n_control_cells'] for record in summary['guides']],[20,20,20])
        self.assertEqual(summary['guides'][-1]['status'],'insufficient cells')
        self.assertEqual(len(summary['pairs']),1)
        self.assertEqual(summary['guides'][0]['gene_retrieval']['n'],3)
        self.assertEqual(summary['guides'][0]['gene_retrieval']['n_positive'],1)
        self.assertEqual(summary['guides'][0]['gene_retrieval']['bootstrap'],0)
        self.assertIsNone(summary['guides'][0]['gene_retrieval']['average_precision_ci95'])
        self.assertEqual(len(arrays),2)
        self.assertNotIn('pvalue',summary['pairs'][0])


    def test_complete_command_preserves_guide_counts_and_shuffled_cell_alignment(self):
        import anndata
        from scipy import sparse
        from rev_guide_concordance import main
        rng=np.random.default_rng(7)
        with tempfile.TemporaryDirectory() as temporary:
            base=Path(temporary)
            features_dir,catalog_dir=base/'features',base/'catalog'
            features_dir.mkdir();catalog_dir.mkdir()
            adata=anndata.AnnData(X=sparse.csr_matrix(np.ones((14,4))))
            adata.var_names=['A','B','C','TF']
            adata.obs['guide_id']=['NTg1']*6+['TFg1']*4+['TFg2']*4
            adata.obs['c6_gene']=['control']*6+['TF']*8
            adata.obs['c6_recorded_batch']=['rep1-tx']*14
            adata.write_h5ad(base/'counts.h5ad')
            design={'groups':[{'group':'control','kind':'control','cells':list(range(6)),
                               'guide_counts_selected':{'NTg1':6}},
                              {'group':'TF','kind':'tf','cells':list(range(6,14)),
                               'guide_counts_selected':{'TFg1':4,'TFg2':4}}]}
            (base/'design.json').write_text(json.dumps(design))
            rows=[{'cell_index':cell,'group':('control' if cell<6 else 'TF'),
                   'kind':('control' if cell<6 else 'tf')} for cell in reversed(range(14))]
            (features_dir/'rows.json').write_text(json.dumps(rows))
            np.savez(features_dir/'feat_L08.npz',X=rng.normal(size=(14,4)))
            (catalog_dir/'feature_catalog_L08.json').write_text(json.dumps({'features':[
                {'feature_idx':index,'top_genes':[gene]} for index,gene in enumerate(adata.var_names)]}))
            (base/'trrust.tsv').write_text('TF\tA\tActivation\t1\nTF\tTF\tActivation\t2\n')
            command=['rev_guide_concordance.py','--h5ad',str(base/'counts.h5ad'),'--design',str(base/'design.json'),
                     '--features-dir',str(features_dir),'--catalog-dir',str(catalog_dir),'--trrust',str(base/'trrust.tsv'),
                     '--layers','8','--out',str(base/'output/guide_concordance.json')]
            with patch('sys.argv',command):main()
            output=json.loads((base/'output/guide_concordance.json').read_text())
            self.assertEqual(output['n_fixed_control_cells'],6)
            self.assertEqual(output['control_h5ad_cell_indices'],[5,4,3,2,1,0])
            self.assertEqual(output['records'][0]['n_guides_evaluable'],2)
            with np.load(base/'output/guide_statistics_L08.npz') as arrays:
                self.assertEqual(arrays['effect'].shape,(2,4))
                self.assertEqual(arrays['guides'].tolist(),['TFg1','TFg2'])


if __name__=='__main__':unittest.main()
