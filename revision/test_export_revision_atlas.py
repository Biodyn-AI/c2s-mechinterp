"""Checks for complete annotation export and separation of network evidence."""
import unittest
import json
import hashlib
import gzip
from pathlib import Path
import subprocess
import sys
import tempfile

from export_revision_atlas import (export_feature, export_perturbations, validate_annotation,
                                  write_annotation_chunks, finalize_transport, check_delivery_limits)


class AtlasExportTests(unittest.TestCase):
    def setUp(self):
        self.catalogue = {"feature_idx": 3, "is_dead": False, "activation_freq": 0.1,
                          "mean_activation": 2, "fire_count": 30, "is_svd_aligned": False,
                          "top_genes": [{"gene_name": "MCM2", "mean_activation": 2, "fire_count": 10}]}
        self.annotation = {"feature_idx": 3, "annotated": True, "n_sig": 2,
                           "significant_term_ids": ["GO_BP::DNA replication"],
                           "terms": [{"source": "GO_BP", "term": "DNA replication", "q": 0.01,
                                      "p": 0.0001, "fold": 5, "k": 2, "genes": ["MCM2"]},
                                     {"source": "STRING", "term": "interaction enrichment", "q": 0.02,
                                      "p": 0.001, "k": 3, "expected": 0.2}]}

    def test_network_test_is_not_exported_as_an_ontology_term(self):
        compact, detailed = export_feature(self.catalogue, self.annotation, -1)
        self.assertEqual(compact["na"], 1)
        self.assertEqual(compact["ne"], 1)
        self.assertEqual([term["o"] for term in detailed["anns"]], ["GO_BP"])
        self.assertEqual(detailed["significant_term_ids"], ["GO_BP::DNA replication"])
        self.assertEqual(detailed["network_evidence"][0]["source"], "STRING")
        self.assertEqual(detailed["anns"][0]["effect_kind"], "fold_enrichment")

    def test_missing_details_for_a_significant_id_are_rejected(self):
        self.annotation["significant_term_ids"].append("Reactome::DNA repair")
        with self.assertRaisesRegex(ValueError, "Missing numeric details"):
            validate_annotation(self.annotation)

    def test_nonsignificant_display_fallback_is_not_an_annotation(self):
        self.annotation = {"feature_idx": 3, "annotated": False, "n_sig": 0,
                           "significant_term_ids": [],
                           "terms": [{"source": "GO_BP", "term": "DNA replication", "q": 0.8, "fold": 3}]}
        compact, detailed = export_feature(self.catalogue, self.annotation, -1)
        self.assertEqual(compact["na"], 0)
        self.assertEqual(compact["lb"], "")
        self.assertEqual(detailed["anns"], [])

    def test_staging_export_preserves_counts_without_modifying_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run, annotations, positions, evaluation = [root / name for name in ("run", "annotations", "positions", "evaluation")]
            for path in (run / "sae", annotations, positions, evaluation):
                path.mkdir(parents=True, exist_ok=True)
            dead = dict(self.catalogue, feature_idx=4, is_dead=True, fire_count=0, activation_freq=0)
            catalogue = dict(self.catalogue, top_genes=self.catalogue["top_genes"] + [
                {"gene_name": f"GENE{rank}", "mean_activation": 1, "fire_count": 1} for rank in range(1, 6)])
            payloads = {
                run / "feature_catalog_L00.json": {"features": [catalogue, dead], "summary": {
                    "n_svd_aligned": 0, "n_novel": 1, "mean_feature_cosine": 0.02,
                    "svd_info": {"variance_explained_top50": 0.5}}},
                run / "sae/train_L00.json": {"variance_explained": 0.9, "dead_features": 0},
                evaluation / "2b_eval_L00.json": {"layer": 0, "n_features": 2,
                    "variance_explained_reextracted": 0.91, "dead_val_reextracted": 1},
                run / "cross_layer_tracking.json": {}, run / "circuit_graph.json": {"n_edges": 7, "edges": [
                    {"source_layer": 0, "target_layer": 0, "source_feature": 3,
                     "target_feature": target, "cohens_d": target + 1} for target in range(7)]},
                annotations / "ann_v2_L00.json": {"summary": {"scorer_version": "3-plan-aligned", "n_annotated": 1,
                    "floors": {"freq": {"mean": 0.4, "sd": 0.1, "values": [0.3, 0.5],
                        "ontology_mean": 0.2, "ontology_values": [0.1, 0.3]}}}, "features": [self.annotation]},
                positions / "layer_00_positions.json": [[0.0, 0.0], [1.0, 1.0]],
            }
            for path, payload in payloads.items():
                path.write_text(json.dumps(payload))
            before = {path: path.read_bytes() for path in payloads}
            completed = subprocess.run([sys.executable, str(Path(__file__).with_name("export_revision_atlas.py")),
                "--run", str(run), "--annotations", str(annotations), "--positions-dir", str(positions),
                "--evaluation-dir", str(evaluation),
                "--out", str(root / "staging"), "--layers", "0", "--gene-index-cap", "1"], capture_output=True, text=True)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            summary = json.loads((root / "staging/global_summary.json").read_text())
            self.assertEqual(summary["total_annotated"], 1)
            self.assertEqual(summary["total_annotated_any"], 1)
            self.assertEqual(summary["total_network_evidence"], 1)
            layer = summary["layers"][0]
            # Publication metrics must use the fresh extraction, never the stale train values.
            self.assertEqual(layer["variance_explained"], 0.91)
            self.assertEqual(layer["heldout_dead_features"], 1)
            self.assertEqual(layer["dead"], 1)
            self.assertEqual(layer["annotation_rate"], 0.5)
            self.assertEqual(layer["annotation_rate_alive"], 1)
            self.assertEqual(layer["annotation_rate_any"], 0.5)
            self.assertEqual(layer["annotation_rate_any_alive"], 1)
            self.assertEqual(layer["annotation_floors_any"]["freq"]["mean"], 0.2)
            self.assertEqual(layer["annotation_floors_ontology"]["freq"]["mean"], 0.1)
            self.assertEqual(layer["annotation_floors_ontology"]["freq"]["values"], [0.05, 0.15])
            self.assertEqual(summary["gene_index_top_genes_per_feature"], 5)
            self.assertEqual(summary["gene_index_per_gene_cap"], 1)
            gene_index = json.loads((root / "staging/gene_index.json").read_text())
            gene_counts = json.loads((root / "staging/gene_index_counts.json").read_text())
            self.assertEqual(len(gene_index["MCM2"]), 1)
            self.assertEqual(gene_counts["MCM2"], {"total": 2, "by_layer": {"0": 2}})
            self.assertNotIn("GENE5", gene_index)
            transport = json.loads((root / "staging/data_transport.json").read_text())
            chunk = transport["annotations"]["0"]["chunks"]["0"]
            resource = transport["files"][chunk["resource"]]
            self.assertEqual(len(json.loads(gzip.decompress((root / "staging" / resource["path"]).read_bytes()))["3"]["genes"]), 6)
            self.assertFalse((root / "staging/layer_00_annotations.json").exists())
            svd = json.loads((root / "staging/svd_comparison.json").read_text())
            self.assertEqual(svd["per_layer"]["0"]["sae_variance"], 0.91)
            manifest = json.loads((root / "staging/export_manifest.json").read_text())
            self.assertEqual(manifest["status"], "complete")
            self.assertEqual(manifest["schema_version"], "c2s-atlas-v4")
            self.assertIn(resource["path"], manifest["output_files"])
            self.assertEqual(manifest["inputs_sha256"]["2b_eval_L00.json"],
                             hashlib.sha256((evaluation / "2b_eval_L00.json").read_bytes()).hexdigest())
            graph = json.loads((root / "staging/cross_layer_graph.json").read_text())["L00_L00"]
            self.assertEqual(graph["summary"]["n_edges_stored"], 7)
            self.assertEqual(graph["summary"]["n_display_edges"], 5)
            self.assertEqual(graph["summary"]["display_cap_per_source_target_layer"], 5)
            self.assertEqual(len(graph["deps"][0]["deps"]), 5)
            self.assertEqual(graph["deps"][0]["deps"][0]["b"], 6)
            self.assertTrue(all(path.read_bytes() == content for path, content in before.items()))

    def test_fresh_evaluation_is_required_before_creating_staging_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            completed = subprocess.run([sys.executable, str(Path(__file__).with_name("export_revision_atlas.py")),
                "--run", str(root / "run"), "--annotations", str(root / "annotations"),
                "--evaluation-dir", str(root / "evaluation"), "--out", str(root / "staging"),
                "--layers", "0"], capture_output=True, text=True)
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("Fresh evaluation required", completed.stderr)
            self.assertFalse((root / "staging").exists())

    def test_perturbation_layer_and_current_labels_are_explicit_metadata(self):
        import numpy as np
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            group = {"representation": "model", "layer": 13, "kind": "tf", "group": "TF1",
                     "n_eligible_known_target_genes": 2, "n_responding": 2, "n_cells": 15,
                     "specificity": {"n_target_responses": 1, "q": 0.02}}
            (root / "regulatory_statistics.json").write_text(json.dumps({"groups": [group]}))
            np.savez(root / "feature_statistics_L13.npz", groups=np.array(["TF1"]),
                     responding=np.array([[True, False, True]]), effect=np.array([[0.8, 0.0, -1.2]]))
            export_perturbations(root, root, {(13, 2): "Current named term"}, layer=13)
            response = json.loads((root / "perturbation_response.json").read_text())
            self.assertEqual(response["summary"]["layer"], 13)
            self.assertEqual(response["summary"]["n_trrust_tfs"], 1)
            self.assertEqual(response["targets"][0]["top"][0]["i"], 2)
            self.assertEqual(response["targets"][0]["top"][0]["lb"], "Current named term")

    def test_all_chunk_features_round_trip_with_complete_details_and_byte_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            expected = {}
            # Deliberately unsorted sparse IDs include both sides of chunk boundaries.
            for feature in (512, 0, 511, 255, 256, 1023):
                catalogue = dict(self.catalogue, feature_idx=feature, is_dead=feature == 1023,
                    top_genes=[{"gene_name": f"GENE{index}", "mean_activation": index/7, "fire_count": index} for index in range(20)])
                annotation = dict(self.annotation, feature_idx=feature)
                if feature == 1023:
                    annotation = {"feature_idx": feature, "annotated": False, "n_sig": 0, "significant_term_ids": []}
                expected[str(feature)] = export_feature(catalogue, annotation, -1)[1]
            layer, files = write_annotation_chunks(root, 22, expected, chunk_size=256)
            self.assertEqual(layer["n_features"], 6)
            self.assertEqual(set(layer["chunks"]), {"0", "1", "2", "3"})
            reconstructed = {}
            for chunk in layer["chunks"].values():
                resource = files[chunk["resource"]]
                encoded = (root/resource["path"]).read_bytes()
                decoded = gzip.decompress(encoded)
                self.assertEqual(hashlib.sha256(encoded).hexdigest(), resource["encoded_sha256"])
                self.assertEqual(hashlib.sha256(decoded).hexdigest(), resource["decoded_sha256"])
                self.assertEqual(len(encoded), resource["encoded_bytes"])
                self.assertEqual(len(decoded), resource["decoded_bytes"])
                records = json.loads(decoded)
                self.assertEqual(len(records), chunk["n_features"])
                self.assertFalse(set(reconstructed) & set(records))
                reconstructed.update(records)
            self.assertEqual(reconstructed, expected)
            self.assertEqual(len(reconstructed["1023"]["genes"]), 20)
            self.assertEqual(reconstructed["1023"]["anns"], [])
            self.assertEqual(reconstructed["0"]["significant_term_ids"], ["GO_BP::DNA replication"])
            self.assertEqual(reconstructed["0"]["network_evidence"], self.annotation["terms"][1:])
            self.assertFalse(list(root.glob("layer_*_annotations.json")))

    def test_other_large_json_is_gzipped_and_delivery_limits_include_nested_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = {"large": ["complete λ record"]*100}
            (root/"large.json").write_text(json.dumps(payload))
            transport = finalize_transport(root, {}, {}, gzip_threshold=100)
            resource = transport["files"]["large.json"]
            self.assertEqual(resource["encoding"], "gzip")
            self.assertEqual(json.loads(gzip.decompress((root/resource["path"]).read_bytes())), payload)
            self.assertFalse((root/"large.json").exists())
            nested = root/"nested/large.bin"
            nested.parent.mkdir()
            nested.write_bytes(b"x"*20000)
            with self.assertRaisesRegex(ValueError, "nested/large.bin"):
                check_delivery_limits(root, max_site_bytes=100000, max_file_bytes=20000)
            nested.unlink()
            with self.assertRaisesRegex(ValueError, "site-size"):
                check_delivery_limits(root, max_site_bytes=1, max_file_bytes=10000)
            with self.assertRaisesRegex(ValueError, "file-size.*export_manifest"):
                check_delivery_limits(root, max_site_bytes=100000, max_file_bytes=20000,
                                      additional_files=[("export_manifest.json", 20000)])


if __name__ == "__main__":
    unittest.main()
