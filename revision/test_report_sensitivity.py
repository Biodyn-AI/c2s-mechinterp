"""Small fixtures for reporting saved results without rescoring them."""
import copy
import importlib.util
from pathlib import Path
import unittest

SPEC = importlib.util.spec_from_file_location("report_sensitivity", Path(__file__).with_name("report_sensitivity.py"))
report = importlib.util.module_from_spec(SPEC)
if SPEC.loader:
    SPEC.loader.exec_module(report)


def fixture():
    catalogue = {"summary": {"n_features": 4, "n_alive": 3, "n_dead": 1}, "features": [
        {"feature_idx": i, "is_dead": i == 3, "activation_freq": value}
        for i, value in enumerate([0.1, 0.2, 0.3, 0.0])]}
    annotation = {"summary": {"n_features_total": 4, "n_alive": 3, "n_sets": 3,
        "n_testable": 3, "n_annotated": 2, "n_ontology_annotated": 1,
        "n_network_or_membership_only": 1, "annotation_rate": 2 / 3,
        "annotation_rate_all_features": 0.5, "ontology_annotation_rate_alive": 1 / 3,
        "ontology_annotation_rate_all_features": 0.25, "scorer_version": "3-plan-aligned",
        "alpha": 0.05, "terms_by_source": {"GO_BP": 4}, "tests_per_feature": 7,
        "universe_size": 12, "legacy_libraries": False,
        "floors": {kind: {"mean": 0.2, "values": [0.1, 0.3],
            "ontology_mean": 0.1, "ontology_values": [0.05, 0.15]}
            for kind in ("freq", "uniform")}}, "features": [
        {"feature_idx": 0, "annotated": True, "n_sig": 1,
         "significant_term_ids": ["GO_BP::term"], "terms": [
             {"source": "GO_BP", "term": "term", "q": 0.01, "p": 0.001, "fold": 2}]},
        {"feature_idx": 1, "annotated": True, "n_sig": 1,
         "significant_term_ids": [], "terms": [{"source": "STRING", "q": 0.01}]},
        {"feature_idx": 2, "annotated": False, "n_sig": 0,
         "significant_term_ids": [], "terms": [{"source": "GO_BP", "term": "no", "q": 0.2}]}]}
    return catalogue, annotation


class SensitivityReportingTests(unittest.TestCase):
    def test_annotation_rates_and_floors_use_matching_denominators(self):
        catalogue, annotation = fixture()
        result = report.annotation_report(catalogue, annotation)
        self.assertEqual(result["n_ontology"], 1)
        self.assertEqual(result["n_network_only"], 1)
        self.assertEqual(result["rates_all_features"]["ontology"], 0.25)
        self.assertEqual(result["rates_catalogue_active"]["ontology"], 1 / 3)
        self.assertAlmostEqual(result["floors"]["freq"]["all_features"]["any_mean"], 0.15)
        self.assertEqual(result["floors"]["freq"]["catalogue_active"]["ontology_values"], [0.05, 0.15])

    def test_recount_rejects_summary_ids_or_incomplete_numeric_details(self):
        for mutation in ("summary", "duplicate", "missing_detail", "dead_row"):
            catalogue, annotation = fixture()
            if mutation == "summary":
                annotation["summary"]["n_ontology_annotated"] = 2
            elif mutation == "duplicate":
                annotation["features"].append(copy.deepcopy(annotation["features"][0]))
            elif mutation == "missing_detail":
                annotation["features"][0]["terms"] = []
            else:
                annotation["features"][2]["feature_idx"] = 3
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                report.annotation_report(catalogue, annotation)

    def test_baseline_uses_fresh_evaluation_and_fixed_weight_training_cosine(self):
        train = {"variance_explained": 0.1, "dead_features": 3, "mean_decoder_cos": 0.07}
        fresh = {"variance_explained_reextracted": 0.9, "dead_val_reextracted": 1,
                 "dead_all_rows": 0, "mean_active_per_position": 32.0}
        result = report.metric_report(train, fresh, baseline=True)
        self.assertEqual(result["variance_explained"], 0.9)
        self.assertEqual(result["heldout_inactive"], 1)
        self.assertEqual(result["mean_decoder_cos"], 0.07)
        self.assertEqual(result["firing"]["mean_active_per_position"], 32.0)
        with self.assertRaises(ValueError):
            report.metric_report(train, None, baseline=True)

    def test_variant_uses_saved_new_training_evaluation(self):
        result = report.metric_report({"variance_explained": 0.8, "dead_features": 2,
            "mean_decoder_cos": 0.04, "dead_all_rows": 1}, None)
        self.assertEqual(result["evaluation_source"], "new_variant_training_metrics")
        self.assertEqual(result["heldout_inactive"], 2)
        self.assertEqual(result["firing"]["dead_all_rows"], 1)


if __name__ == "__main__":
    unittest.main()
