"""Small fixtures for aggregation of saved text-deletion retrieval records."""
import copy
import importlib.util
from pathlib import Path
import unittest

SPEC = importlib.util.spec_from_file_location("report_text_deletion", Path(__file__).with_name("report_text_deletion.py"))
reporter = importlib.util.module_from_spec(SPEC)
if SPEC.loader:
    SPEC.loader.exec_module(reporter)


def fixture():
    rows = []
    for tf in ("A", "B"):
        for layer in (8, 16):
            evaluated = tf == "A"
            rows.append({"tf": tf, "layer": layer, "n": 10, "n_positive": 2 if evaluated else 0,
                "average_precision": 0.2 if evaluated else None, "auroc": 0.7 if evaluated else None,
                "prevalence": 0.2 if evaluated else 0, "random_label_expected_ap": 0.3 if evaluated else 0,
                "ap_lift": 1.0 if evaluated else None, "prevalence_lift": 1.0 if evaluated else None,
                "lift_over_random_expected_ap": 2/3 if evaluated else None})
    return {"meta": {"tfs": ["A", "B"], "genes": ["A", "B", "C"], "layers": [8, 16],
        "n_pairs": 30, "max_pre_deletion_cosdist_mean": 1e-5, "max_pre_deletion_cosdist_max": 2e-5},
        "retrieval": rows}


class TextDeletionReportingTests(unittest.TestCase):
    def test_macro_uses_only_evaluable_tfs_and_retains_full_screened_family(self):
        record = reporter.summarize_line(fixture(), bootstrap=25, seed=42)
        self.assertEqual(record["n_screened_named_tfs"], 2)
        self.assertEqual(record["n_pairs"], 30)
        for layer in record["layers"]:
            self.assertEqual(layer["n_tfs_evaluable"], 1)
            self.assertEqual(layer["evaluable_tfs"], ["A"])
            self.assertEqual(layer["mean_average_precision"], 0.2)
            self.assertEqual(layer["mean_average_precision_ci95"], [0.2, 0.2])
            self.assertEqual(layer["mean_random_label_expected_ap"], 0.3)
            self.assertEqual(layer["mean_random_label_expected_ap_ci95"], [0.3, 0.3])
            self.assertEqual(layer["bootstrap_unit"], "transcription factors")

    def test_rejects_incomplete_or_duplicate_tf_layer_grid(self):
        for kind in ("missing", "duplicate", "foreign_tf"):
            payload = fixture()
            if kind == "missing":
                payload["retrieval"].pop()
            elif kind == "duplicate":
                payload["retrieval"].append(copy.deepcopy(payload["retrieval"][0]))
            else:
                payload["retrieval"][0]["tf"] = "foreign"
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                reporter.summarize_line(payload, bootstrap=25, seed=42)

    def test_copies_exact_noise_fields_and_rejects_missing_finite_reference(self):
        payload = fixture()
        record = reporter.summarize_line(payload, bootstrap=25, seed=42)
        self.assertEqual(record["max_pre_deletion_cosdist_mean"], 1e-5)
        self.assertEqual(record["max_pre_deletion_cosdist_max"], 2e-5)
        del payload["retrieval"][0]["random_label_expected_ap"]
        with self.assertRaises(KeyError):
            reporter.summarize_line(payload, bootstrap=25, seed=42)


if __name__ == "__main__":
    unittest.main()
