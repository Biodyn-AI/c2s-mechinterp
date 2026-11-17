import unittest

import numpy as np

from audit_sensitivity import stream_moments, selected_gene_similarity


class SensitivityAuditTests(unittest.TestCase):
    def test_streamed_input_moments_are_centered_and_chunk_invariant(self):
        matrix = np.array([[10., 0], [12., 2], [14., 4], [16., 6]])
        for chunk in [1, 3]:
            mean, trace = stream_moments(matrix, chunk)
            np.testing.assert_allclose(mean, matrix.mean(axis=0))
            self.assertAlmostEqual(trace, np.var(matrix, axis=0).sum())

    def test_gene_geometry_averages_occurrences_before_centering(self):
        matrix = np.array([[1., 0], [3., 0], [0., 4], [4., 4], [2., 2]])
        gene_rows = [np.array([0, 1]), np.array([2]), np.array([3, 4])]
        means = np.array([[2., 0], [0., 4], [3., 3]])
        centered = means-means.mean(axis=0)
        unit = centered/np.linalg.norm(centered, axis=1, keepdims=True)
        np.testing.assert_allclose(selected_gene_similarity(matrix, gene_rows), unit@unit.T)


if __name__ == '__main__': unittest.main()
