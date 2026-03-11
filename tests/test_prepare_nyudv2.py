import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.prepare_nyudv2 import derive_edge_from_segmentation, fuse_edge_annotations, write_edge_targets


class PrepareNyudv2Test(unittest.TestCase):
    def test_derived_edges_stay_on_class_boundary(self):
        seg = np.array(
            [
                [1, 1, 2, 2],
                [1, 1, 2, 2],
                [1, 1, 2, 2],
                [1, 1, 2, 2],
            ],
            dtype=np.int32,
        )

        edge = derive_edge_from_segmentation(seg)

        self.assertGreater(edge.sum(), 0)
        self.assertEqual(edge[:, :1].sum(), 0)
        self.assertEqual(edge[:, 3:].sum(), 0)

    def test_majority_vote_fuses_multi_annotator_edges(self):
        edge_stack = np.array(
            [
                [[0, 1], [0, 0]],
                [[0, 1], [0, 0]],
                [[0, 0], [0, 0]],
            ],
            dtype=np.float32,
        )

        fused = fuse_edge_annotations(edge_stack)
        expected = np.array([[0, 1], [0, 0]], dtype=np.float32)
        np.testing.assert_array_equal(fused, expected)

    def test_majority_vote_rejects_even_split_ties(self):
        edge_stack = np.array(
            [
                [[0, 1], [0, 0]],
                [[0, 0], [0, 0]],
            ],
            dtype=np.float32,
        )

        fused = fuse_edge_annotations(edge_stack)
        expected = np.zeros((2, 2), dtype=np.float32)
        np.testing.assert_array_equal(fused, expected)

    def test_write_edge_targets_emits_edge_eval_for_multi_gt(self):
        edge_stack = np.array(
            [
                [[0, 1], [0, 0]],
                [[0, 1], [0, 0]],
            ],
            dtype=np.float32,
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            edge_dst = tmp_path / "edge" / "00001.npy"
            edge_eval_dst = tmp_path / "edge_eval" / "00001.npz"
            write_edge_targets(edge_dst, edge_eval_dst, edge_stack)

            self.assertTrue(edge_dst.is_file())
            self.assertTrue(edge_eval_dst.is_file())
            np.testing.assert_array_equal(np.load(edge_dst), np.array([[0, 1], [0, 0]], dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
