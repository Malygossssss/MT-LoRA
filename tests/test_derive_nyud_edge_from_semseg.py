import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from scripts.derive_nyud_edge_from_semseg import derive_edges_for_prepared_dataset


class DeriveNyudEdgeFromSemsegTest(unittest.TestCase):
    def test_replace_edge_creates_backup_and_nonzero_edges(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_root = Path(tmp_dir) / "NYUv2_MT"
            segmentation_dir = dataset_root / "segmentation"
            edge_dir = dataset_root / "edge"
            segmentation_dir.mkdir(parents=True)
            edge_dir.mkdir(parents=True)

            seg = np.array(
                [
                    [1, 1, 2, 2],
                    [1, 1, 2, 2],
                    [3, 3, 2, 2],
                    [3, 3, 2, 2],
                ],
                dtype=np.uint8,
            )
            Image.fromarray(seg).save(segmentation_dir / "00001.png")
            np.save(edge_dir / "00001.npy", np.zeros((4, 4), dtype=np.float32))

            results = derive_edges_for_prepared_dataset(
                dataset_root=dataset_root,
                replace_edge=True,
                overwrite=True,
            )

            active_edge_path = dataset_root / "edge" / "00001.npy"
            backup_edge_path = dataset_root / "edge_zero_backup" / "00001.npy"

            self.assertTrue(active_edge_path.is_file())
            self.assertTrue(backup_edge_path.is_file())
            self.assertGreater(np.load(active_edge_path).sum(), 0.0)
            np.testing.assert_array_equal(np.load(backup_edge_path), np.zeros((4, 4), dtype=np.float32))
            self.assertEqual(results["num_files"], 1)


if __name__ == "__main__":
    unittest.main()
