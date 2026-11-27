"""Synthetic contract tests for the external preprocessing adapter."""
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import anndata as ad
import numpy as np
from scipy import sparse
import zarr

from sigmma.cli.preprocess import build_manifest, main, output_ids, safe_id


class PreprocessTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.staging = self.root / "raw"
        self.samples = self.root / "samples.txt"
        self.samples.write_text("project/section\n")
        self.sample("project/section")
        self.out = self.root / "results"
        self.manifest = self.out / "manifest.csv"

    def sample(self, ident, rois=None):
        folder = self.staging / ident
        bundle = folder / "bundle"
        bundle.mkdir(parents=True)
        for filename in ("cell_feature_matrix.h5", "cells.csv.gz", "experiment.xenium",
                         "morphology_focus.ome.tif"):
            (bundle / filename).touch()
        (folder / "slide.svs").touch()
        roi = {"xenium": {"file": "bundle"}, "hae": {"file": "slide.svs"}}
        if rois is not None:
            roi["xenium"]["spatial_um"] = rois
        (folder / "roi.json").write_text(json.dumps(roi))

    def assets(self):
        image_path = self.out / "aligned_images_zarr/project/section.zarr"
        image_path.parent.mkdir(parents=True)
        image = zarr.open(str(image_path), mode="w", shape=(3, 512, 512), dtype="uint8")
        image.attrs["parameters"] = {"resolution_um_per_px": 0.5}
        index_path = self.out / "index_array/project/section/tile_256_stride_256.zarr"
        index_path.parent.mkdir(parents=True)
        index = zarr.open(str(index_path), mode="w", shape=(2, 3), dtype="int64")
        index[:] = [[0, 0, 65536], [256, 256, 30000]]
        spatial_path = self.out / "terra_adata/project/section.h5ad"
        spatial_path.parent.mkdir(parents=True)
        data = ad.AnnData(np.ones((2, 3), dtype=np.float32))
        data.obsm["spatial"] = np.array([[10., 20.], [150., 140.]])
        data.obsm["cell_emb"] = np.ones((2, 4), dtype=np.float32)
        data.obsp["spatial_connectivities"] = sparse.csr_matrix([[0., 1.], [1., 0.]])
        data.write_h5ad(spatial_path)
        return image, index, spatial_path

    def test_manifest_matches_loader_columns_and_absolute_paths(self):
        self.assets()
        build_manifest(output_ids(self.samples, self.staging), self.out, self.manifest,
                       embedding_dim=4)
        with self.manifest.open() as handle:
            row, = list(csv.DictReader(handle))
        self.assertEqual(row["sample_name"], "project/section")
        self.assertEqual(row["annotation"], "")
        for key in ("hematoxylin_eosin_image", "spatial_preprocessed", "index"):
            self.assertTrue(Path(row[key]).is_absolute())
            self.assertTrue(Path(row[key]).exists())
        with self.assertRaises(FileExistsError):
            build_manifest(["project/section"], self.out, self.manifest)

    def test_invalid_outputs_do_not_publish_manifest(self):
        image, index, path = self.assets()
        for case in ("resolution", "bounds", "missing_embedding", "nan_embedding", "width", "coords"):
            with self.subTest(case=case):
                image.attrs["parameters"] = {"resolution_um_per_px": 0.5}
                index[:] = [[0, 0, 65536], [256, 256, 30000]]
                data = ad.read_h5ad(path)
                data.obsm["cell_emb"] = np.ones((2, 4), dtype=np.float32)
                data.obsm["spatial"] = np.array([[10., 20.], [150., 140.]])
                if case == "resolution":
                    image.attrs["parameters"] = {"resolution_um_per_px": 1.0}
                elif case == "bounds":
                    index[0] = [300, 0, 65536]
                elif case == "missing_embedding":
                    del data.obsm["cell_emb"]
                elif case == "nan_embedding":
                    data.obsm["cell_emb"][0, 0] = np.nan
                elif case == "coords":
                    data.obsm["spatial"][0, 0] = 1000
                data.write_h5ad(path)
                with self.assertRaises(ValueError):
                    build_manifest(["project/section"], self.out, self.manifest,
                                   embedding_dim=5 if case == "width" else 4)
                self.assertFalse(self.manifest.exists())

    def test_roi_names_follow_upstream_and_collisions_fail(self):
        self.sample("project/second", {"core_1": "POLYGON EMPTY"})
        self.samples.write_text("project/second\n")
        self.assertEqual(output_ids(self.samples, self.staging), ["project/core_1"])
        self.sample("project/third", {"core_1": "POLYGON EMPTY"})
        self.samples.write_text("project/second\nproject/third\n")
        with self.assertRaisesRegex(ValueError, "collision"):
            output_ids(self.samples, self.staging)

    def test_unsafe_ids_and_missing_inputs(self):
        for value in ("../a", "/a", "a//b", "a/../b", "a;touch x", "a/$(id)"):
            with self.assertRaises(ValueError):
                safe_id(value)
        (self.staging / "project/section/slide.svs").unlink()
        with self.assertRaisesRegex(ValueError, "Missing H&E"):
            output_ids(self.samples, self.staging)

    def test_dry_run_never_executes_and_run_failure_never_builds(self):
        repo = self.root / "pipeline"
        repo.mkdir()
        (repo / "main.nf").touch()
        argv = ["run", "--samples", str(self.samples), "--staging-path", str(self.staging),
                "--outdir", str(self.out), "--manifest", str(self.manifest),
                "--preprocessing-repo", str(repo)]
        with patch("sigmma.cli.preprocess.subprocess.run") as run:
            main(argv + ["--dry-run"])
            run.assert_not_called()
        with patch("sigmma.cli.preprocess.subprocess.run", side_effect=RuntimeError("failed")):
            with self.assertRaises(RuntimeError):
                main(argv)
        self.assertFalse(self.manifest.exists())


if __name__ == "__main__":
    unittest.main()
