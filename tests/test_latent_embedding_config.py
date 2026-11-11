"""Regression coverage for configurable latent features, including graph caches."""
import importlib.util
from pathlib import Path

import anndata as ad
import numpy as np
import tempfile
import unittest
from unittest.mock import patch
import scipy.sparse as sp
import torch
import zarr

from sigmma.datasets.dataset import SpatialPatchDataset, MultiSectionDataset

FOV = 8
NUM_RES = 1                     # 2x2 sub-FOVs
GRID = FOV // 2 ** NUM_RES      # 4 px
UM_PER_PX = 0.5

# Cell positions in pixels. AnnData row order deliberately differs from
# sub-FOV order (x outer, y inner): sub-FOV order is [0, 2, 3, 5, 1, 4].
COORDS_PX = np.array([
    [1.0, 1.0],   # sub-FOV (0,0)
    [5.0, 1.0],   # sub-FOV (1,0)
    [1.0, 2.0],   # sub-FOV (0,0)
    [1.0, 5.0],   # sub-FOV (0,1)
    [6.0, 6.0],   # sub-FOV (1,1)
    [2.0, 6.0],   # sub-FOV (0,1)
], dtype=np.float32)


def _make_section(root: Path) -> dict:
    section_dir = root / "tissue" / "section0"
    section_dir.mkdir(parents=True)
    n = COORDS_PX.shape[0]
    conn = sp.csr_matrix(np.ones((n, n), dtype=np.float32) - np.eye(n, dtype=np.float32))
    adata = ad.AnnData(
        X=sp.csr_matrix(np.random.default_rng(0).poisson(1.0, (n, 5)).astype(np.float32)),
        obsm={
            "spatial": COORDS_PX * UM_PER_PX,          # stored in um
            "nemo_latent": np.arange(n * 4, dtype=np.float32).reshape(n, 4),
        },
        obsp={"spatial_connectivities": conn},
    )
    adata.var_names = [f"g{i}" for i in range(5)]
    spatial_file = section_dir / "spatial.h5ad"
    adata.write_h5ad(spatial_file)

    image_file = section_dir / "image.zarr"
    img = zarr.open(str(image_file), mode="w", shape=(3, FOV, FOV), dtype="uint8")
    img[:] = 0
    indices_file = section_dir / "index.zarr"
    idx = zarr.open(str(indices_file), mode="w", shape=(1, 3), dtype="int64")
    idx[:] = np.array([[0, 0, 1]])                     # (y_start, x_start, score)
    return dict(spatial_file=spatial_file, image_file=image_file, indices_file=indices_file)


def _build_cache(files: dict, cache_dir: Path) -> None:
    script = Path(__file__).resolve().parents[1] / "scripts" / "precompute_hierarchical_subgraphs.py"
    spec = importlib.util.spec_from_file_location("precompute_hier", script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    cache_dir.mkdir()
    mod.precompute_slide(str(files["spatial_file"]), str(files["indices_file"]),
                         FOV, NUM_RES, None, str(cache_dir / "section0.npz"),
                         image_resolution_um_per_pixel=UM_PER_PX)


class LatentEmbeddingTests(unittest.TestCase):
    def test_configured_features_and_legacy(self):
        for key in (None, "cell_emb", "spatial_cell_emb", "neighborhood_emb"):
            for use_cache in (False, True):
                with self.subTest(key=key, cache=use_cache), tempfile.TemporaryDirectory() as tmp:
                    root = Path(tmp)
                    files = _make_section(root)
                    expected = np.arange(24, dtype=np.float32).reshape(6, 4)
                    if key is not None:
                        data = ad.read_h5ad(files["spatial_file"])
                        del data.obsm["nemo_latent"]
                        expected += 100
                        data.obsm[key] = expected
                        data.write_h5ad(files["spatial_file"])
                    cache = root / "cache" if use_cache else None
                    if use_cache:
                        _build_cache(files, cache)
                    ds = SpatialPatchDataset(
                        **files, fov_size=FOV, num_resolutions=NUM_RES,
                        tissue_filtering_threshold=None, hier_cache_dir=cache,
                        latent_embedding_name=key,
                    )
                    # Observe subgraph features too: the final graph retains full-FOV features.
                    observed = []
                    original = ds._get_cell_embeddings
                    def select(mask):
                        result = original(mask)
                        observed.append(result)
                        return result
                    with patch.object(ds, "_get_cell_embeddings", side_effect=select):
                        item = ds[0]
                    np.testing.assert_array_equal(item["spatial_graph"].ndata["feat"].numpy(), expected)
                    np.testing.assert_array_equal(item["spatial"].numpy(), expected.mean(axis=0))
                    self.assertGreater(len(observed), 1)
                    self.assertTrue(all(np.isin(v, expected).all() for v in observed))
                    wrapper = MultiSectionDataset.__new__(MultiSectionDataset)
                    wrapper.first_dataset = ds
                    self.assertEqual(wrapper.get_num_genes(), 4)

    def test_config_reaches_each_section(self):
        import csv
        from ml_collections import ConfigDict
        from sigmma.utils.train_utils import load_dataset
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _make_section(root)
            data = ad.read_h5ad(files["spatial_file"])
            data.obsm["custom_latent"] = data.obsm.pop("nemo_latent")
            data.write_h5ad(files["spatial_file"])
            manifest = root / "manifest.csv"
            row = dict(sample_name="section0", hematoxylin_eosin_image=files["image_file"],
                       index=files["indices_file"], spatial_preprocessed=files["spatial_file"],
                       annotation="")
            with manifest.open("w") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(row))
                writer.writeheader()
                writer.writerow(row)
            dataset = load_dataset(ConfigDict(dict(
                manifest_path=str(manifest), pretrained_cell_embedding=True,
                latent_embedding_name="custom_latent", fov_size=FOV,
                num_resolutions=NUM_RES, tissue_filtering_threshold=None,
            )))
            self.assertEqual(dataset.first_dataset.latent_embedding_name, "custom_latent")
            self.assertEqual(dataset.get_num_genes(), 4)
            self.assertEqual(tuple(dataset[0]["spatial"].shape), (4,))

    def test_terra_default_prefers_new_key(self):
        with tempfile.TemporaryDirectory() as tmp:
            files = _make_section(Path(tmp))
            data = ad.read_h5ad(files["spatial_file"])
            expected = data.obsm["nemo_latent"] + 100
            data.obsm["terra_latent"] = expected
            data.write_h5ad(files["spatial_file"])
            ds = SpatialPatchDataset(**files)
            self.assertEqual(ds.latent_embedding_name, "terra_latent")
            np.testing.assert_array_equal(ds._latent, expected)
            # An explicit legacy key takes precedence even when TERRA is present.
            legacy = SpatialPatchDataset(**files, latent_embedding_name="nemo_latent")
            np.testing.assert_array_equal(legacy._latent, data.obsm["nemo_latent"])

    def test_terra_default_warns_on_legacy_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            files = _make_section(Path(tmp))
            with self.assertWarnsRegex(UserWarning, "using legacy nemo_latent"):
                ds = SpatialPatchDataset(**files)
            self.assertEqual(ds.latent_embedding_name, "nemo_latent")
            np.testing.assert_array_equal(ds._latent, ad.read_h5ad(files["spatial_file"]).obsm["nemo_latent"])

    def test_explicit_terra_does_not_fall_back(self):
        with tempfile.TemporaryDirectory() as tmp:
            files = _make_section(Path(tmp))
            with self.assertRaisesRegex(ValueError, "terra_latent.*available obsm keys"):
                SpatialPatchDataset(**files, latent_embedding_name="terra_latent")
            legacy = SpatialPatchDataset(**files, type_pretrained_cell_embedding="nemo")
            self.assertEqual(legacy.latent_embedding_name, "nemo_latent")

    def test_training_loader_defaults_to_terra(self):
        from ml_collections import ConfigDict
        from sigmma.utils.train_utils import load_dataset
        cfg = ConfigDict(dict(manifest_path="unused.csv", pretrained_cell_embedding=True))
        with patch("sigmma.utils.train_utils.MultiSectionDataset") as constructor:
            load_dataset(cfg)
        self.assertEqual(constructor.call_args.kwargs["type_pretrained_cell_embedding"], "terra")

    def test_missing_key_is_actionable(self):
        with tempfile.TemporaryDirectory() as tmp:
            files = _make_section(Path(tmp))
            with self.assertRaisesRegex(ValueError, "missing.*available obsm keys"):
                SpatialPatchDataset(**files, latent_embedding_name="missing")

    def test_raw_expression_needs_no_latent(self):
        with tempfile.TemporaryDirectory() as tmp:
            files = _make_section(Path(tmp))
            data = ad.read_h5ad(files["spatial_file"])
            del data.obsm["nemo_latent"]
            expected = data.X.toarray()
            data.write_h5ad(files["spatial_file"])
            ds = SpatialPatchDataset(**files, pretrained_cell_embedding=False,
                                     fov_size=FOV, num_resolutions=NUM_RES,
                                     tissue_filtering_threshold=None)
            np.testing.assert_array_equal(ds[0]["spatial_graph"].ndata["feat"].numpy(), expected)

if __name__ == "__main__":
    unittest.main()
