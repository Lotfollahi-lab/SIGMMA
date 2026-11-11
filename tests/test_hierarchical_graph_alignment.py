"""Regression test: hierarchical-graph node/edge attributes must be aligned
with the full-FOV spatial graph.

`spatial_graph` nodes follow AnnData row order, while the hierarchical
(disjoint) graph built by `dgl.batch` follows sub-FOV order. Node attributes
(`x_pos`, `y_pos`, `cell_coordinates`) and the intra/inter edge mask must be
transferred by cell identity, not by local node number.
"""
import importlib.util
from pathlib import Path

import anndata as ad
import numpy as np
import pytest
import scipy.sparse as sp
import torch
import zarr

from sigmma.datasets.dataset import SpatialPatchDataset

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


@pytest.mark.parametrize("use_cache", [False, True])
def test_hierarchical_graph_alignment(tmp_path, use_cache):
    files = _make_section(tmp_path)
    cache_dir = None
    if use_cache:
        cache_dir = tmp_path / "hier_cache"
        _build_cache(files, cache_dir)

    ds = SpatialPatchDataset(
        image_file=files["image_file"],
        spatial_file=files["spatial_file"],
        indices_file=files["indices_file"],
        fov_size=FOV,
        num_resolutions=NUM_RES,
        image_resolution_um_per_pixel=UM_PER_PX,
        tissue_filtering_threshold=None,
        hierarchical_graph=True,
        number_of_genes=3,
        hier_cache_dir=cache_dir,
    )
    if use_cache:
        assert ds._hier_cache is not None
    g = ds[0]["spatial_graph"]

    cell_id = g.ndata["cell_id"].long()
    assert cell_id.tolist() == list(range(len(COORDS_PX)))
    true_coords = torch.from_numpy(COORDS_PX)[cell_id]

    # Node attributes belong to the node's own cell.
    assert torch.equal(g.ndata["cell_coordinates"].float(), true_coords)
    assert torch.equal(g.ndata["x_pos"].long(), (true_coords[:, 0] // GRID).long())
    assert torch.equal(g.ndata["y_pos"].long(), (true_coords[:, 1] // GRID).long())

    # Edge is intra (-1, weight 1) iff both endpoints share a sub-FOV.
    src, dst = g.edges()
    sf = (true_coords[:, 0] // GRID) * (FOV // GRID) + (true_coords[:, 1] // GRID)
    same = sf[src] == sf[dst]
    assert torch.equal(g.edata["mask"] == -1, same)
    assert torch.equal(g.edata["weight"] == 1.0, same)
    assert same.any() and (~same).any()
