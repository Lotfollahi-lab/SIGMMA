"""Run the Xenium preprocessing workflow and validate its SIGMMA manifest.

The external workflow owns registration, graph construction and TERRA inference;
this adapter does not substitute preprocessing algorithms or cell embeddings.
"""
import argparse
import csv
import json
from pathlib import Path, PurePosixPath
import re
import shlex
import subprocess


def safe_id(value):
    # IDs are interpolated into shell scripts by the upstream Nextflow workflow.
    if not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_./-]*", value):
        raise ValueError(f"Unsafe sample/ROI ID: {value!r}")
    if any(p in ("", ".", "..") for p in value.split("/")):
        raise ValueError(f"Unsafe sample/ROI ID: {value!r}")
    return value


def output_ids(samples, staging_path):
    """Match upstream roiIds, rejecting ambiguous output destinations."""
    ids, seen_samples = [], set()
    for line in Path(samples).read_text().splitlines():
        if not line.strip():
            continue
        sample = safe_id(line.strip())
        if sample in seen_samples:
            raise ValueError(f"Duplicate sample: {sample}")
        seen_samples.add(sample)
        folder = Path(staging_path) / sample
        roi = json.loads((folder / "roi.json").read_text())
        bundle = folder / roi["xenium"]["file"]
        for name in ("cell_feature_matrix.h5", "cells.csv.gz", "experiment.xenium"):
            if not (bundle / name).is_file():
                raise ValueError(f"Missing Xenium input: {bundle / name}")
        if not ((bundle / "morphology_focus").is_dir() or
                (bundle / "morphology_focus.ome.tif").is_file()):
            raise ValueError(f"Missing Xenium morphology: {bundle}")
        rois = roi["xenium"].get("spatial_um")
        if rois is not None and (not isinstance(rois, dict) or not rois):
            raise ValueError(f"spatial_um must be a nonempty ROI mapping: {sample}")
        images = roi["hae"]["file"]
        if isinstance(images, dict):
            if not rois or set(images) != set(rois):
                raise ValueError(f"H&E mapping must match ROI names: {sample}")
            images = list(images.values())
        else:
            images = [images]
        for image in images:
            if not (folder / image).is_file():
                raise ValueError(f"Missing H&E image: {folder / image}")
        for name in rois or [None]:
            if name is not None:
                safe_id(name)
                if "/" in name:
                    raise ValueError("ROI names must be single path components")
            ident = str(PurePosixPath(sample).parent / name) if name else sample
            if ident in ids:
                raise ValueError(f"Output ID collision: {ident}")
            ids.append(ident)
    if not ids:
        raise ValueError("Sample list is empty")
    return ids


def validate_section(image_path, spatial_path, index_path, latent_key, tile_size,
                     embedding_dim=None):
    """Validate shapes, units, finite features and index bounds without loading WSI pixels."""
    import anndata as ad
    import numpy as np
    import zarr
    from scipy import sparse

    image = zarr.open(str(image_path), mode="r")
    if len(image.shape) != 3 or image.shape[0] != 3 or image.dtype != np.dtype("uint8"):
        raise ValueError(f"Expected uint8 RGB (3,H,W): {image_path}")
    resolution = image.attrs.get("parameters", {}).get("resolution_um_per_px")
    if resolution is None or not np.isclose(float(resolution), 0.5):
        raise ValueError(f"SIGMMA CLI requires image metadata at 0.5 um/pixel: {image_path}")
    index = np.asarray(zarr.open(str(index_path), mode="r")[:])
    if index.ndim != 2 or index.shape[1] != 3 or not len(index):
        raise ValueError(f"Expected nonempty (N,3) tile index: {index_path}")
    if (not np.isfinite(index).all() or (index < 0).any() or
            not np.equal(index, np.floor(index)).all() or
            (index[:, 0] + tile_size > image.shape[1]).any() or
            (index[:, 1] + tile_size > image.shape[2]).any() or
            (index[:, 2] > tile_size ** 2).any()):
        raise ValueError(f"Invalid tile coordinates/tissue counts: {index_path}")
    adata = ad.read_h5ad(spatial_path)
    if not adata.n_obs or not adata.obs_names.is_unique:
        raise ValueError(f"Cells must be nonempty with unique IDs: {spatial_path}")
    coords = np.asarray(adata.obsm["spatial"])
    if coords.shape != (adata.n_obs, 2) or not np.isfinite(coords).all():
        raise ValueError("spatial must contain finite (x,y) coordinates in micrometres")
    if ((coords < 0).any() or (coords[:, 0] >= image.shape[2] * 0.5).any() or
            (coords[:, 1] >= image.shape[1] * 0.5).any()):
        raise ValueError("Cell coordinates fall outside the image; inspect registration/ROI origin")
    graph = adata.obsp["spatial_connectivities"]
    values = graph.data if sparse.issparse(graph) else np.asarray(graph)
    if graph.shape != (adata.n_obs, adata.n_obs) or not np.isfinite(values).all():
        raise ValueError("Invalid spatial_connectivities")
    if latent_key not in adata.obsm:
        raise ValueError(f"Missing obsm[{latent_key!r}]; available: {list(adata.obsm)}")
    features = adata.obsm[latent_key]
    values = features.data if sparse.issparse(features) else np.asarray(features)
    if (len(features.shape) != 2 or features.shape[0] != adata.n_obs or
            features.shape[1] == 0 or not np.isfinite(values).all()):
        raise ValueError(f"Invalid cell embeddings: {latent_key}")
    if embedding_dim is not None and features.shape[1] != embedding_dim:
        raise ValueError(f"Embedding width {features.shape[1]} != checkpoint width {embedding_dim}")
    return features.shape[1]


def build_manifest(ids, outdir, manifest, latent_key="cell_emb", tile_size=256,
                   stride=256, embedding_dim=None):
    outdir, manifest = Path(outdir).resolve(), Path(manifest).resolve()
    if manifest.exists():
        raise FileExistsError(f"Refusing to overwrite {manifest}")
    rows, widths = [], set()
    for ident in ids:
        safe_id(ident)
        image = outdir / "aligned_images_zarr" / f"{ident}.zarr"
        spatial = outdir / "terra_adata" / f"{ident}.h5ad"
        index = outdir / "index_array" / ident / f"tile_{tile_size}_stride_{stride}.zarr"
        widths.add(validate_section(image, spatial, index, latent_key, tile_size, embedding_dim))
        rows.append(dict(sample_name=ident, hematoxylin_eosin_image=str(image),
                         spatial_preprocessed=str(spatial), annotation="", index=str(index)))
    if not rows or len(widths) != 1:
        raise ValueError("Sections must be nonempty and share one embedding width")
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {manifest}: {len(rows)} sections; latent_embedding_name={latent_key}; "
          f"spatial_encoder_input_dim={next(iter(widths))}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["run", "manifest"])
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--staging-path", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--preprocessing-repo", type=Path)
    parser.add_argument("--profile", choices=["docker", "singularity"], default="docker")
    parser.add_argument("--config", type=Path, help="Site/container Nextflow configuration")
    parser.add_argument("--work-dir", type=Path, default=Path("work-preprocessing"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs and print command only")
    parser.add_argument("--latent-key", default="cell_emb")
    parser.add_argument("--embedding-dim", type=int)
    parser.add_argument("--tile-size", type=int, default=256)
    parser.add_argument("--stride", type=int, default=256)
    args = parser.parse_args(argv)
    if args.tile_size <= 0 or args.stride <= 0:
        parser.error("tile-size and stride must be positive")
    if args.manifest.exists():
        parser.error(f"Manifest already exists: {args.manifest}; choose a new destination")
    ids = output_ids(args.samples, args.staging_path)
    if args.action == "run":
        if not args.preprocessing_repo or not (args.preprocessing_repo / "main.nf").is_file():
            parser.error("run requires --preprocessing-repo pointing to a checkout with main.nf")
        command = ["nextflow", "run", str(args.preprocessing_repo.resolve() / "main.nf"),
                   "-profile", args.profile, "-work-dir", str(args.work_dir.resolve())]
        if args.config:
            if not args.config.is_file():
                parser.error(f"Missing Nextflow config: {args.config}")
            command += ["-c", str(args.config.resolve())]
        command += ["--samples", str(args.samples.resolve()), "--staging_path",
                    str(args.staging_path.resolve()), "--outdir", str(args.outdir.resolve()),
                    "--mode", "both", "--resolution_um_per_px", "0.5", "--tile_configs",
                    f"{args.tile_size}x{args.stride}"]
        if args.resume:
            command.append("-resume")
        print(shlex.join(command), flush=True)
        if args.dry_run:
            return
        subprocess.run(command, check=True)
    elif args.dry_run:
        parser.error("--dry-run applies only to run")
    build_manifest(ids, args.outdir, args.manifest, args.latent_key,
                   args.tile_size, args.stride, args.embedding_dim)


if __name__ == "__main__":
    main()
