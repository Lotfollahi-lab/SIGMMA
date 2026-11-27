# Process new H&E and spatial transcriptomics data

This tutorial connects the [SIGMMA preprocessing workflow](https://github.com/Lotfollahi-lab/SIGMMA_preprocessing)
to SIGMMA. The raw-input workflow currently supports **10x Xenium bundles with
matching H&E images**. Visium, CosMx and other ST formats need their own input
adapter and registration; do not submit them as Xenium bundles.

The adapter runs the upstream Nextflow pipeline, checks the resulting files,
and writes a manifest consumed by SIGMMA. It does not modify the raw inputs.
The interface was checked against the local preprocessing checkout available
on 2026-11-26; keep a record of your preprocessing Git revision and Nextflow
configuration alongside your results.

## 1. Prepare the environments

Use the SIGMMA environment described in {doc}`../installation` for the adapter
and manifest validation. From this source checkout you can run
`python -m sigmma.cli.preprocess`; installing this revision also provides
`sigmma-preprocess` (not included in the previously published 0.1.0 wheel).

Clone the preprocessing repository separately. Follow its README to install
Nextflow >=24.04 and build/configure the Palom, Zarr and TERRA containers.
Docker or Singularity/Apptainer and an NVIDIA GPU for TERRA are required.
The registration and TERRA processes request roughly 50 GB and 64 GB RAM by
default. TERRA dependencies live in its container, separate from SIGMMA.

```bash
git clone https://github.com/Lotfollahi-lab/SIGMMA_preprocessing.git
```

Build the images as documented upstream, then set their names in a site file:

```groovy
// preprocessing.local.config -- paths/names must exist on your machine
params.container_palom = 'sigmma-palom'
params.container_zarr = 'sigmma-zarr'
params.container_terra = 'sigmma-terra'
// A local TERRA model folder can also be supplied here:
// params.terra_model = '/models/TERRA-96M'
```

For a cluster, add executor/resource settings as described upstream. Make sure
the staging, results and model paths are visible inside containers. The TERRA
model may be downloaded on first use unless a local folder is configured.

## 2. Describe the section

Place your inputs in this layout (the directory names below are examples):

```text
data/raw/new_section/
  slide.svs
  roi.json
  output-XETG.../
    cell_feature_matrix.h5
    cells.csv.gz
    experiment.xenium
    morphology_focus/           # v2+, or morphology_focus.ome.tif for v1
```

Write `data/raw/new_section/roi.json`:

```json
{
  "xenium": {"file": "output-XETG..."},
  "hae": {"file": "slide.svs"}
}
```

Write `samples.txt`, with one sample ID per line:

```text
new_section
```

For TMA cores, use the upstream `examples/roi.json` format: `xenium.spatial_um`
maps ROI names to WKT polygons in Xenium micrometres, and `hae.pixel` describes
H&E crops in image pixels. The upstream workflow shifts cell coordinates to
the ROI origin. Nested sample IDs such as `project/donor/section` are supported;
ROI outputs replace the last component (`project/donor/core_1`). ROI names
must be unique within each parent directory. The adapter rejects collisions.

## 3. Run preprocessing

First validate the input layout and print the command without launching jobs:

```bash
python -m sigmma.cli.preprocess run \
  --preprocessing-repo /path/to/SIGMMA_preprocessing \
  --samples samples.txt --staging-path data/raw \
  --outdir outputs/new_data --manifest outputs/new_data/manifest.csv \
  --profile docker --config preprocessing.local.config \
  --work-dir work-preprocessing --dry-run
```

Remove `--dry-run` to execute. Use `--profile singularity` for that container
engine and `--resume` to reuse Nextflow task results after an interrupted run.
Keep the same work directory and launch directory when resuming. A dry run
checks file presence and IDs; it does not validate image contents, registration,
container availability or GPU execution.

The pipeline performs:

1. Palom H&E-to-Xenium DAPI registration, followed by conversion to RGB Zarr at
   0.5 micrometres/pixel.
2. Tissue masking and 256-pixel tiles with a 256-pixel stride.
3. Xenium cell/count loading, centroid filtering, optional ROI splitting, and
   the upstream 10-nearest-neighbour spatial graph.
4. TERRA inference, yielding `obsm['cell_emb']` and `obsm['neighborhood_emb']`.
5. Validation and a manifest referencing the published outputs by absolute path.

Use `--tile-size` and `--stride` only when your downstream FOV configuration
matches. This adapter fixes image resolution to the SIGMMA CLI's 0.5 µm/pixel.

If preprocessing already finished, build just the manifest:

```bash
python -m sigmma.cli.preprocess manifest \
  --samples samples.txt --staging-path data/raw \
  --outdir outputs/new_data --manifest outputs/new_data/manifest.csv \
  --latent-key cell_emb
```

Keep the original sample list, `roi.json` and raw input layout available for
this step. Existing manifests are never overwritten; choose a new filename
to regenerate one. Missing outputs, empty indices, malformed arrays, duplicate
cell IDs, nonfinite embeddings, out-of-image coordinates and inconsistent
embedding widths stop manifest creation. Image pixels are not loaded into RAM;
validation loads each section's AnnData and index in turn. Inspect the aligned
H&E/DAPI overlays yourself: passing numerical checks does not establish good
registration. Cells outside the image cause an error rather than being silently
discarded; investigate alignment and ROI boundaries before proceeding.

## 4. Use the results in SIGMMA

The manifest contains `sample_name`, `hematoxylin_eosin_image`,
`spatial_preprocessed`, `annotation` (empty) and `index`. Copy the configuration
supplied with your **TERRA-trained SIGMMA checkpoint** and set:

```yaml
dataset:
  query_dataset:
    manifest_path: /absolute/path/to/outputs/new_data/manifest.csv
    pretrained_cell_embedding: true
    latent_embedding_name: cell_emb
    fov_size: 256
    num_resolutions: 3  # must match the checkpoint
    tissue_filtering_threshold: 20000
```

Keep the model architecture, checkpoint path and UNI2-h path from your own
configuration. The adapter prints the embedding width; optionally supply
`--embedding-dim` to require the checkpoint's `spatial_encoder_input_dim`.
Matching width alone does not establish compatibility: use the same TERRA
model and feature key used during SIGMMA training. Do not rename `cell_emb`
to `nemo_latent` to use the legacy tutorial checkpoint. If no compatible
checkpoint exists, train SIGMMA with these features first.

```bash
sigmma-extract-embeddings \
  -config_file my_terra_inference.yaml -out_dir outputs/new_data/embeddings
```

Continue with {doc}`quickstart` and the downstream tutorials using your own
manifest and compatible configuration. This tutorial's adapter is tested on
synthetic outputs; a full registration/TERRA/GPU run requires your actual data
and configured preprocessing environment.
