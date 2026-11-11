"""Source-specific graph caches. Paths/options are fingerprinted, not file contents.

Rebuild in a fresh directory if source files are changed in place.
"""
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np

INDEX_NAME = 'cache_index.json'
FORMAT = 'hier-cache-source-v2'


def source_metadata(spatial_file, indices_file, fov_size, num_resolutions,
                    tissue_threshold, image_resolution_um_per_pixel):
    spatial = Path(spatial_file).expanduser().resolve()
    return {
        'format': FORMAT, 'section_id': spatial.parent.name,
        'spatial_file': str(spatial),
        'indices_file': str(Path(indices_file).expanduser().resolve()),
        'fov_size': int(fov_size), 'num_resolutions': int(num_resolutions),
        'tissue_threshold': None if tissue_threshold is None else float(tissue_threshold),
        'image_resolution_um_per_pixel': float(image_resolution_um_per_pixel),
    }


def cache_key(metadata):
    payload = json.dumps(metadata, sort_keys=True, separators=(',', ':'), allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def cache_filename(metadata):
    return f"{metadata['section_id']}__{cache_key(metadata)}.npz"


@contextmanager
def exclusive_output(path):
    """Publish atomically without replacing any existing file or symlink."""
    path = Path(path)
    if os.path.lexists(path):
        raise FileExistsError(f'Refusing to overwrite {path}')
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix='.partial-', delete=False) as f:
            temporary = Path(f.name)
            yield f
            f.flush()
            os.fsync(f.fileno())
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def initialize_index(directory, metadata_entries):
    """Write an immutable plan before workers run; existing directories must be empty."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if any(directory.iterdir()):
        raise FileExistsError(f'Use a new empty directory for source-v2: {directory}')
    entries = {cache_key(m): {'file': cache_filename(m), 'metadata': m}
               for m in metadata_entries}
    index = {'format': FORMAT, 'entries': entries}
    with exclusive_output(directory / INDEX_NAME) as f:
        f.write((json.dumps(index, indent=2, sort_keys=True) + '\n').encode())
    return index


def save_source_cache(path, metadata, **arrays):
    with exclusive_output(path) as f:
        np.savez(f, source_metadata_json=np.array(json.dumps(metadata, sort_keys=True)), **arrays)


def load_hier_cache(directory, metadata):
    """Strict v2 lookup; section-only legacy lookup only in unindexed directories."""
    directory = Path(directory)
    index_path = directory / INDEX_NAME
    if index_path.exists():
        index = json.loads(index_path.read_text())
        if index.get('format') != FORMAT:
            raise ValueError(f'Unsupported hierarchical cache index: {index_path}')
        entry = index.get('entries', {}).get(cache_key(metadata))
        if entry is None:
            raise ValueError(f'No cache for source/options: {metadata} in {index_path}; '
                             'section-name fallback is disabled')
        if entry.get('metadata') != metadata or entry.get('file') != cache_filename(metadata):
            raise ValueError(f'Cache index metadata mismatch: {index_path}')
        path = directory / entry['file']
        if not path.is_file():
            raise FileNotFoundError(f'Indexed cache is not complete: {path}')
        with np.load(path, allow_pickle=False) as f:
            if 'source_metadata_json' not in f:
                raise ValueError(f'Cache lacks source metadata: {path}')
            if json.loads(str(f['source_metadata_json'].item())) != metadata:
                raise ValueError(f'Cache file source/options mismatch: {path}')
            return {k: f[k] for k in f.files if k != 'source_metadata_json'}
    if any(directory.glob('*__*.npz')):
        raise ValueError(f'Source-specific files found without {INDEX_NAME}: {directory}')
    path = directory / f"{metadata['section_id']}.npz"
    if not path.exists():
        return None  # existing legacy on-the-fly behavior
    with np.load(path, allow_pickle=False) as f:
        return {k: f[k] for k in f.files}
