"""Sphinx configuration for the SIGMMA documentation."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

project = "SIGMMA"
author = "Lotfollahi Lab"
copyright = "2026, Lotfollahi Lab"

try:
    from importlib.metadata import version as _v
    release = _v("sigmma")
except Exception:
    release = "2.0.0"
version = release

extensions = [
    "myst_nb",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
    "sphinx_design",
]

# .rst first: autosummary writes its stubs with the first suffix.
source_suffix = {".rst": "restructuredtext", ".md": "myst-nb", ".ipynb": "myst-nb"}
exclude_patterns = ["_build", "**.ipynb_checkpoints", "requirements.txt"]

# Tutorials are executed on the cluster (GPU + data) and committed with outputs.
nb_execution_mode = "off"
myst_enable_extensions = ["colon_fence", "dollarmath"]

# Runtime dependencies are not installed on Read the Docs.
autodoc_mock_imports = [
    "torch", "torchvision", "dgl", "timm", "peft", "transformers", "wandb",
    "zarr", "numcodecs", "h5py", "anndata", "scanpy", "umap", "sklearn",
    "scipy", "pandas", "numpy", "pyarrow", "skimage", "xgboost", "einops",
    "ml_collections", "yaml", "sympy", "PIL", "matplotlib", "cv2", "shapely",
    "openslide", "tifffile", "huggingface_hub", "uni",
    "novae", "liana", "gseapy", "scvi", "lifelines",
]
autosummary_generate = True
autodoc_typehints = "description"
autodoc_member_order = "bysource"
napoleon_google_docstring = True
napoleon_numpy_docstring = False

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "anndata": ("https://anndata.readthedocs.io/en/stable/", None),
    "scanpy": ("https://scanpy.readthedocs.io/en/stable/", None),
}

html_theme = "sphinx_book_theme"
html_title = "SIGMMA"
html_static_path = ["_static"]
html_theme_options = {
    "repository_url": "https://github.com/Lotfollahi-lab/SIGMMA2.0",
    "repository_branch": "main",
    "path_to_docs": "docs",
    "use_repository_button": True,
    "use_issues_button": True,
    "navigation_with_keys": False,
}
