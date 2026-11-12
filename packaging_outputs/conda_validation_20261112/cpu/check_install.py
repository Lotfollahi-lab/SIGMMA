"""Check a regular SIGMMA wheel installation without downloading model weights."""
from __future__ import annotations

import argparse
from importlib.metadata import version
import json
from pathlib import Path
import platform
import sys


def validate_versions(versions: dict[str, str], device: str) -> None:
    """Reject research/nightly binaries and mismatched release candidates."""
    for name, expected in {'torch': '2.3.1', 'torchvision': '0.18.1'}.items():
        if versions[name].split('+')[0] != expected:
            raise ValueError(f'{name}: expected {expected}, found {versions[name]}')
    expected_dgl = '2.4.0+cu121' if device == 'cuda' else '2.4.0'
    if versions['dgl'] != expected_dgl:
        raise ValueError(f'dgl: expected {expected_dgl}, found {versions["dgl"]}; use the matching official wheel index')
    major, minor = map(int, versions['numpy'].split('.')[:2])
    if major != 1 or minor < 26:
        raise ValueError(f'numpy: expected >=1.26,<2, found {versions["numpy"]}')


def check_install(device_name: str) -> dict:
    if sys.version_info[:2] != (3, 11):
        raise ValueError('The release candidate requires Python 3.11.')
    if platform.system() != 'Linux' or platform.machine() != 'x86_64':
        raise ValueError('The release candidate targets Linux x86_64.')
    versions = {name: version(name) for name in ('sigmma', 'torch', 'torchvision', 'dgl', 'numpy')}
    validate_versions(versions, device_name)

    import torch
    import torchvision
    import dgl
    import sigmma
    from sigmma.datasets import SpatialPatchDataset
    from sigmma.utils.train_utils import load_model
    from sigmma.model.layers_graph import StructureLearning

    package_file = Path(sigmma.__file__).resolve()
    if not package_file.is_relative_to(Path(sys.prefix).resolve()):
        raise ValueError(f'SIGMMA resolves outside this environment: {package_file}; install the wheel rather than using source/PYTHONPATH')
    if device_name == 'cuda':
        if torch.version.cuda != '12.1' or not torch.cuda.is_available():
            raise ValueError('CUDA check requires a CUDA 12.1 PyTorch build and an available GPU.')
    elif torch.version.cuda is not None:
        raise ValueError('CPU profile requires the CPU PyTorch build; use the CPU index.')
    device = torch.device(device_name)
    graph = dgl.graph(([0, 1, 2, 3], [1, 2, 3, 0]), num_nodes=4).to(device)
    graph = dgl.add_self_loop(graph)
    features = torch.arange(12, dtype=torch.float32, device=device).reshape(4, 3).requires_grad_()
    convolution = dgl.nn.GraphConv(3, 2).to(device)
    sage = dgl.nn.SAGEConv(3, 2, 'lstm').to(device)
    output = convolution(graph, features) + sage(graph, features)
    output.square().mean().backward()
    if features.grad is None or not torch.isfinite(features.grad).all():
        raise AssertionError('Graph convolution backward produced invalid gradients.')
    weights = dgl.nn.functional.edge_softmax(graph, torch.ones(graph.num_edges(), 1, device=device))
    if not torch.isfinite(weights).all():
        raise AssertionError('Edge softmax produced invalid values.')
    batch = dgl.batch([graph, graph])
    if [g.num_nodes() for g in dgl.unbatch(batch)] != [4, 4]:
        raise AssertionError('Graph batch/unbatch failed.')
    subgraph = dgl.edge_subgraph(graph, torch.arange(4, device=device), relabel_nodes=False)
    if subgraph.num_nodes() != 4 or subgraph.num_edges() != 4:
        raise AssertionError('Edge subgraph failed.')
    return {'status': 'passed', 'device': device_name, 'versions': versions,
            'sigmma_file': str(package_file), 'model_weights_downloaded': False,
            'checks': ['core_imports', 'graphconv_forward_backward', 'sageconv_lstm_forward_backward',
                       'edge_softmax', 'batch_unbatch', 'edge_subgraph']}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    args = parser.parse_args()
    try:
        print(json.dumps(check_install(args.device), indent=2))
    except Exception as exc:
        print(f'Installation check failed: {exc}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
