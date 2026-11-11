"""CPU checks for the library/benchmark boundary and moved checkpoint behavior."""
import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]


class SeparationTests(unittest.TestCase):
    def test_core_has_no_benchmark_imports(self):
        for path in (ROOT / 'sigmma').rglob('*.py'):
            for node in ast.walk(ast.parse(path.read_text())):
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom):
                    names = [node.module or '']
                else:
                    continue
                self.assertFalse(any(n.startswith(('sigmma.baselines', 'sigmma_benchmarks'))
                                     for n in names), str(path))
        self.assertFalse((ROOT / 'sigmma/baselines').exists())

    def test_spatial_inputs_include_wrapped_models(self):
        from sigmma.model.inputs import spatial_input
        import torch
        tensor = torch.tensor([[1., 2.]])
        batch = {'spatial_graph': tensor, 'spatial': tensor + 1,
                 'spatial_text': ['gene A']}
        self.assertTrue(torch.equal(spatial_input(SimpleNamespace(), batch, 'cpu'), tensor))
        wrapped = SimpleNamespace(module=SimpleNamespace(spatial_input_key='spatial'))
        self.assertTrue(torch.equal(spatial_input(wrapped, batch, 'cpu'), tensor + 1))
        text_model = SimpleNamespace(spatial_input_key='spatial_text')
        self.assertIs(spatial_input(text_model, batch, 'cpu'), batch['spatial_text'])

    def test_external_factory_and_checkpoint_roundtrip(self):
        import torch
        from ml_collections import ConfigDict
        from sigmma.utils.train_utils import load_model
        cfg = ConfigDict({'model': {'name': 'test_external'}})
        with patch('torch.cuda.is_available', return_value=False):
            with self.assertRaises(NotImplementedError):
                load_model(cfg, train=False)
            calls = []
            def factory(config, train):
                calls.append((config.model.name, train))
                return torch.nn.Linear(2, 2)
            model = load_model(cfg, train=False, model_factory=factory)
            with tempfile.TemporaryDirectory() as tmp:
                checkpoint = Path(tmp) / 'model.pt'
                torch.save({'model_state_dict': model.state_dict()}, checkpoint)
                restored = load_model(cfg, train=False, model_factory=factory,
                                      pretrained_ckpt_path=checkpoint)
            self.assertEqual(calls, [('test_external', False)] * 2)
            for key, value in model.state_dict().items():
                torch.testing.assert_close(value, restored.state_dict()[key])

    def test_moved_bleep_loss_and_state_dict(self):
        import torch
        from sigmma_benchmarks.baselines.bleep.model import BLEEP, bleep_cross_entropy
        with patch('torchvision.models.resnet50') as backbone:
            # Small backbone avoids downloads and the cost of two ResNet instances.
            backbone.return_value = torch.nn.Sequential(torch.nn.AdaptiveAvgPool2d(1),
                                                       torch.nn.Identity())
            model = BLEEP(spot_input_dim=3, image_embedding_dim=3, projection_dim=2,
                          dropout=0., pretrained_image_encoder=False)
            restored = BLEEP(spot_input_dim=3, image_embedding_dim=3, projection_dim=2,
                             dropout=0., pretrained_image_encoder=False)
        restored.load_state_dict(model.state_dict(), strict=True)
        self.assertFalse(any('spatial_input_key' in k for k in model.state_dict()))
        x = torch.tensor([[1., 2.], [3., 1.]], requires_grad=True)
        y = torch.tensor([[0.8, 0.2], [0.1, 0.9]])
        expected = -(y * torch.log_softmax(x, dim=-1)).sum(1)
        torch.testing.assert_close(bleep_cross_entropy(x, y), expected)
        bleep_cross_entropy(x, y, reduction='mean').backward()
        self.assertTrue(torch.isfinite(x.grad).all())

    def test_factories_import_only_selected_baseline(self):
        import sys
        import types
        from ml_collections import ConfigDict
        from sigmma_benchmarks.models import build_model
        fake = types.ModuleType('sigmma_benchmarks.baselines.plip')
        received = {}
        def constructor(**kwargs):
            received.update(kwargs)
            return SimpleNamespace(spatial_input_key='spatial_text')
        fake.PLIP = constructor
        with patch.dict(sys.modules, {'sigmma_benchmarks.baselines.plip': fake}):
            model = build_model(ConfigDict({'model': {'name': 'PLIP'}}), train=False)
        self.assertEqual(model.spatial_input_key, 'spatial_text')
        self.assertTrue(received['freeze_encoders'])
        self.assertIsNone(received['loss_temperature'])


if __name__ == '__main__':
    unittest.main()
