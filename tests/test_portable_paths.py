"""User configuration must work without any author-specific filesystem."""
import importlib
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from sigmma import caches
from sigmma.utils import tissue_map
from sigmma.evaluation import svg_selection, transcript_binning


class PortablePathsTest(unittest.TestCase):
    def test_cache_default_and_override(self):
        with patch.dict(os.environ, {}, clear=True):
            importlib.reload(caches)
            self.assertEqual(caches.CACHE_ROOT, Path.home() / '.cache/sigmma')
        with patch.dict(os.environ, {'SIGMMA_CACHE_ROOT': '~/custom-cache'}):
            importlib.reload(caches)
            self.assertEqual(caches.CACHE_ROOT, Path.home() / 'custom-cache')
            self.assertEqual(caches.sigmma_cache_dir_gex_2d(ckpt_tag='x', root='custom'), Path('custom/sigmma/gex_2d/x'))
        importlib.reload(caches)

    def test_missing_mapping_is_lazy_and_actionable(self):
        with patch.dict(os.environ, {}, clear=True), patch.object(svg_selection, 'SECTION_PANEL_CSV', None):
            with self.assertRaisesRegex(ValueError, 'SIGMMA_SAMPLE_METADATA_CSV'):
                tissue_map.load_tissue_map()
            with self.assertRaisesRegex(ValueError, 'SIGMMA_SECTION_PANEL_CSV'):
                svg_selection.tissue_of('example')

    def test_mapping_paths_and_cache_switch(self):
        with tempfile.TemporaryDirectory() as d:
            a, b = Path(d) / 'a.csv', Path(d) / 'b.csv'
            a.write_text('sample_id,tissue_type,sample_name,tissue\ns1,lung,s1,lung\n')
            b.write_text('sample_id,tissue_type,sample_name,tissue\ns1,liver,s1,liver\n')
            with patch.dict(os.environ, {'SIGMMA_SAMPLE_METADATA_CSV': str(a), 'SIGMMA_SECTION_PANEL_CSV': str(a)}):
                self.assertEqual(tissue_map.tissue_of('s1'), 'lung')
                self.assertEqual(svg_selection.tissue_of('s1'), 'lung')
                self.assertEqual(tissue_map.load_tissue_map(b)['s1'], 'liver')
                self.assertEqual(svg_selection.tissue_of('s1', csv_path=str(b)), 'liver')
                self.assertEqual(svg_selection.tissue_of('s1'), 'lung')
                with self.assertRaises(FileNotFoundError):
                    tissue_map.load_tissue_map(Path(d) / 'missing.csv')
                with self.assertRaises(FileNotFoundError):
                    svg_selection.tissue_of('s1', csv_path=str(Path(d) / 'missing.csv'))
            with patch.dict(os.environ, {'SIGMMA_SAMPLE_METADATA_CSV': str(b)}):
                self.assertEqual(tissue_map.tissue_of('s1'), 'liver')

    def test_manifest_resolves_without_global_roots(self):
        with patch.dict(os.environ, {}, clear=True), tempfile.TemporaryDirectory() as d:
            importlib.reload(transcript_binning)
            self.assertIsNone(transcript_binning.LUSTRE_HE_ROOT)
            self.assertIsNone(transcript_binning.NFS_XENIUM_ROOT)
            image = Path(d) / 'hematoxylin_eosin.zarr'
            self.assertIsNone(transcript_binning.manifest_path_to_transcripts_parquet(str(image)))
            parquet = Path(d) / 'transcripts.parquet'
            parquet.touch()
            self.assertEqual(transcript_binning.manifest_path_to_transcripts_parquet(str(image)), str(parquet))


if __name__ == '__main__':
    unittest.main()
