"""Memory-aware inference must keep the original pixel grid and settings."""
import numpy as np
import pytest
import src.segmentation as segmentation
from src.types import SegmentationParams


@pytest.mark.parametrize('free_gib,total_gib,expected', [(3,4,1), (1,16,1), (5,8,2), (3,16,2), (6,16,4), (10,16,8)])
def test_tile_batch_avoids_small_gpu_memory_spill(free_gib,total_gib,expected):
    assert segmentation._tile_batch_for_memory(free_gib*1024**3,total_gib*1024**3) == expected


def test_cellpose_receives_small_batch_without_resizing(monkeypatch):
    seen = {}
    class Model:
        def eval(self, image, **kwargs):
            seen.update(kwargs)
            seen['shape'] = image.shape
            return (np.ones(image.shape[:2], dtype=np.int32),)
    monkeypatch.setattr(segmentation, 'cellpose_available', lambda: True)
    monkeypatch.setattr(segmentation, '_resolve_device', lambda use: True)
    monkeypatch.setattr(segmentation, '_get_cellpose_model', lambda *args: Model())
    monkeypatch.setattr(segmentation, '_inference_tile_batch', lambda use: 1)
    labels, info = segmentation.segment_with_cellpose(np.zeros((80,120)), SegmentationParams())
    assert seen['batch_size'] == 1
    assert seen['shape'] == (80,120,3)
    assert labels.shape == (80,120)
    assert info['tile_batch_size'] == 1
    assert info['analysis_scale'] == 1.0
    assert info['device'] == 'cuda'
