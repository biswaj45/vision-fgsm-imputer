"""
App package for anti-deepfake protection.
"""

from .gradio_app import AntiDeepfakeApp
from .demo_utils import create_side_by_side, compute_difference_heatmap

__all__ = ['AntiDeepfakeApp', 'create_side_by_side', 'compute_difference_heatmap']
