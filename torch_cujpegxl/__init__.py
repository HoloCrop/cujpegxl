"""Encode full-resolution CUDA RGB tensors as independent JPEG XL files."""

import torch as torch

from ._native import Encoder

__all__ = ['Encoder']
