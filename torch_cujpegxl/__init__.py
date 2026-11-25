"""Encode and decode independent camera JPEG XL files with CUDA tensors."""

import torch as torch

from ._native import Decoder, Encoder

__all__ = ['Decoder', 'Encoder']
