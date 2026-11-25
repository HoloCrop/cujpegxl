# SPDX-License-Identifier: BSD-3-Clause
import pytest
import torch

from torch_cujpegxl import Decoder, Encoder


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA camera decoder')
def test_decode_remap_on_current_stream():
    height, width = 259, 513
    decoder = Decoder(width, height)
    with torch.cuda.stream(torch.cuda.Stream()):
        images = torch.randint(256, (2, height, width, 3), device='cuda', dtype=torch.uint8)
        files = Encoder(width, height, 0.36).encode_batch(images)
        rgb = decoder.decode_batch(files)
        assert (rgb.float() - images).abs().mean() < 50
        maps = torch.tensor([[[[-1, 0], [3.5, 7]]], [[[width - 1, height - 1], [12, 15]]]],
                            device='cuda', dtype=torch.float32)
        views = decoder.decode_batch(files, maps)
        torch.testing.assert_close(views[0, 0, 0], torch.zeros_like(rgb[0, 0, 0]))
        torch.testing.assert_close(views[0, 0, 1], ((rgb[0, 7, 3].int() + rgb[0, 7, 4] + 1) // 2).byte())
        torch.testing.assert_close(views[1, 0, 0], rgb[1, -1, -1])
        torch.testing.assert_close(views[1, 0, 1], rgb[1, 15, 12])
        with pytest.raises(RuntimeError, match='batch index 1'):
            decoder.decode_batch([files[0], files[1][:-100]])


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA camera decoder')
def test_corrupt_entropy_group_reports_batch_index_and_preserves_next_call():
    width, height = 513, 259
    image = torch.randint(256, (height, width, 3), device='cuda', dtype=torch.uint8)
    file = Encoder(width, height, 0.36).encode(image)
    decoder = Decoder(width, height)
    expected = decoder.decode(file)
    damaged = file[:-64] + bytes(64)
    with pytest.raises(RuntimeError, match='batch index 1'):
        decoder.decode_batch([file, damaged, file])
    torch.testing.assert_close(decoder.decode(file), expected)
