# SPDX-License-Identifier: BSD-3-Clause
import pytest
import torch

from torch_cujpegxl import Encoder


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA encoder')
def test_streaming_retains_pitched_inputs_and_returns_owned_files_in_order():
    height, width = 259, 513
    generator = torch.Generator().manual_seed(17)
    source = torch.randint(256, (7, height, width + 5, 3), dtype=torch.uint8, generator=generator)
    expected = Encoder(width, height, 0.9).encode_batch(source[:, :, :width].contiguous().cuda())
    encoder = Encoder(width, height, 0.9)
    producer = torch.cuda.Stream()
    with torch.cuda.stream(producer):
        first = encoder.push(list(source[:2].cuda()[:, :, :width].unbind()))
        # A bad batch must leave the two outstanding images intact.
        with pytest.raises(RuntimeError, match='HWC RGB'):
            encoder.push([source[2, :, :width].cuda(), source[3].cuda()])
        rest = encoder.push(list(source[2:].cuda()[:, :, :width].unbind()))
    actual = first + rest + encoder.flush()
    assert actual == expected
    assert len(set(actual)) == len(source)
    assert encoder.flush() == []
