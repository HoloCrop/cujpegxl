# SPDX-License-Identifier: BSD-3-Clause
import pytest
import torch
from torch_cujpegxl import AsyncDecoder, AsyncEncoder, Decoder, Encoder


def test_async_batches_across_streams_and_ring_wrap():
    width, height = 513, 259
    source = torch.cuda.Stream()
    with torch.cuda.stream(source):
        frames = torch.randint(256, (9, height, width, 3), dtype=torch.uint8, device='cuda')
        with AsyncEncoder(width, height, .36, pipeline_depth=2) as encoder:
            first = encoder.encode_batch_async(frames[:3])
            second = encoder.encode_batch_async(frames[3:4])
            third = encoder.encode_batch_async(frames[4:])
        files = first.result() + second.result() + third.result()
        reference = Decoder(width, height).decode_batch(Encoder(width, height, .36).encode_batch(frames))
        assert len(files) == 9
        maps = torch.tensor([[[3., 7.], [12., 15.]]], device='cuda')
        with AsyncDecoder(width, height) as decoder:
            full = decoder.decode_batch_async(files)
            remapped = decoder.decode_batch_async(files, maps)
    torch.cuda.current_stream().wait_stream(source)
    torch.testing.assert_close(full.result(), reference)
    torch.testing.assert_close(remapped.result()[:, 0, 0], reference[:, 7, 3])
    torch.testing.assert_close(remapped.result()[:, 0, 1], reference[:, 15, 12])


def test_async_error_and_drain():
    width, height = 513, 259
    image = torch.randint(256, (height, width, 3), dtype=torch.uint8, device='cuda')
    with AsyncEncoder(width, height, .36, pipeline_depth=2) as encoder:
        file = encoder.encode_batch_async([image]).result()[0]
        invalid = encoder.encode_batch_async([image[1:]])
    with pytest.raises(RuntimeError, match="expected HWC"):
        invalid.result()
    with AsyncDecoder(width, height) as decoder:
        corrupt = decoder.decode_batch_async([file, file[:-64] + bytes(64)])
        valid = decoder.decode_batch_async([file])
    with pytest.raises(RuntimeError, match="batch index 1"):
        corrupt.result()
    assert torch.equal(valid.result()[0], Decoder(width, height).decode(file))
