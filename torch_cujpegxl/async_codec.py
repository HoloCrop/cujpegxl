"""Bounded batch submission. Futures resolve to completed bytes or CUDA pixels."""
from collections import deque
from collections.abc import Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from threading import Condition, Lock, Semaphore, Thread

import torch

from ._native import Decoder, Encoder


@dataclass
class EncodeBatch:
    images: tuple[torch.Tensor, ...]
    ready: torch.cuda.Event
    future: Future[list[bytes]]
    count: int
    files: list[bytes] = field(default_factory=list)


class AsyncEncoder(AbstractContextManager):
    """Encode successive batches through one native slot pool, preserving submission order."""

    def __init__(self, width: int, height: int, distance: float, device_index: int = 0,
                 pipeline_depth: int = 6, max_batches: int = 2):
        if max_batches < 1:
            raise ValueError("max_batches must be positive")
        self.device = torch.device("cuda", device_index)
        self.encoder = Encoder(width, height, distance, device_index, pipeline_depth)
        self.stream = torch.cuda.Stream(device=self.device)
        self.capacity = max_batches
        self.active = 0
        self.closed = False
        self.condition = Condition()
        self.queue: deque[EncodeBatch] = deque()
        self.worker = Thread(target=self._run, name="jxl-encode")
        self.worker.start()

    def encode_batch_async(self, images: Sequence[torch.Tensor]) -> Future[list[bytes]]:
        """Submit on the caller's current CUDA stream. Blocks only for queue capacity."""
        images = tuple(images)
        future: Future[list[bytes]] = Future()
        with self.condition:
            self.condition.wait_for(lambda: self.closed or self.active < self.capacity)
            if self.closed:
                raise RuntimeError("encoder is closed")
            future.set_running_or_notify_cancel()
            if not images:
                future.set_result([])
                return future
            ready = torch.cuda.current_stream(self.device).record_event()
            self.queue.append(EncodeBatch(images, ready, future, len(images)))
            self.active += 1
            self.condition.notify_all()
        return future

    def _run(self):
        pending: deque[EncodeBatch] = deque()
        try:
            with torch.cuda.stream(self.stream):
                while True:
                    with self.condition:
                        self.condition.wait_for(lambda: self.closed or self.queue or pending)
                        batch = self.queue.popleft() if self.queue else None
                        if batch is None and not pending:
                            return
                    if batch is not None:
                        pending.append(batch)
                        self.stream.wait_event(batch.ready)
                        files = self.encoder.push(batch.images)
                        batch.images = ()
                    else:
                        files = [self.encoder._receive()]
                    for file in files:
                        first = pending[0]
                        first.files.append(file)
                        if len(first.files) == first.count:
                            pending.popleft()
                            with self.condition:
                                self.active -= 1
                                self.condition.notify_all()
                            first.future.set_result(first.files)
        except BaseException as error:
            with self.condition:
                self.closed = True
                pending.extend(self.queue)
                self.queue.clear()
                self.condition.notify_all()
            for batch in pending:
                batch.future.set_exception(error)

    def close(self):
        with self.condition:
            self.closed = True
            self.condition.notify_all()
        self.worker.join()

    def __exit__(self, *exc):
        self.close()


class AsyncDecoder(AbstractContextManager):
    """Overlap CPU parsing and CUDA rendering with completion of earlier batches."""

    def __init__(self, width: int, height: int, device_index: int = 0, max_batches: int = 2):
        if max_batches < 1:
            raise ValueError("max_batches must be positive")
        self.device = torch.device("cuda", device_index)
        self.decoder = Decoder(width, height, device_index)
        self.stream = torch.cuda.Stream(device=self.device)
        self.slots = Semaphore(max_batches)
        self.lock = Lock()
        self.submitter = ThreadPoolExecutor(1, thread_name_prefix="jxl-decode")
        self.finisher = ThreadPoolExecutor(1, thread_name_prefix="jxl-complete")
        self.closed = False

    def decode_batch_async(self, images: Sequence[bytes], maps: torch.Tensor | None = None) -> Future[torch.Tensor]:
        """Result is validated and ready on any CUDA stream. Optional maps belong to the caller's stream."""
        files = tuple(images)
        ready = torch.cuda.current_stream(self.device).record_event() if maps is not None else None

        def enqueue():
            with torch.cuda.stream(self.stream):
                if maps is not None:
                    self.stream.wait_event(ready)
                    maps.record_stream(self.stream)
                return self.decoder._enqueue(files, maps)

        self.slots.acquire()
        with self.lock:
            if self.closed:
                self.slots.release()
                raise RuntimeError("decoder is closed")
            submitted = self.submitter.submit(enqueue)
            result: Future[torch.Tensor] = Future()
            result.set_running_or_notify_cancel()

            def finish():
                try:
                    pixels = submitted.result().result()
                except BaseException as error:
                    self.slots.release()
                    result.set_exception(error)
                else:
                    self.slots.release()
                    result.set_result(pixels)

            self.finisher.submit(finish)
        return result

    def close(self):
        with self.lock:
            self.closed = True
        self.submitter.shutdown()
        self.finisher.shutdown()

    def __exit__(self, *exc):
        self.close()
