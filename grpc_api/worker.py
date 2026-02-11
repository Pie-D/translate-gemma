import asyncio
import logging
from typing import Any, List, Optional, Tuple

# Reuse model + processor + batching sync function from existing implementation.
import main as http_api


# Queue riêng cho gRPC để tránh phụ thuộc vào worker của FastAPI (`main._gpu_worker`)
grpc_request_queue: asyncio.Queue = asyncio.Queue()

_WORKER_TASK: Optional[asyncio.Task] = None
_START_LOCK = asyncio.Lock()


async def _grpc_gpu_worker() -> None:
    """
    Worker gom batch và chạy GPU inference.

    Khác với `main._gpu_worker()`: có guard `fut.done()/cancelled()` để tránh
    InvalidStateError khi client huỷ request / timeout.
    """
    loop = asyncio.get_running_loop()
    while True:
        batch: List[Tuple[str, Any, asyncio.Future]] = []
        try:
            while len(batch) < getattr(http_api, "BATCH_SIZE", 32):
                item = await asyncio.wait_for(
                    grpc_request_queue.get(),
                    timeout=getattr(http_api, "BATCH_TIMEOUT", 0.05),
                )
                batch.append(item)
        except asyncio.TimeoutError:
            pass

        if not batch:
            continue

        try:
            results = await loop.run_in_executor(
                getattr(http_api, "_executor"),
                getattr(http_api, "_process_batch_sync"),
                batch,
            )
        except Exception:
            logging.exception("gRPC GPU worker failed while processing batch")
            # Không để worker chết, tiếp tục vòng lặp
            continue

        for fut, res in results:
            # Client có thể đã huỷ request / deadline → future bị cancel/done.
            if fut.cancelled() or fut.done():
                continue
            try:
                if isinstance(res, Exception):
                    fut.set_exception(res)
                else:
                    fut.set_result(res)
            except asyncio.InvalidStateError:
                # Race condition: future vừa bị cancel/done
                continue


async def ensure_worker_started() -> None:
    """
    Đảm bảo chỉ có 1 worker chạy cho gRPC server.
    """
    global _WORKER_TASK
    async with _START_LOCK:
        if _WORKER_TASK is None or _WORKER_TASK.done():
            _WORKER_TASK = asyncio.create_task(_grpc_gpu_worker())

