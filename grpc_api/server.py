import argparse
import asyncio
import logging
from typing import Optional

import grpc
from grpc import aio

# Dùng lại model + queue + worker batching từ FastAPI implementation hiện tại.
# Lưu ý: import này sẽ load model ngay khi import `main`.
import main as http_api  # noqa: E402

from grpc_api._proto import load_generated  # noqa: E402
from grpc_api.worker import ensure_worker_started, grpc_request_queue  # noqa: E402

pb2, pb2_grpc = load_generated()


class TranslateGemmaService(pb2_grpc.TranslateGemmaServiceServicer):
    async def TranslateText(
        self, request, context: grpc.aio.ServicerContext
    ):
        await ensure_worker_started()
        loop = asyncio.get_running_loop()
        fut: asyncio.Future = loop.create_future()
        try:
            req = http_api.TextTranslateRequest(
                source_lang_code=request.source_lang_code,
                target_lang_code=request.target_lang_code,
                text=request.text,
            )
            await grpc_request_queue.put(("text", req, fut))
            res = await fut
            return pb2.TranslateResponse(translation=res.get("translation", ""))
        except asyncio.CancelledError:
            if not fut.done():
                fut.cancel()
            raise
        except Exception as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return pb2.TranslateResponse(translation="")

    async def TranslateImage(
        self, request, context: grpc.aio.ServicerContext
    ):
        await ensure_worker_started()
        loop = asyncio.get_running_loop()
        fut: asyncio.Future = loop.create_future()
        try:
            req = http_api.ImageTranslateRequest(
                source_lang_code=request.source_lang_code,
                target_lang_code=request.target_lang_code,
                url=request.url,
            )
            await grpc_request_queue.put(("image", req, fut))
            res = await fut
            return pb2.TranslateResponse(translation=res.get("translation", ""))
        except asyncio.CancelledError:
            if not fut.done():
                fut.cancel()
            raise
        except Exception as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return pb2.TranslateResponse(translation="")


async def serve(host: str, port: int, max_workers: int) -> None:
    server = aio.server(options=[
        # Giữ kết nối HTTP/2 ổn định hơn trong môi trường NAT/LB
        ("grpc.keepalive_time_ms", 30_000),
        ("grpc.keepalive_timeout_ms", 10_000),
        ("grpc.http2.max_pings_without_data", 0),
        ("grpc.keepalive_permit_without_calls", 1),
    ])
    pb2_grpc.add_TranslateGemmaServiceServicer_to_server(
        TranslateGemmaService(), server
    )
    listen_addr = f"{host}:{port}"
    server.add_insecure_port(listen_addr)

    logging.info("gRPC listening on %s", listen_addr)
    # Start GPU worker trước khi nhận request để tránh race khi request đầu tiên vào dồn dập.
    await ensure_worker_started()
    await server.start()
    await server.wait_for_termination()


def main() -> None:
    parser = argparse.ArgumentParser(description="TranslateGemma gRPC server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=50051)
    parser.add_argument(
        "--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"]
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=10,
        help="Số worker thread nội bộ của gRPC (khác với GPU executor).",
    )
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level))
    asyncio.run(serve(args.host, args.port, args.max_workers))


if __name__ == "__main__":
    main()

