import argparse
import asyncio

import grpc

from grpc_api._proto import load_generated


pb2, pb2_grpc = load_generated()


async def main() -> None:
    parser = argparse.ArgumentParser(description="TranslateGemma gRPC client example")
    parser.add_argument("--target", default="127.0.0.1:50051")
    parser.add_argument("--source", default="en")
    parser.add_argument("--dest", default="vi")
    parser.add_argument("--text", default="Hello, how are you?")
    parser.add_argument("--image-url", default="")
    args = parser.parse_args()

    async with grpc.aio.insecure_channel(args.target) as channel:
        stub = pb2_grpc.TranslateGemmaServiceStub(channel)

        if args.image_url:
            resp = await stub.TranslateImage(
                pb2.TranslateImageRequest(
                    source_lang_code=args.source,
                    target_lang_code=args.dest,
                    url=args.image_url,
                )
            )
            print(resp.translation)
        else:
            resp = await stub.TranslateText(
                pb2.TranslateTextRequest(
                    source_lang_code=args.source,
                    target_lang_code=args.dest,
                    text=args.text,
                )
            )
            print(resp.translation)


if __name__ == "__main__":
    asyncio.run(main())

