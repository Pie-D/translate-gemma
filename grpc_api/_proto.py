import importlib
import os
import sys
from pathlib import Path
from typing import Tuple


def _workspace_root() -> Path:
    # grpc_api/ nằm ngay dưới repo root
    return Path(__file__).resolve().parent.parent


def _proto_path() -> Path:
    return Path(__file__).resolve().parent / "translategemma.proto"


def _generated_ok() -> bool:
    pkg_dir = Path(__file__).resolve().parent
    return (pkg_dir / "translategemma_pb2.py").exists() and (
        pkg_dir / "translategemma_pb2_grpc.py"
    ).exists()


def generate_proto() -> None:
    """
    Generate `translategemma_pb2.py` và `translategemma_pb2_grpc.py` vào thư mục grpc_api/.
    Yêu cầu: `pip install grpcio-tools`.
    """
    try:
        from grpc_tools import protoc  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Chưa có grpcio-tools. Hãy cài: pip install -r requirements-grpc.txt"
        ) from e

    repo_root = _workspace_root()
    proto = _proto_path()
    out_dir = Path(__file__).resolve().parent

    # Đảm bảo import path có repo root để `import grpc_api.*` hoạt động.
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    args = [
        "grpc_tools.protoc",
        f"-I{proto.parent}",
        f"--python_out={out_dir}",
        f"--grpc_python_out={out_dir}",
        str(proto),
    ]

    rc = protoc.main(args)
    if rc != 0:  # pragma: no cover
        raise RuntimeError(f"Generate proto failed, exit code={rc}")

    # Trên một số môi trường, file có thể được generate nhưng timestamp chưa flush ngay.
    os.sync() if hasattr(os, "sync") else None


def load_generated() -> Tuple[object, object]:
    """
    Trả về tuple (pb2, pb2_grpc). Nếu chưa có file generate thì tự generate.
    """
    if not _generated_ok():
        generate_proto()

    pb2 = importlib.import_module("grpc_api.translategemma_pb2")
    # Các file generate mặc định dùng `import translategemma_pb2 ...` (không prefix package),
    # nên cần đăng ký alias này trong sys.modules trước khi import *_pb2_grpc.
    sys.modules.setdefault("translategemma_pb2", pb2)

    pb2_grpc = importlib.import_module("grpc_api.translategemma_pb2_grpc")
    sys.modules.setdefault("translategemma_pb2_grpc", pb2_grpc)
    return pb2, pb2_grpc

