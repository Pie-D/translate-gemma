## gRPC cho translate-gemma (không sửa FastAPI hiện có)

Thư mục `grpc_api/` cung cấp server gRPC dùng lại **model + batching queue** từ `main.py` thông qua `import main`.

### 1. Cài dependency cho gRPC

```bash
cd /home/nvhung/translate-gemma
python -m pip install -r requirements.txt
python -m pip install -r requirements-grpc.txt
```

### 2. Chạy gRPC server

```bash
cd /home/nvhung/translate-gemma
python -m grpc_api.server --host 0.0.0.0 --port 50051
```

Lần chạy đầu, server sẽ tự generate các file stub:

- `grpc_api/translategemma_pb2.py`
- `grpc_api/translategemma_pb2_grpc.py`

…từ `grpc_api/translategemma.proto` (cần `grpcio-tools`).

### 3. Test nhanh bằng client example

```bash
cd /home/nvhung/translate-gemma
python -m grpc_api.client_example --target 127.0.0.1:50051 --source en --dest vi --text "Hello world"
```

Test dịch ảnh (URL):

```bash
python -m grpc_api.client_example --target 127.0.0.1:50051 --source en --dest vi --image-url "https://example.com/a.png"
```

### 4. Notes quan trọng

- gRPC server **không** chạy FastAPI/uvicorn; nó import `main.py` để dùng lại:
  - `request_queue`
  - `_gpu_worker()` (batching)
  - `TextTranslateRequest`, `ImageTranslateRequest`
- Import `main.py` sẽ **load model ngay** khi khởi động gRPC server (giống FastAPI).
- Nếu bạn chạy FastAPI và gRPC ở **2 process khác nhau**, model sẽ được load **2 lần** (tốn VRAM). Muốn dùng chung 1 model thì phải dùng 1 process hoặc tách inference thành service riêng.

