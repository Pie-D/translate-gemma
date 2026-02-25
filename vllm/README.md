# TranslateGemma 4B với vLLM

Thư mục này chạy **google/translategemma-4b-it** bằng [vLLM](https://docs.vllm.ai/) (OpenAI-compatible API), không dùng code hay file từ thư mục gốc.

**Lưu ý:** vLLM hỗ trợ TranslateGemma có thể cần `--trust-remote-code` hoặc phiên bản mới. Nếu khi chạy engine báo lỗi không nhận diện model, hãy cập nhật vLLM lên bản mới nhất hoặc theo dõi [vLLM #32446](https://github.com/vllm-project/vllm/issues/32446).

---

## Cấu trúc

- **app.py** – FastAPI wrapper: nhận `POST /translate/text`, build prompt bằng processor (transformers), gọi vLLM backend, trả về bản dịch.
- **start_engine.sh** – Script khởi động vLLM server (model server).
- **Dockerfile** – Image chạy vLLM engine.
- **Dockerfile.app** – Image chạy FastAPI app.
- **docker-compose.yml** – Chạy cả engine + app.

---

## Cách chạy

### 1. Chạy trực tiếp (2 terminal)

**Terminal 1 – vLLM engine (port 8000):**
```bash
cd vllm
pip install -r requirements.txt
chmod +x start_engine.sh
./start_engine.sh
```

**Terminal 2 – FastAPI app (port 8001):**
```bash
cd vllm
export VLLM_BASE_URL=http://localhost:8000/v1
uvicorn app:app --host 0.0.0.0 --port 8001
```

Gọi API dịch:
```bash
curl -X POST http://localhost:8001/translate/text \
  -H "Content-Type: application/json" \
  -d '{"source_lang_code":"en","target_lang_code":"vi","text":"Hello world."}'
```

### 2. Chạy bằng Docker Compose

```bash
cd vllm
docker compose up --build
```

- vLLM engine: `http://localhost:8000`
- API dịch (giống API cũ): `http://localhost:8001/translate/text`

Ví dụ:
```bash
curl -X POST http://localhost:8001/translate/text \
  -H "Content-Type: application/json" \
  -d '{"source_lang_code":"en","target_lang_code":"vi","text":"Hello world."}'
```

---

## Biến môi trường

| Biến | Mặc định | Mô tả |
|------|----------|--------|
| `VLLM_BASE_URL` | `http://localhost:8000/v1` | URL API vLLM (dùng bởi app.py) |
| `VLLM_PORT` | `8000` | Port vLLM (dùng bởi start_engine.sh) |
| `MODEL_ID` | `google/translategemma-4b-it` | Model Hugging Face (start_engine.sh) |

---

## So với bản Transformers

- **vLLM:** PagedAttention, continuous batching, kernel tối ưu → thường tăng throughput và có thể giảm latency khi có nhiều request.
- **Transformers (main.py / flash-attention):** Đơn giản, ít phụ thuộc; phù hợp khi ít request hoặc ưu tiên ổn định.

Nếu vLLM báo lỗi không load được `google/translategemma-4b-it`, tạm thời dùng bản Transformers (hoặc flash-attention) cho đến khi vLLM hỗ trợ chính thức.
