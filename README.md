## Translate-Gemma API

`translate-gemma` là một dịch vụ **dịch đa ngôn ngữ** (text và image-to-text) xây dựng trên mô hình **Google TranslateGemma** từ thư viện `transformers`.  
Dự án cung cấp:

- **API FastAPI** dùng model `google/translategemma-4b-it` (tối ưu cho phục vụ qua HTTP, có batching).
- **Script ví dụ** dùng model `google/translategemma-12b-it` để chạy dịch nhanh trên máy local.

---

### 1. Tính năng chính

- **Dịch văn bản** giữa nhiều ngôn ngữ với `TranslateGemma`.
- **Trích xuất + dịch từ ảnh (URL)**: input là URL ảnh, model sẽ đọc text trên ảnh và dịch sang ngôn ngữ đích.
- **Batching request text**: nhiều request `/translate/text` được gom lại thành 1 lần gọi GPU để tận dụng tài nguyên.
- **Triển khai dễ dàng** bằng Docker + `docker-compose` với hỗ trợ GPU (NVIDIA).

---

### 2. Cấu trúc chính

Các file quan trọng:

- `main.py`  
  - Khởi tạo model `google/translategemma-4b-it` với `AutoModelForImageTextToText` và `AutoProcessor`.
  - Định nghĩa API FastAPI:
    - `POST /translate/text`
    - `POST /translate/image`
  - Cơ chế batching dựa trên `asyncio.Queue` + `ThreadPoolExecutor`.

- `translate_gemma.py`  
  - Ví dụ sử dụng model `google/translategemma-12b-it` cho:
    - Dịch text.
    - Trích xuất + dịch từ ảnh URL.
  - Chạy trực tiếp bằng `python translate_gemma.py`.

- `requirements.txt`  
  - Danh sách dependency: `transformers`, `accelerate`, `torch`, `fastapi`, `uvicorn`, …

- `Dockerfile`  
  - Image runtime CUDA 12.3 + Python.
  - Cài `requirements.txt`, copy `main.py`, expose cổng 8000.
  - Dùng `uvicorn main:app` để chạy server.

- `docker-compose.yml`  
  - Service `translategemma-api`.
  - Mount thư mục `./models` để cache model (`HF_HOME`, `TRANSFORMERS_CACHE`, `TORCH_HOME`).
  - Yêu cầu GPU thông qua `deploy.resources.reservations.devices`.

---

### 3. Cài đặt & chạy local (không Docker)

#### 3.1. Yêu cầu

- Python 3.9+  
- GPU NVIDIA + CUDA (khuyến nghị, nhưng có thể chạy CPU chậm hơn).  
- Tài khoản Hugging Face (nếu mô hình yêu cầu token), và biến môi trường `HF_TOKEN` nếu cần.

#### 3.2. Cài đặt

cd translate-gemma

python3 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
