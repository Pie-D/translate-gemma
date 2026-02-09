import asyncio
import torch
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional, List, Tuple, Any, Union
from transformers import AutoModelForImageTextToText, AutoProcessor

# ================= CONFIG & MODEL =================
MODEL_ID = "google/translategemma-4b-it"
BATCH_SIZE = 16
BATCH_TIMEOUT = 0.01  # giây – gom request trước khi xử lý
MAX_NEW_TOKENS = 512

print("Loading processor & model...")
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_ID,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)
# Tránh warning open-end generation
if getattr(model.config, "pad_token_id", None) is None:
    model.config.pad_token_id = model.config.eos_token_id
print("Model loaded.")

# ================= BATCH QUEUE =================
request_queue: asyncio.Queue = asyncio.Queue()
_executor = ThreadPoolExecutor(max_workers=1)

# ================= FASTAPI APP =================
app = FastAPI(title="Translategemma FastAPI")


class TextTranslateRequest(BaseModel):
    source_lang_code: str
    target_lang_code: str
    text: str


class ImageTranslateRequest(BaseModel):
    source_lang_code: str
    target_lang_code: str
    url: str  # URL ảnh đầu vào


def _messages_text(req: TextTranslateRequest) -> list:
    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "source_lang_code": req.source_lang_code,
                    "target_lang_code": req.target_lang_code,
                    "text": req.text,
                }
            ],
        }
    ]


def _messages_image(req: ImageTranslateRequest) -> list:
    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source_lang_code": req.source_lang_code,
                    "target_lang_code": req.target_lang_code,
                    "url": req.url,
                }
            ],
        }
    ]


def _run_model_single(messages: list) -> str:
    """Một request (dùng cho ảnh hoặc single)."""
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    for k, v in inputs.items():
        if hasattr(v, "to"):
            if k in ("input_ids", "attention_mask"):
                inputs[k] = v.to(model.device)
            else:
                inputs[k] = v.to(model.device, dtype=torch.bfloat16)
    input_len = inputs["input_ids"].shape[1]
    with torch.inference_mode():
        out = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=MAX_NEW_TOKENS,
            pad_token_id=model.config.pad_token_id,
        )
    decoded = processor.decode(out[0][input_len:], skip_special_tokens=True)
    return decoded


def _run_model_batch_text(messages_list: List[list]) -> List[str]:
    """Gom nhiều request text thành một lần generate."""
    if not messages_list:
        return []
    if len(messages_list) == 1:
        return [_run_model_single(messages_list[0])]

    encoded = []
    input_lens = []
    for messages in messages_list:
        inp = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        encoded.append(inp)
        input_lens.append(inp["input_ids"].shape[1])

    pad_id = getattr(
        processor.tokenizer,
        "pad_token_id",
        model.config.eos_token_id,
    )
    if pad_id is None:
        pad_id = model.config.eos_token_id

    input_ids = torch.nn.utils.rnn.pad_sequence(
        [e["input_ids"][0] for e in encoded],
        batch_first=True,
        padding_value=pad_id,
    ).to(model.device)
    if "attention_mask" in encoded[0]:
        attention_mask = torch.nn.utils.rnn.pad_sequence(
            [e["attention_mask"][0] for e in encoded],
            batch_first=True,
            padding_value=0,
        ).to(model.device)
    else:
        attention_mask = (input_ids != pad_id).long()

    inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
    with torch.inference_mode():
        out = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=MAX_NEW_TOKENS,
            pad_token_id=pad_id,
        )

    results = []
    for i in range(len(messages_list)):
        new_tokens = out[i, input_lens[i] :]
        results.append(
            processor.decode(new_tokens, skip_special_tokens=True)
        )
    return results


def _process_batch_sync(
    batch: List[Tuple[str, Any, asyncio.Future]],
) -> List[Tuple[asyncio.Future, Union[dict, Exception]]]:
    """Chạy trong executor: xử lý batch, trả về [(future, result|exception)] để set ở main thread."""
    out: List[Tuple[asyncio.Future, Union[dict, Exception]]] = []
    text_items = [(i, m, f) for i, (k, m, f) in enumerate(batch) if k == "text"]
    image_items = [(i, m, f) for i, (k, m, f) in enumerate(batch) if k == "image"]

    if text_items:
        messages_list = [_messages_text(m) for (_, m, _) in text_items]
        try:
            translations = _run_model_batch_text(messages_list)
            for (_, _, fut), tr in zip(text_items, translations):
                out.append((fut, {"translation": tr}))
        except Exception as e:
            for _, _, fut in text_items:
                out.append((fut, e))

    for _, req, fut in image_items:
        try:
            messages = _messages_image(req)
            tr = _run_model_single(messages)
            out.append((fut, {"translation": tr}))
        except Exception as e:
            out.append((fut, e))
    return out


async def _gpu_worker() -> None:
    """Worker: gom request từ queue, xử lý batch rồi trả kết quả."""
    loop = asyncio.get_event_loop()
    while True:
        batch: List[Tuple[str, Any, asyncio.Future]] = []
        try:
            while len(batch) < BATCH_SIZE:
                item = await asyncio.wait_for(
                    request_queue.get(), timeout=BATCH_TIMEOUT
                )
                batch.append(item)
        except asyncio.TimeoutError:
            pass
        if not batch:
            continue
        results = await loop.run_in_executor(
            _executor,
            _process_batch_sync,
            batch,
        )
        for fut, res in results:
            if isinstance(res, Exception):
                fut.set_exception(res)
            else:
                fut.set_result(res)


@app.post("/translate/text")
async def translate_text(req: TextTranslateRequest):
    """Dịch đoạn text – request được gom batch với các request khác."""
    loop = asyncio.get_event_loop()
    fut = loop.create_future()
    await request_queue.put(("text", req, fut))
    return await fut


@app.post("/translate/image")
async def translate_image(req: ImageTranslateRequest):
    """Trích xuất + dịch từ ảnh (URL) – xử lý tuần tự trong worker."""
    loop = asyncio.get_event_loop()
    fut = loop.create_future()
    await request_queue.put(("image", req, fut))
    return await fut


@app.on_event("startup")
async def startup_event():
    asyncio.create_task(_gpu_worker())


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000)