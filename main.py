import asyncio
import torch
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Any, List, Tuple, Union
from transformers import AutoModelForImageTextToText, AutoProcessor

# ================= CONFIG =================
MODEL_ID = "google/translategemma-4b-it"
BATCH_SIZE = 32
BATCH_TIMEOUT = 0.01   # giảm timeout để giảm latency
MAX_NEW_TOKENS = 256   # giảm latency rất nhiều

print("Loading processor & model...")
processor = AutoProcessor.from_pretrained(MODEL_ID)

model = AutoModelForImageTextToText.from_pretrained(
    MODEL_ID,
    device_map="cuda:0",
    torch_dtype=torch.bfloat16,
    attn_implementation="sdpa",   # ổn định hơn flash
)

# compile giảm overhead forward
model = torch.compile(model, mode="reduce-overhead")

if getattr(model.config, "pad_token_id", None) is None:
    model.config.pad_token_id = model.config.eos_token_id

print("Model loaded.")

# ================= FASTAPI =================
app = FastAPI()

request_queue: asyncio.Queue = asyncio.Queue()


# ================= REQUEST MODELS =================
class TextTranslateRequest(BaseModel):
    source_lang_code: str
    target_lang_code: str
    text: str


# ================= TOKENIZE SỚM (GIẢM TTFB) =================
def _build_messages(req: TextTranslateRequest) -> list:
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


def _tokenize(messages):
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    return inputs


# ================= GPU FORWARD =================
def _run_batch(tokenized_list):
    if len(tokenized_list) == 1:
        inp = tokenized_list[0]
        for k, v in inp.items():
            inp[k] = v.to(model.device)

        input_len = inp["input_ids"].shape[1]

        with torch.inference_mode():
            out = model.generate(
                **inp,
                do_sample=False,
                num_beams=1,
                use_cache=True,
                max_new_tokens=MAX_NEW_TOKENS,
                pad_token_id=model.config.pad_token_id,
            )

        return [
            processor.decode(
                out[0][input_len:], skip_special_tokens=True
            )
        ]

    # pad_id = processor.tokenizer.pad_token_id or model.config.eos_token_id
    pad_id = processor.tokenizer.pad_token_id
    if pad_id is None:
        pad_id = model.config.eos_token_id
    if pad_id is None:
        pad_id = 1

    input_ids = torch.nn.utils.rnn.pad_sequence(
        [e["input_ids"][0] for e in tokenized_list],
        batch_first=True,
        padding_value=pad_id,
    ).to(model.device)

    attention_mask = torch.nn.utils.rnn.pad_sequence(
        [e["attention_mask"][0] for e in tokenized_list],
        batch_first=True,
        padding_value=0,
    ).to(model.device)

    input_lens = [e["input_ids"].shape[1] for e in tokenized_list]

    with torch.inference_mode():
        out = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            do_sample=False,
            num_beams=1,
            use_cache=True,
            max_new_tokens=MAX_NEW_TOKENS,
            pad_token_id=pad_id,
        )

    results = []
    for i in range(len(tokenized_list)):
        new_tokens = out[i, input_lens[i]:]
        results.append(
            processor.decode(new_tokens, skip_special_tokens=True)
        )

    return results


# ================= GPU WORKER =================
async def _gpu_worker():
    loop = asyncio.get_event_loop()

    while True:
        batch: List[Tuple[Any, asyncio.Future]] = []

        try:
            while len(batch) < BATCH_SIZE:
                item = await asyncio.wait_for(
                    request_queue.get(),
                    timeout=BATCH_TIMEOUT
                )
                batch.append(item)
        except asyncio.TimeoutError:
            pass

        if not batch:
            continue

        tokenized_list = [t for t, _ in batch]

        try:
            # 🔥 chạy trực tiếp — KHÔNG dùng ThreadPoolExecutor
            translations = _run_batch(tokenized_list)

            for (_, fut), tr in zip(batch, translations):
                if not fut.cancelled():
                    fut.set_result({"translation": tr})

        except Exception as e:
            for _, fut in batch:
                if not fut.cancelled():
                    fut.set_exception(e)


# ================= API =================
@app.post("/translate/text")
async def translate_text(req: TextTranslateRequest):
    loop = asyncio.get_event_loop()
    fut = loop.create_future()

    messages = _build_messages(req)
    tokenized = _tokenize(messages)   # tokenize trước → giảm TTFB

    await request_queue.put((tokenized, fut))
    return await fut


@app.on_event("startup")
async def startup_event():
    asyncio.create_task(_gpu_worker())


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000)
