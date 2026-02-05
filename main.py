import torch
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
from transformers import AutoModelForImageTextToText, AutoProcessor

# ================= CONFIG & MODEL =================
MODEL_ID = "google/translategemma-4b-it"

print("Loading processor & model...")
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_ID,
    device_map="auto",
    torch_dtype=torch.bfloat16,  
)
print("Model loaded.")

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


def _run_model(messages):
    """Hàm dùng chung để gọi model giống trong translate_gemma.py."""
    inputs = processor.apply_chat_template(
    messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
).to(model.device, dtype=torch.bfloat16)
    input_len = len(inputs["input_ids"][0])

    with torch.inference_mode():
        generation = model.generate(**inputs, do_sample=False, max_new_tokens=1024, )

    generation = generation[0][input_len:]
    decoded = processor.decode(generation, skip_special_tokens=True)
    return decoded


@app.post("/translate/text")
async def translate_text(req: TextTranslateRequest):
    """Dịch đoạn text theo đúng format của Translategemma."""
    messages = [
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

    translation = _run_model(messages)
    return {"translation": translation}


@app.post("/translate/image")
async def translate_image(req: ImageTranslateRequest):
    """Trích xuất + dịch text từ ảnh (URL) theo đúng format của Translategemma."""
    messages = [
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

    translation = _run_model(messages)
    return {"translation": translation}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000)