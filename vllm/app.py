"""
FastAPI wrapper gọi vLLM OpenAI-compatible API để dịch.
Dùng processor (transformers) chỉ để build đúng chat template của TranslateGemma.
"""
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import OpenAI
from transformers import AutoProcessor

# ================= CONFIG =================
MODEL_ID = "google/translategemma-4b-it"
VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
MAX_TOKENS = 256

print("Loading processor for chat template...")
processor = AutoProcessor.from_pretrained(MODEL_ID)
print("Processor loaded.")

app = FastAPI(title="TranslateGemma vLLM API")
client = OpenAI(base_url=VLLM_BASE_URL, api_key="dummy")


# ================= REQUEST MODEL =================
class TextTranslateRequest(BaseModel):
    source_lang_code: str
    target_lang_code: str
    text: str


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


def _get_prompt(req: TextTranslateRequest) -> str:
    """Chuỗi prompt đúng format TranslateGemma (từ chat template)."""
    messages = _build_messages(req)
    prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    return prompt


@app.post("/translate/text")
async def translate_text(req: TextTranslateRequest):
    prompt = _get_prompt(req)
    try:
        resp = client.chat.completions.create(
            model=MODEL_ID,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=MAX_TOKENS,
            temperature=0.0,
        )
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"vLLM backend error: {e!s}")
    choice = resp.choices[0]
    text = (choice.message.content or "").strip()
    return {"translation": text}


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8001)
