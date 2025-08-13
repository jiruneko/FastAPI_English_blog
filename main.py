# main.py
from __future__ import annotations
from typing import Literal

import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from openai import OpenAI, APIConnectionError, RateLimitError, APIStatusError

load_dotenv()

class GrammarReq(BaseModel):
    text: str
    locale: Literal["en", "ja"] = "en"

class GrammarRes(BaseModel):
    before: str
    after: str

app = FastAPI(title="AI Grammar API (LLM)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000", "http://localhost:3001",
        "http://127.0.0.1:3000", "http://127.0.0.1:3001",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
TIMEOUT = int(os.environ.get("OPENAI_TIMEOUT_SECONDS", "20"))

SYSTEM_PROMPT = (
    "You are a professional English copy editor.\n"
    "- Correct grammar, article usage, agreement, and naturalness.\n"
    "- Preserve meaning and style.\n"
    "- Do NOT paraphrase more than necessary.\n"
    "- Return only the corrected text, no explanations.\n"
)

def _truncate_for_tokens(text: str, max_chars: int = 8000) -> str:
    # 超長文でも落ちないようにざっくり長さ制限（必要なら分割実装へ拡張）
    return text[:max_chars]

@retry(
    reraise=True,
    stop=stop_after_attempt(2),
    wait=wait_exponential(multiplier=0.5, min=0.5, max=2),
    retry=retry_if_exception_type((APIConnectionError, RateLimitError, APIStatusError)),
)
def _call_llm_correct(text: str) -> str:
    # Responses API でも Chat Completions でもOK。安定の chat.completions を使用。
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _truncate_for_tokens(text)},
        ],
        temperature=0,
        timeout=TIMEOUT,
    )
    content = resp.choices[0].message.content or ""
    return content.strip()

@app.get("/healthz")
def healthz():
    return {"ok": True, "model": MODEL}

@app.post("/grammar", response_model=GrammarRes)
def grammar(req: GrammarReq):
    source = req.text.strip()
    if not source:
        raise HTTPException(status_code=422, detail="text is empty")

    try:
        corrected = _call_llm_correct(source)
    except (APIConnectionError, RateLimitError, APIStatusError) as e:
        raise HTTPException(status_code=502, detail=f"LLM upstream error: {e}") from e
    except Exception as e:  # 予期しない例外は 500
        raise HTTPException(status_code=500, detail="LLM call failed") from e

    # LLMが空返しをするケースは稀だが、保険として原文を返す
    if not corrected:
        corrected = source

    return {"before": source, "after": corrected}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
