from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
import base64
import mimetypes
import os
from dotenv import load_dotenv
import requests
from typing import Optional
from io import BytesIO
from PIL import Image

# Load env
load_dotenv()
API_URL = os.getenv("API_URL")
API_KEY = os.getenv("API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME")

if not API_URL or not API_KEY or not MODEL_NAME:
    raise RuntimeError("Please set API_URL, API_KEY and MODEL_NAME in your .env file")

# Endpoints
CHAT_COMPLETIONS_PATH = "/chat/completions"  # vLLM / OpenAI-compatible

app = FastAPI(title="Imagenizer - FastAPI backend for Qwen3-VL (vLLM)")

# Allow your frontend origin (or * for all origins)
origins = [
    "http://127.0.0.1:5500",
    "http://localhost:5500",
    "http://localhost:6767",  # if needed
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # or ["*"] to allow all origins (simpler for dev)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
llm = ChatOpenAI(
    base_url=API_URL,
    api_key=API_KEY,
    model=MODEL_NAME,
    temperature=0,
)


class Base64ImageIn(BaseModel):
    image_base64: str


class ImageUrlIn(BaseModel):
    image_url: str


def image_bytes_to_data_url(bytes_data: bytes, filename: Optional[str] = None) -> str:
    # Try to determine mime type from filename, fallback to webp/jpeg
    mime = None
    if filename:
        mime, _ = mimetypes.guess_type(filename)
    if not mime:
        # try to detect via PIL
        try:
            img = Image.open(BytesIO(bytes_data))
            fmt = img.format.lower()
            mime = f"image/{fmt}"
        except Exception:
            mime = "image/jpeg"
    b64 = base64.b64encode(bytes_data).decode()
    return f"data:{mime};base64,{b64}"


def call_vllm_chat_completion(
    data_url: str, prompt_text: str, temperature: float = 0.0
) -> dict:
    """Send a single-message multimodal chat request to the vLLM endpoint.
    Uses the OpenAI-compatible payload with an `image_url` message component.
    """
    api_endpoint = API_URL.rstrip("/") + CHAT_COMPLETIONS_PATH
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_url}},
                    {"type": "text", "text": "Classify this image."},
                ],
            }
        ],
        "temperature": 0,
    }

    resp = requests.post(api_endpoint, json=payload, headers=headers, timeout=60)
    if resp.status_code != 200:
        raise HTTPException(
            status_code=502, detail={"status_code": resp.status_code, "body": resp.text}
        )
    return resp.json()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/grade-kanji")
async def grade_kanji(data: dict):
    target_word = data.get("target_word")
    image_b64 = data.get("image")

    if not target_word or not image_b64:
        raise HTTPException(status_code=400, detail="Missing word or image")

    image_part = {
        "type": "image_url",
        "image_url": {"url": f"data:image/png;base64,{image_b64}"},
    }

    text_part = {
        "type": "text",
        "text": f"""
You are a Kanji handwriting judge.
The target word is: {target_word}

Please review the student's handwritten Kanji and answer in JSON:

{{
  "correct": true/false,
  "kanji_detected": "KANJI",
  "reason": "brief explanation"
}}
""",
    }

    response = llm.invoke([text_part, image_part])
    return {"result": response}


@app.post("/classify")
async def classify_image(
    file: Optional[UploadFile] = File(None), payload: Optional[Base64ImageIn] = None
):
    """Accepts either a multipart file upload (form) or JSON body {"image_base64": "..."}
    Returns the raw vLLM response and attempts to extract a single-line label.
    """
    if file is None and payload is None:
        raise HTTPException(
            status_code=400,
            detail="Provide either an uploaded file or JSON with image_base64",
        )

    # Get bytes
    if file is not None:
        contents = await file.read()
        filename = file.filename
    else:
        # payload provided
        try:
            contents = base64.b64decode(payload.image_base64)
            filename = None
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid base64 payload")

    data_url = image_bytes_to_data_url(contents, filename)

    # Prompt - classification
    prompt = "Classify this image in ONE short label (single word or short phrase). Return only the label."

    try:
        vllm_resp = call_vllm_chat_completion(
            data_url=data_url, prompt_text=prompt, temperature=0.0
        )
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # Extract text content heuristically from the vLLM response
    label = None
    try:
        choices = vllm_resp.get("choices")
        if choices and len(choices) > 0:
            # Many OpenAI-compatible backends put the assistant message under choices[0].message.content
            msg = choices[0].get("message") or choices[0]
            # The model's content may be a string or structured
            content = msg.get("content") if isinstance(msg, dict) else None
            if isinstance(content, str):
                label = content.strip().strip('"')
            elif isinstance(content, list):
                # sometimes content is a list of components
                # find the first text component
                for c in content:
                    if c.get("type") in ("output_text", "text"):
                        label = c.get("text")
                        break
            else:
                # fallback to stringifying
                label = str(msg)
    except Exception:
        label = None

    return JSONResponse({"vllm_raw": vllm_resp, "label": label})


@app.post("/classify_url")
async def classify_url(body: ImageUrlIn):
    # simple wrapper that forwards a public URL as image_url component
    data_url = body.image_url
    prompt = "Classify this image in ONE short label (single word or short phrase). Return only the label."
    try:
        vllm_resp = call_vllm_chat_completion(
            data_url=data_url, prompt_text=prompt, temperature=0.0
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # extract label similar to /classify
    label = None
    try:
        choices = vllm_resp.get("choices")
        if choices and len(choices) > 0:
            msg = choices[0].get("message") or choices[0]
            content = msg.get("content") if isinstance(msg, dict) else None
            if isinstance(content, str):
                label = content.strip().strip('"')
            elif isinstance(content, list):
                for c in content:
                    if c.get("type") in ("output_text", "text"):
                        label = c.get("text")
                        break
            else:
                label = str(msg)
    except Exception:
        label = None

    return JSONResponse({"vllm_raw": vllm_resp, "label": label})
