import os
import json
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

# Load env
load_dotenv()
API_URL = os.getenv("API_URL")
API_KEY = os.getenv("API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME")
app = FastAPI(title="The Japanese Writing Game")

origins = [
    "http://127.0.0.1:5501",
    "http://localhost:5501",
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
    api_key=API_KEY,  # type: ignore
    model=MODEL_NAME,  # pyright: ignore[reportArgumentType]
    temperature=0,
)


@app.post("/grade-kanji")
async def grade_kanji(data: dict):
    """Grade student's kanji"""
    target_word = data.get("target_word")
    image_b64 = data.get("image")

    if not target_word or not image_b64:
        raise HTTPException(status_code=400, detail="Missing word or image")

    message = HumanMessage(
        content=[
            {
                "type": "text",
                "text": f"""
                You are a friendly and lenient Kanji handwriting judge.

                Target English word: {target_word}

                1. Convert the English target word into its correct Kanji.
                2. Compare the student's handwritten Kanji drawing from the image.

                ### VERY IMPORTANT JUDGING RULES:
                - Do NOT judge based on stroke thickness.
                - Do NOT judge based on perfect proportions.
                - Small distortions or messy handwriting are acceptable.
                - As long as the overall *shape* and *structure* look close to the correct Kanji, mark it as correct.
                - Only mark incorrect if the drawing is clearly NOT the intended Kanji.

                ### What counts as correct:
                - Rough shape is correct  
                - Strokes are present but slightly misplaced  
                - Shape resembles the correct kanji even if messy  
                - The kanji is rotated slightly or uneven — still acceptable  

                ### Respond ONLY in strict JSON (no explanations outside JSON):
                {{
                "correct": true/false,
                "kanji_correct": "THE_CORRECT_KANJI",
                "kanji_detected": "KANJI_DETECTED_FROM_IMAGE",
                "pronouce_in_romanji": "Pronunciation in romanji, easy for user to say it",
                "pronouce_in_hiragana": "Pronunciation in hiragana, easy for user to say it",
                "reason": "Very short, friendly explanation"
                }}
                """,
            },
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{image_b64}"},
            },
        ]
    )

    result = llm.invoke([message])
    raw = result.content.strip()  # type: ignore

    try:
        parsed = json.loads(raw)
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail="Model reutrned invalid JSON"
        ) from exc

    return {
        "correct": parsed.get("correct"),
        "kanji_detected": parsed.get("kanji_detected"),
        "kanji_correct": parsed.get("kanji_correct"),
        "pronouce_in_romanji": parsed.get("pronouce_in_romanji"),
        "pronouce_in_hiragana": parsed.get("pronouce_in_hiragana"),
        "similarity": parsed.get("similarity"),
        "reason": parsed.get("reason"),
    }


@app.post("/grade-katakana")
async def grade_katakana(data: dict):
    """Grade student's katakana"""
    target_word = data.get("target_word")
    image_b64 = data.get("image")

    if not target_word or not image_b64:
        raise HTTPException(status_code=400, detail="Missing word")

    message = HumanMessage(
        content=[
            {
                "type": "text",
                "text": f"""
                You are a friendly and lenient Katakana handwriting judge.

                Target English word: {target_word}

                1. Convert the English target word into its correct Katakana.
                2. Compare the student's handwritten Katakana drawing from the image.

                ### VERY IMPORTANT JUDGING RULES:
                - Do NOT judge based on stroke thickness.
                - Do NOT judge based on perfect proportions.
                - Small distortions or messy handwriting are acceptable.
                - As long as the overall *shape* and *structure* look close to the correct Katakana, mark it as correct.
                - Only mark incorrect if the drawing is clearly NOT the intended Katakana.

                ### What counts as correct:
                - Rough shape is correct  
                - Strokes are present but slightly misplaced  
                - Shape resembles the correct katakana even if messy  
                - The katakana is rotated slightly or uneven — still acceptable  

                ### Respond ONLY in strict JSON (no explanations outside JSON):
                {{
                "correct": true/false,
                "katakana_correct": "THE_CORRECT_KATAKANA",
                "katakana_detected": "KATAKANA_DETECTED_FROM_IMAGE",
                "reason": "Very short, friendly explanation"
                }}
                """,
            },
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{image_b64}"},
            },
        ]
    )

    result = llm.invoke([message])
    raw = result.content.strip()  # type: ignore

    try:
        parsed = json.loads(raw)
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail="Model reutrned invalid JSON"
        ) from exc

    return {
        "correct": parsed.get("correct"),
        "katakana_detected": parsed.get("katakana_detected"),
        "katakana_correct": parsed.get("katakana_correct"),
        "similarity": parsed.get("similarity"),
        "reason": parsed.get("reason"),
    }


@app.post("/grade-hiragana")
async def grade_hiragana(data: dict):
    """Grade student's hiragana"""
    target_word = data.get("target_word")
    image_b64 = data.get("image")

    if not target_word or not image_b64:
        raise HTTPException(status_code=400, detail="Missing word")

    message = HumanMessage(
        content=[
            {
                "type": "text",
                "text": f"""
                You are a friendly and lenient Hiragana handwriting judge.

                Target English word: {target_word}

                1. Convert the English target word into its correct Hiragana.
                2. Compare the student's handwritten Hiragana drawing from the image.

                ### VERY IMPORTANT JUDGING RULES:
                - Do NOT judge based on stroke thickness.
                - Do NOT judge based on perfect proportions.
                - Small distortions or messy handwriting are acceptable.
                - As long as the overall *shape* and *structure* look close to the correct Hiragana, mark it as correct.
                - Only mark incorrect if the drawing is clearly NOT the intended Hiragana.

                ### What counts as correct:
                - Rough shape is correct  
                - Strokes are present but slightly misplaced  
                - Shape resembles the correct hiragana even if messy  
                - The hiragana is rotated slightly or uneven — still acceptable  

                ### Respond ONLY in strict JSON (no explanations outside JSON):
                {{
                "correct": true/false,
                "hiragana_correct": "THE_CORRECT_HIRAGANA",
                "hiragana_detected": "HIRAGANA_DETECTED_FROM_IMAGE",
                "reason": "Very short, friendly explanation"
                }}
                """,
            },
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{image_b64}"},
            },
        ]
    )

    result = llm.invoke([message])
    raw = result.content.strip()  # type: ignore

    try:
        parsed = json.loads(raw)
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail="Model reutrned invalid JSON"
        ) from exc

    return {
        "correct": parsed.get("correct"),
        "hiragana_detected": parsed.get("hiragana_detected"),
        "hiragana_correct": parsed.get("hiragana_correct"),
        "similarity": parsed.get("similarity"),
        "reason": parsed.get("reason"),
    }
