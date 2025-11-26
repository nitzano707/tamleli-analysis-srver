import uuid
import json
import asyncio
import re
import logging
from typing import Union, Optional

from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx

# ─────────────────────────────────────────────
# לוגים
# ─────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Qualitative Agent – Multi-Model")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# JOB STORE
# ============================================================
JOBS = {}


def create_job(job_id):
    JOBS[job_id] = {
        "status": "running",
        "progress": 0,
        "result": None,
        "error": None
    }


def update_progress(job_id, v):
    if job_id in JOBS:
        JOBS[job_id]["progress"] = int(v)


def set_result(job_id, r):
    JOBS[job_id]["status"] = "done"
    JOBS[job_id]["progress"] = 100
    JOBS[job_id]["result"] = r


def set_error(job_id, err):
    JOBS[job_id]["status"] = "error"
    JOBS[job_id]["error"] = str(err)


# ============================================================
# REQUEST MODEL
# ============================================================
class AnalysisRequest(BaseModel):
    transcript: dict | list
    research_context: dict = {}
    model: str = "gemini"  # gpt / claude / gemini
    api_key: str | None = None


# ============================================================
# JSON EXTRACTION HELPER
# ============================================================
def extract_json(raw_text: str):
    """מחלץ JSON מתגובת המודל, גם אם עטופה ב-markdown"""
    text = raw_text.strip()
    
    # הסרת markdown code blocks
    code_block_pattern = r'```(?:json|JSON)?\s*\n?([\s\S]*?)\n?```'
    match = re.search(code_block_pattern, text)
    if match:
        text = match.group(1).strip()
    
    # מציאת תחילת JSON
    if not text.startswith('{') and not text.startswith('['):
        for i, char in enumerate(text):
            if char in '{[':
                text = text[i:]
                break
    
    try:
        return json.loads(text)
    except:
        logger.error(f"Failed to parse JSON: {text[:300]}")
        return None


# ============================================================
# AI PROVIDERS
# ============================================================
async def call_gpt(prompt: str, api_key: str) -> str:
    headers = {"Authorization": f"Bearer {api_key}"}
    body = {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,
    }
    async with httpx.AsyncClient(timeout=120) as client:
        r = await client.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=body,
        )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


async def call_claude(prompt: str, api_key: str) -> str:
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    body = {
        "model": "claude-sonnet-4-20250514",
        "max_tokens": 4000,
        "messages": [{"role": "user", "content": prompt}],
    }
    async with httpx.AsyncClient(timeout=120) as client:
        r = await client.post(
            "https://api.anthropic.com/v1/messages",
            headers=headers,
            json=body,
        )
    r.raise_for_status()
    return r.json()["content"][0]["text"]


async def call_gemini(prompt: str, api_key: str) -> str:
    url = (
        "https://generativelanguage.googleapis.com/"
        "v1beta/models/gemini-1.5-pro:generateContent"
        f"?key={api_key}"
    )
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.3}
    }
    async with httpx.AsyncClient(timeout=120) as client:
        r = await client.post(url, json=payload)
    r.raise_for_status()
    return r.json()["candidates"][0]["content"]["parts"][0]["text"]


async def model_call(prompt: str, model: str, api_key: str) -> str:
    if not api_key:
        raise Exception("חסר API Key")
    
    if model == "gpt":
        return await call_gpt(prompt, api_key)
    elif model == "claude":
        return await call_claude(prompt, api_key)
    else:  # gemini
        return await call_gemini(prompt, api_key)


# ============================================================
# TRANSCRIPT EXTRACTION
# ============================================================
def extract_transcript(raw):
    """מחלץ את רשימת המקטעים מהתמלול"""
    if isinstance(raw, list):
        return raw
    if isinstance(raw, dict):
        if "versions" in raw:
            return raw["versions"][-1].get("segments", [])
        if "segments" in raw:
            return raw["segments"]
        if "utterances" in raw:
            return raw["utterances"]
    return []


# ============================================================
# PROMPTS - Braun & Clarke Method
# ============================================================
def p_initial_coding(segments):
    text = "\n".join([
        f"[{i+1}] {s.get('speaker', 'דובר')}: {s.get('text', '')}"
        for i, s in enumerate(segments)
    ])
    return f"""
בצע קידוד פתוח (Initial Coding) לפי שיטת Braun & Clarke.
עבור כל משפט, זהה קודים רלוונטיים.

התמלול:
{text}

החזר JSON בלבד (ללא markdown):
[
  {{"segment_index": 1, "text": "הטקסט", "speaker": "דובר", "codes": ["קוד1", "קוד2"]}}
]
"""


def p_initial_themes(codes):
    return f"""
קבץ את הקודים הבאים לתימות ראשוניות (Initial Themes) לפי Braun & Clarke.

קודים:
{json.dumps(codes, ensure_ascii=False)}

החזר JSON בלבד:
[
  {{"theme": "שם התימה", "codes": ["קוד1", "קוד2"], "description": "תיאור קצר"}}
]
"""


def p_review_themes(themes, codes):
    return f"""
בצע סקירה וחידוד של התימות (Reviewing Themes) לפי Braun & Clarke.
בדוק שהתימות מגובשות ונפרדות זו מזו.

תימות נוכחיות:
{json.dumps(themes, ensure_ascii=False)}

קודים מקוריים:
{json.dumps(codes, ensure_ascii=False)}

החזר JSON בלבד עם תימות מחודדות:
[
  {{"theme": "שם מחודד", "codes": ["קוד1"], "description": "תיאור מחודד"}}
]
"""


def p_define_themes(themes, segments):
    return f"""
הגדר ושמה את התימות הסופיות (Define & Name Themes) לפי Braun & Clarke.
לכל תימה הוסף ציטוטים תומכים מהתמלול.

תימות:
{json.dumps(themes, ensure_ascii=False)}

תמלול מקורי:
{json.dumps(segments, ensure_ascii=False)}

החזר JSON בלבד:
[
  {{
    "theme": "שם סופי",
    "definition": "הגדרה מפורטת",
    "codes": ["קוד1", "קוד2"],
    "quotes": [
      {{"text": "ציטוט", "speaker": "דובר", "start": 0, "end": 10}}
    ]
  }}
]
"""


def p_report(themes_defined):
    return f"""
כתוב דו"ח מחקרי מסכם (Final Report) לפי Braun & Clarke.

תימות מוגדרות:
{json.dumps(themes_defined, ensure_ascii=False)}

החזר JSON בלבד:
{{
  "summary": "תקציר הממצאים",
  "themes": [
    {{"theme": "שם", "findings": "ממצאים", "significance": "משמעות"}}
  ],
  "implications": "השלכות מחקריות ומעשיות"
}}
"""


def p_matrix(themes_defined):
    return f"""
צור מטריצת תימות מסכמת.

תימות:
{json.dumps(themes_defined, ensure_ascii=False)}

החזר JSON בלבד:
[
  {{
    "theme": "שם התימה",
    "codes_count": 5,
    "quotes_count": 3,
    "key_insight": "תובנה מרכזית"
  }}
]
"""


# ============================================================
# PIPELINE - Full Braun & Clarke
# ============================================================
async def run_pipeline(job_id, transcript_raw, ctx, model, api_key):
    try:
        update_progress(job_id, 5)
        segments = extract_transcript(transcript_raw)
        
        if not segments:
            raise Exception("לא נמצאו מקטעים בתמלול")
        
        logger.info(f"Starting analysis with {len(segments)} segments")

        # שלב 1: קידוד פתוח (20%)
        update_progress(job_id, 10)
        logger.info("Step 1: Initial Coding")
        raw = await model_call(p_initial_coding(segments), model, api_key)
        codes = extract_json(raw)
        if not codes:
            codes = [{"segment_index": i+1, "text": s.get("text", ""), "codes": ["קוד כללי"]} 
                     for i, s in enumerate(segments)]
        update_progress(job_id, 25)

        # שלב 2: תימות ראשוניות (40%)
        logger.info("Step 2: Initial Themes")
        raw = await model_call(p_initial_themes(codes), model, api_key)
        themes_initial = extract_json(raw)
        if not themes_initial:
            themes_initial = [{"theme": "תימה כללית", "codes": [], "description": ""}]
        update_progress(job_id, 40)

        # שלב 3: סקירת תימות (55%)
        logger.info("Step 3: Reviewing Themes")
        raw = await model_call(p_review_themes(themes_initial, codes), model, api_key)
        themes_reviewed = extract_json(raw)
        if not themes_reviewed:
            themes_reviewed = themes_initial
        update_progress(job_id, 55)

        # שלב 4: הגדרת תימות (70%)
        logger.info("Step 4: Define & Name Themes")
        raw = await model_call(p_define_themes(themes_reviewed, segments), model, api_key)
        themes_defined = extract_json(raw)
        if not themes_defined:
            themes_defined = [{"theme": t.get("theme", "תימה"), "definition": "", "codes": t.get("codes", []), "quotes": []} 
                             for t in themes_reviewed]
        update_progress(job_id, 70)

        # שלב 5: דו"ח סופי (85%)
        logger.info("Step 5: Final Report")
        raw = await model_call(p_report(themes_defined), model, api_key)
        report = extract_json(raw)
        if not report:
            report = {"summary": "לא נוצר סיכום", "themes": [], "implications": ""}
        update_progress(job_id, 85)

        # שלב 6: מטריצה (95%)
        logger.info("Step 6: Matrix")
        raw = await model_call(p_matrix(themes_defined), model, api_key)
        matrix = extract_json(raw)
        if not matrix:
            matrix = []
        update_progress(job_id, 95)

        logger.info("Pipeline completed successfully")
        
        return {
            "clean_transcript": segments,
            "codes": codes,
            "themes_initial": themes_initial,
            "themes_reviewed": themes_reviewed,
            "themes_defined": themes_defined,
            "report": report,
            "matrix": matrix
        }

    except Exception as e:
        logger.error(f"Pipeline error: {e}")
        raise


# ============================================================
# BACKGROUND RUNNER
# ============================================================
async def background_run(job_id, transcript, ctx, model, api_key):
    logger.info(f"Background task started for job {job_id}")
    try:
        result = await run_pipeline(job_id, transcript, ctx, model, api_key)
        set_result(job_id, result)
    except Exception as e:
        logger.error(f"Job {job_id} failed: {e}")
        set_error(job_id, str(e))


# ============================================================
# API ENDPOINTS
# ============================================================
@app.post("/agent/analyze")
async def analyze(req: AnalysisRequest):
    job_id = str(uuid.uuid4())
    create_job(job_id)
    
    asyncio.create_task(
        background_run(job_id, req.transcript, req.research_context, req.model, req.api_key)
    )
    
    return {"job_id": job_id, "status": "processing"}


@app.get("/agent/status/{job_id}")
async def status(job_id: str):
    return JOBS.get(job_id, {"error": "not found"})


@app.get("/ping")
async def ping():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000)
