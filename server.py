import uuid
import json
import asyncio
from typing import Union, Optional

from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx
import re


# ─────────────────────────────────────────────
# FASTAPI + CORS
# ─────────────────────────────────────────────
app = FastAPI(title="Qualitative Analysis Agent")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # בפרודקשן מומלץ להגביל
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────
# זיכרון עבודות (אפשר להחליף ל-Redis)
# ─────────────────────────────────────────────
JOBS = {}

def create_job(job_id: str):
    JOBS[job_id] = {
        "status": "running",
        "progress": 0,
        "result": None,
        "error": None
    }

def update_progress(job_id: str, p: int):
    if job_id in JOBS:
        JOBS[job_id]["progress"] = p

def set_result(job_id: str, r: dict):
    if job_id in JOBS:
        JOBS[job_id]["status"] = "done"
        JOBS[job_id]["result"] = r
        JOBS[job_id]["progress"] = 100

def set_error(job_id: str, e: str):
    if job_id in JOBS:
        JOBS[job_id]["status"] = "error"
        JOBS[job_id]["error"] = e


# ─────────────────────────────────────────────
# מודל בקשה
# ─────────────────────────────────────────────
class AnalysisRequest(BaseModel):
    transcript: Union[list, dict]
    research_context: dict = {}
    model: str = "gpt"
    api_key: Optional[str] = None


# ─────────────────────────────────────────────
# JSON FIX – תקן JSON שבור
# ─────────────────────────────────────────────
def safe_json(text: str):
    """מתקן JSON שבור שמודלים מחזירים לעיתים"""
    if not text:
        return None

    # מחיקת ```json ``` 
    cleaned = re.sub(r"```json", "", text, flags=re.IGNORECASE)
    cleaned = re.sub(r"```", "", cleaned).strip()

    # ניסיון ראשון
    try:
        return json.loads(cleaned)
    except:
        pass

    # ניסיון שני — השלמת סוגריים
    try:
        cleaned2 = cleaned + "}" if cleaned.count("{") > cleaned.count("}") else cleaned
        cleaned2 = cleaned2 + "]" if cleaned.count("[") > cleaned.count("]") else cleaned2
        return json.loads(cleaned2)
    except:
        return None


# ─────────────────────────────────────────────
# NORMALIZE – המרת התמלול לפורמט אחיד
# ─────────────────────────────────────────────
def normalize_transcript(t):
    if isinstance(t, list):
        return t

    if isinstance(t, dict):
        if isinstance(t.get("segments"), list):
            return t["segments"]
        if isinstance(t.get("utterances"), list):
            return t["utterances"]

    raise Exception("פורמט תמלול לא מזוהה (נדרש segments[] או רשימה).")


# ─────────────────────────────────────────────
# OpenAI (GPT-4.1 / GPT-4.1-mini / GPT-5.1)
# ─────────────────────────────────────────────
async def call_gpt(prompt: str, api_key: str):
    headers = {"Authorization": f"Bearer {api_key}"}

    body = {
        "model": "gpt-5.1",     # הדגם החדש ביותר
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0.1
    }

    async with httpx.AsyncClient(timeout=200) as client:
        r = await client.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=body
        )

    if r.status_code >= 400:
        raise Exception(f"OpenAI error: {r.text}")

    return r.json()["choices"][0]["message"]["content"]


# ─────────────────────────────────────────────
# Unified model router
# ─────────────────────────────────────────────
async def model_call(prompt: str, model: str, api_key: str):
    if model == "gpt":
        return await call_gpt(prompt, api_key)
    return await call_gpt(prompt, api_key)  # ברירת מחדל


# ─────────────────────────────────────────────
# PROMPTS – גרסה קשיחה ל-GPT
# ─────────────────────────────────────────────
def P_OPEN(text, ctx):
    return f"""
אתה מנתח איכותני. עליך להחזיר JSON בלבד, ללא הסברים.
אם אינך יכול להחזיר JSON תקני – החזר: {{"codes": []}}

טקסט לניתוח:
{text}

הקשר מחקרי:
{json.dumps(ctx, ensure_ascii=False)}

החזר JSON בלבד בפורמט:
{{
  "codes": [
    {{
      "sentence": "…",
      "code": "…"
    }}
  ]
}}
"""


def P_AXIAL(open_codes, ctx):
    return f"""
אתה מנתח איכותני. בצע Axial Coding.

החזר JSON בלבד בפורמט:
{{ "axial": [...] }}

קלט:
קידוד פתוח:
{json.dumps(open_codes, ensure_ascii=False)}
הקשר:
{json.dumps(ctx, ensure_ascii=False)}
"""


def P_THEMES(axial, ctx):
    return f"""
אתה מנתח איכותני. זהה Themes.

החזר JSON בלבד:
{{ "themes": [...] }}

Axial:
{json.dumps(axial, ensure_ascii=False)}
"""


def P_QUOTES(themes, transcript):
    return f"""
התאם ציטוט לכל קוד.

החזר JSON בלבד:
{{ "quotes": {{ ... }} }}

Themes:
{json.dumps(themes, ensure_ascii=False)}
Transcript:
{json.dumps(transcript, ensure_ascii=False)}
"""


def P_INTERPRET(themes, quotes):
    return f"""
כתוב פרשנות עומק.

החזר JSON בלבד:
{{ "interpretations": {{ ... }} }}

Themes:
{json.dumps(themes, ensure_ascii=False)}
Quotes:
{json.dumps(quotes, ensure_ascii=False)}
"""


def P_MATRIX(themes, quotes, interp):
    return f"""
בנה מטריצה מסכמת.

החזר JSON בלבד:
{{ "matrix": [...] }}

Themes:
{json.dumps(themes, ensure_ascii=False)}
Quotes:
{json.dumps(quotes, ensure_ascii=False)}
Interpretations:
{json.dumps(interp, ensure_ascii=False)}
"""


# ─────────────────────────────────────────────
# ENFORCE – אכיפה קשיחה לכל מקטע
# ─────────────────────────────────────────────
async def enforce_open_code(text, ctx, model, api_key):
    for _ in range(3):
        raw = await model_call(P_OPEN(text, ctx), model, api_key)
        parsed = safe_json(raw)
        if parsed and "codes" in parsed:
            return parsed

    # fallback – לא מפיל את כל התהליך
    return {"codes": []}


# ─────────────────────────────────────────────
# PIPELINE מלא
# ─────────────────────────────────────────────
async def pipeline(job_id, transcript, ctx, model, api_key):

    update_progress(job_id, 5)

    # 1) open coding
    open_codes = []
    total = len(transcript)

    for i, seg in enumerate(transcript, start=1):
        text = seg.get("text") or seg.get("sentence") or ""
        oc = await enforce_open_code(text, ctx, model, api_key)
        open_codes.append({"id": seg.get("id", i), **oc})

        update_progress(job_id, int(5 + (i / total) * 40))

    # 2) Axial
    update_progress(job_id, 55)
    axial_raw = await model_call(P_AXIAL(open_codes, ctx), model, api_key)
    axial = safe_json(axial_raw) or {"axial": []}

    # 3) Themes
    update_progress(job_id, 65)
    t_raw = await model_call(P_THEMES(axial, ctx), model, api_key)
    themes = safe_json(t_raw) or {"themes": []}

    # 4) Quotes
    update_progress(job_id, 75)
    q_raw = await model_call(P_QUOTES(themes, transcript), model, api_key)
    quotes = safe_json(q_raw) or {"quotes": {}}

    # 5) Interpretations
    update_progress(job_id, 85)
    i_raw = await model_call(P_INTERPRET(themes, quotes), model, api_key)
    interp = safe_json(i_raw) or {"interpretations": {}}

    # 6) Matrix
    update_progress(job_id, 95)
    m_raw = await model_call(P_MATRIX(themes, quotes, interp), model, api_key)
    matrix = safe_json(m_raw) or {"matrix": []}

    return {
        "openCoding": open_codes,
        "axial": axial,
        "themes": themes,
        "quotes": quotes,
        "interpretations": interp,
        "matrix": matrix
    }


# ─────────────────────────────────────────────
# BACKGROUND WORKER
# ─────────────────────────────────────────────
async def background(job_id, transcript_raw, ctx, model, api_key):
    try:
        transcript = normalize_transcript(transcript_raw)
        result = await pipeline(job_id, transcript, ctx, model, api_key)
        set_result(job_id, result)
    except Exception as e:
        set_error(job_id, str(e))


# ─────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────
@app.post("/agent/analyze")
async def analyze(req: AnalysisRequest, tasks: BackgroundTasks):
    if not req.api_key:
        raise Exception("נדרש מפתח API חוקי של OpenAI.")

    job_id = str(uuid.uuid4())
    create_job(job_id)

    tasks.add_task(
        background,
        job_id,
        req.transcript,
        req.research_context,
        req.model,
        req.api_key,
    )

    return {"job_id": job_id, "status": "processing"}


@app.get("/agent/status/{job_id}")
async def status(job_id: str):
    return JOBS.get(job_id, {"error": "job not found"})


@app.get("/ping")
async def ping():
    return {"status": "ok"}


# ─────────────────────────────────────────────
# RUN LOCAL
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000)
