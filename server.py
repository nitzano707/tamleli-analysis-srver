import uuid
import json
import asyncio
from typing import Union, Optional

from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx

# ─────────────────────────────────────────────
# FastAPI + CORS
# ─────────────────────────────────────────────
app = FastAPI(title="Qualitative Analysis Agent")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # בפרודקשן: לצמצם למקורות המותרים בלבד
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────
# JOB STORE — בזיכרון (אפשר להחליף ל-Redis בעתיד)
# ─────────────────────────────────────────────
JOBS: dict[str, dict] = {}


def create_job(job_id: str):
    JOBS[job_id] = {
        "status": "running",
        "progress": 0,
        "result": None,
        "error": None,
    }


def update_progress(job_id: str, progress: int):
    if job_id in JOBS:
        JOBS[job_id]["progress"] = progress


def set_result(job_id: str, result: dict):
    if job_id in JOBS:
        JOBS[job_id]["status"] = "done"
        JOBS[job_id]["progress"] = 100
        JOBS[job_id]["result"] = result


def set_error(job_id: str, error: str):
    if job_id in JOBS:
        JOBS[job_id]["status"] = "error"
        JOBS[job_id]["error"] = str(error)


# ─────────────────────────────────────────────
# מודל בקשה מהפרונט
# מאפשר גם list ישיר וגם אובייקט עם segments
# ─────────────────────────────────────────────
class AnalysisRequest(BaseModel):
    transcript: Union[list, dict]
    research_context: dict = {}
    model: str = "gemini"  # gpt / claude / gemini
    api_key: Optional[str] = None


# ─────────────────────────────────────────────
# NORMALIZATION — תמיכה ב-Tamleli JSON וכו'
# ─────────────────────────────────────────────
def normalize_transcript(t: Union[list, dict]) -> list:
    """
    מחזיר תמיד list של מקטעים.
    תומך בפורמטים:
    1. [ {...}, {...} ]
    2. { "segments": [ {...}, ... ] }
    3. { "utterances": [ {...}, ... ] }
    """
    if isinstance(t, list):
        return t

    if isinstance(t, dict):
        if isinstance(t.get("segments"), list):
            return t["segments"]
        if isinstance(t.get("utterances"), list):
            return t["utterances"]

    raise Exception("פורמט התמלול לא מזוהה – נדרש מערך או אובייקט עם 'segments' / 'utterances'.")


# ─────────────────────────────────────────────
# AI PROVIDERS
# ─────────────────────────────────────────────
async def call_gpt(prompt: str, api_key: str) -> str:
    headers = {"Authorization": f"Bearer {api_key}"}
    body = {
        "model": "gpt-4.1",
        "messages": [{"role": "user", "content": prompt}],
    }
    async with httpx.AsyncClient(timeout=200) as client:
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
        "model": "claude-3-5-sonnet-20241022",
        "max_tokens": 4000,
        "messages": [{"role": "user", "content": prompt}],
    }
    async with httpx.AsyncClient(timeout=200) as client:
        r = await client.post(
            "https://api.anthropic.com/v1/messages",
            headers=headers,
            json=body,
        )
    r.raise_for_status()
    return r.json()["content"][0]["text"]


async def call_gemini_pro(prompt: str, api_key: str) -> str:
    url = (
        "https://generativelanguage.googleapis.com/"
        "v1beta/models/gemini-pro:generateContent"
        f"?key={api_key}"
    )
    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    async with httpx.AsyncClient(timeout=200) as client:
        r = await client.post(url, json=payload)
    r.raise_for_status()
    data = r.json()
    return data["candidates"][0]["content"]["parts"][0]["text"]


async def model_call(prompt: str, model: str, api_key: Optional[str]) -> str:
    """
    שכבת הפשטה אחת לכל המודלים.
    כרגע נדרש api_key תקף לכל מודל.
    """
    if not api_key:
        raise Exception("לא הוזן api_key למודל. יש להזין מפתח מתאים למודל שנבחר.")

    if model == "gpt":
        return await call_gpt(prompt, api_key)
    if model == "claude":
        return await call_claude(prompt, api_key)
    if model == "gemini":
        return await call_gemini_pro(prompt, api_key)

    # ברירת מחדל — ננסה GPT
    return await call_gpt(prompt, api_key)


# ─────────────────────────────────────────────
# פרומפטים
# ─────────────────────────────────────────────
def p_open_coding(text: str, context: dict) -> str:
    return f"""
בצע קידוד פתוח (Open Coding) על הקטע הבא.
עליך לזהות קודים ברורים בלבד.
חובה להתייחס לכל משפט, ללא דילוג.

הקשר מחקרי:
{json.dumps(context, ensure_ascii=False, indent=2)}

הטקסט:
{text}

החזר JSON תקני בלבד:
{{"codes":[{{"sentence":"...","code":"..."}}]}}
"""


def p_coverage(text: str, codes_json: dict) -> str:
    return f"""
בדיקת כיסוי. בדוק שכל משפט בטקסט מכוסה בקוד.

טקסט:
{text}

קידוד:
{json.dumps(codes_json, ensure_ascii=False)}

החזר JSON:
{{"status":"ok"}} או
{{"status":"missing","sentences":["..."]}}
"""


def p_cross(text: str, codes_json: dict) -> str:
    return f"""
Cross-check:
האם הקידוד מתייחס לכל המשפטים באופן מלא?

טקסט:
{text}

קידוד:
{json.dumps(codes_json, ensure_ascii=False)}

החזר JSON:
{{"cross_ok": true}} או
{{"cross_ok": false}}
"""


def p_axial(open_coding: list, context: dict) -> str:
    return f"""
בצע Axial Coding על בסיס כלל הקודים.

הקשר מחקרי:
{json.dumps(context, ensure_ascii=False, indent=2)}

קידוד פתוח:
{json.dumps(open_coding, ensure_ascii=False)}

החזר JSON תקני בלבד.
"""


def p_themes(axial: dict, context: dict) -> str:
    return f"""
זהה Themes (תמות) על בסיס הקידוד הצירי.

הקשר מחקרי:
{json.dumps(context, ensure_ascii=False, indent=2)}

Axial:
{json.dumps(axial, ensure_ascii=False)}

החזר JSON תקני בלבד.
"""


def p_quotes(themes: dict, transcript: list) -> str:
    return f"""
התאם לכל קוד ציטוט מדויק מתוך התמלול.

Themes:
{json.dumps(themes, ensure_ascii=False)}

תמלול:
{json.dumps(transcript, ensure_ascii=False)}

החזר JSON תקני בלבד.
"""


def p_interpret(themes: dict, quotes: dict) -> str:
    return f"""
כתוב פרשנות עומק מחקרית לכל תימה.

Themes:
{json.dumps(themes, ensure_ascii=False)}

Quotes:
{json.dumps(quotes, ensure_ascii=False)}

החזר JSON תקני בלבד.
"""


def p_matrix(themes: dict, quotes: dict, interpretations: dict) -> str:
    return f"""
בנה מטריצה מסכמת הכוללת:
- תימה
- קודים
- ציטוטים
- פרשנות

Themes:
{json.dumps(themes, ensure_ascii=False)}

Quotes:
{json.dumps(quotes, ensure_ascii=False)}

Interpretations:
{json.dumps(interpretations, ensure_ascii=False)}

החזר JSON תקני בלבד.
"""


# ─────────────────────────────────────────────
# Strict Enforcement — per segment
# ─────────────────────────────────────────────
async def enforce_segment(text: str, context: dict, model: str, api_key: Optional[str]) -> dict:
    """
    אוכף שהמודל יעבור על כל המקטע: open coding + coverage + cross-check.
    מנסה עד 3 פעמים לפני שמרים שגיאה.
    """
    for attempt in range(3):
        # 1. Open coding
        oc_prompt = p_open_coding(text, context)
        oc_raw = await model_call(oc_prompt, model, api_key)
        try:
            oc = json.loads(oc_raw)
        except Exception:
            continue

        # 2. Coverage
        cov_prompt = p_coverage(text, oc)
        cov_raw = await model_call(cov_prompt, model, api_key)
        try:
            cov = json.loads(cov_raw)
        except Exception:
            continue

        if cov.get("status") == "missing":
            # אם חסרות שורות — ננסה שוב
            continue

        # 3. Cross-check
        cross_prompt = p_cross(text, oc)
        cross_raw = await model_call(cross_prompt, model, api_key)
        try:
            cross = json.loads(cross_raw)
        except Exception:
            continue

        if cross.get("cross_ok") is True:
            return oc

    raise Exception("אכיפת קידוד נכשלה על מקטע טקסט")


# ─────────────────────────────────────────────
# FULL ANALYSIS PIPELINE
# ─────────────────────────────────────────────
async def run_pipeline(
    job_id: str,
    transcript: list,
    context: dict,
    model: str,
    api_key: Optional[str],
) -> dict:
    update_progress(job_id, 5)

    # --- שלב 1: קידוד פתוח על כל המקטעים ---
    open_coding: list[dict] = []
    total = max(len(transcript), 1)

    for idx, seg in enumerate(transcript, start=1):
        seg_id = seg.get("id", idx)
        text = seg.get("text") or seg.get("sentence") or str(seg)

        oc = await enforce_segment(text, context, model, api_key)
        open_coding.append({"segment_id": seg_id, "codes": oc})

        update_progress(job_id, int(5 + (idx / total) * 40))

    # --- שלב 2: Axial ---
    update_progress(job_id, 55)
    axial_raw = await model_call(p_axial(open_coding, context), model, api_key)
    axial = json.loads(axial_raw)

    # --- שלב 3: Themes ---
    update_progress(job_id, 65)
    themes_raw = await model_call(p_themes(axial, context), model, api_key)
    themes = json.loads(themes_raw)

    # --- שלב 4: Quotes ---
    update_progress(job_id, 75)
    quotes_raw = await model_call(p_quotes(themes, transcript), model, api_key)
    quotes = json.loads(quotes_raw)

    # --- שלב 5: Interpretations ---
    update_progress(job_id, 85)
    interp_raw = await model_call(p_interpret(themes, quotes), model, api_key)
    interpretations = json.loads(interp_raw)

    # --- שלב 6: Matrix ---
    update_progress(job_id, 95)
    matrix_raw = await model_call(
        p_matrix(themes, quotes, interpretations),
        model,
        api_key,
    )
    matrix = json.loads(matrix_raw)

    return {
        "openCoding": open_coding,
        "axial": axial,
        "themes": themes,
        "quotes": quotes,
        "interpretations": interpretations,
        "matrix": matrix,
    }


# ─────────────────────────────────────────────
# BACKGROUND TASK RUNNER
# ─────────────────────────────────────────────
async def background_analysis(
    job_id: str,
    transcript_raw: Union[list, dict],
    context: dict,
    model: str,
    api_key: Optional[str],
):
    try:
        transcript = normalize_transcript(transcript_raw)
        result = await run_pipeline(job_id, transcript, context, model, api_key)
        set_result(job_id, result)
    except Exception as e:
        set_error(job_id, str(e))


# ─────────────────────────────────────────────
# API ENDPOINTS
# ─────────────────────────────────────────────
@app.post("/agent/analyze")
async def analyze(req: AnalysisRequest, background_tasks: BackgroundTasks):
    job_id = str(uuid.uuid4())
    create_job(job_id)

    background_tasks.add_task(
        background_analysis,
        job_id,
        req.transcript,
        req.research_context,
        req.model,
        req.api_key,
    )

    return {"job_id": job_id, "status": "processing"}


@app.get("/agent/status/{job_id}")
async def status(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        return {"error": "job not found"}
    return job


@app.get("/ping")
async def ping():
    return {"status": "ok"}


# ─────────────────────────────────────────────
# הפעלה מקומית
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("server:app", host="0.0.0.0", port=8000)
