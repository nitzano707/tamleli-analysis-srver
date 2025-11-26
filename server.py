import uuid
import json
import asyncio
from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx


app = FastAPI(title="Braun & Clarke Agent – GPT-5.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
    if job_id in JOBS:
        JOBS[job_id]["status"] = "done"
        JOBS[job_id]["progress"] = 100
        JOBS[job_id]["result"] = r


def set_error(job_id, err):
    if job_id in JOBS:
        JOBS[job_id]["status"] = "error"
        JOBS[job_id]["error"] = str(err)


class AnalysisRequest(BaseModel):
    transcript: dict | list
    research_context: dict = {}
    model: str = "gpt"
    api_key: str | None = None


# ============================================================
# 🔵 פונקציה שמנקה את התמלול ובוחרת את הגרסה העדכנית
# ============================================================
def extract_latest_version(raw):
    if isinstance(raw, dict):
        if "versions" in raw:             # תמלול עם כמה גרסאות
            latest = raw["versions"][-1]
            return latest.get("segments", [])
        if "segments" in raw:
            return raw["segments"]
    return raw  # אם זה כבר list תקין


# ============================================================
# 🔵 מודל GPT-5.1 החדש
# ============================================================
async def call_gpt_51(prompt, api_key):
    url = "https://api.openai.com/v1/responses"
    headers = {"Authorization": f"Bearer {api_key}"}
    body = {
        "model": "gpt-5.1",
        "input": prompt
    }
    async with httpx.AsyncClient(timeout=300) as client:
        r = await client.post(url, headers=headers, json=body)
    data = r.json()
    return data["output_text"]


async def model_call(prompt, model, api_key):
    return await call_gpt_51(prompt, api_key)


# ============================================================
# 🔵 PROMPTS – Braun & Clarke
# ============================================================

def p_initial_codes(text, ctx):
    return f"""
בצע קידוד ראשוני ברמת Braun & Clarke.
עבור כל משפט החזר JSON:
{{"sentence":"...", "code":"..."}}

הקשר:
{json.dumps(ctx, ensure_ascii=False, indent=2)}

טקסט:
{text}
"""


def p_generate_themes(codes):
    return f"""
צור תימות ראשוניות מתוך הקודים הבאים:

{json.dumps(codes, ensure_ascii=False)}

החזר JSON:
[{{"theme":"...", "codes":["..."]}}]
"""


def p_review_themes(themes, transcript):
    return f"""
סקירת תימות:
צרף ציטוטים מדויקים מהתמלול עם start,end,speaker.

Themes:
{json.dumps(themes, ensure_ascii=False)}

Transcript:
{json.dumps(transcript, ensure_ascii=False)}

החזר JSON.
"""


def p_define_themes(reviewed):
    return f"""
הגדרה ושיום תימות:
החזר JSON עם:
theme, definition, codes, quotes
"""


def p_report(final):
    return f"""
כתוב דו״ח מחקרי סופי לפי Braun & Clarke.
החזר:
{{
 "summary":"...",
 "implications":"...",
 "themes":[...]
}}
"""


def p_matrix(final):
    return f"""
בנה מטריצה:
theme, definition, codes, quotes, interpretation

Data:
{json.dumps(final, ensure_ascii=False)}

החזר JSON.
"""


# ============================================================
# 🔵 PIPELINE
# ============================================================
async def run_pipeline(job_id, transcript_raw, ctx, model, api_key):

    # 1) extract latest version
    update_progress(job_id, 5)
    transcript = extract_latest_version(transcript_raw)

    # 2) initial coding
    all_codes = []
    total = len(transcript)

    for i, seg in enumerate(transcript, start=1):
        text = seg.get("text", "")
        c_raw = await model_call(p_initial_codes(text, ctx), model, api_key)
        codes = json.loads(c_raw)
        all_codes.append({"segment": seg, "codes": codes})

        update_progress(job_id, 5 + (i / total) * 25)

    # 3) generate themes
    update_progress(job_id, 35)
    th_raw = await model_call(p_generate_themes(all_codes), model, api_key)
    themes_initial = json.loads(th_raw)

    # 4) review themes
    update_progress(job_id, 50)
    rv_raw = await model_call(p_review_themes(themes_initial, transcript), model, api_key)
    themes_reviewed = json.loads(rv_raw)

    # 5) define
    update_progress(job_id, 70)
    df_raw = await model_call(p_define_themes(themes_reviewed), model, api_key)
    themes_defined = json.loads(df_raw)

    # 6) report
    update_progress(job_id, 85)
    rpt_raw = await model_call(p_report(themes_defined), model, api_key)
    report = json.loads(rpt_raw)

    # matrix
    update_progress(job_id, 95)
    mx_raw = await model_call(p_matrix(themes_defined), model, api_key)
    matrix = json.loads(mx_raw)

    return {
        "clean_transcript": transcript,
        "codes": all_codes,
        "themes_initial": themes_initial,
        "themes_reviewed": themes_reviewed,
        "themes_defined": themes_defined,
        "report": report,
        "matrix": matrix
    }


async def background(job_id, transcript, ctx, m, key):
    try:
        r = await run_pipeline(job_id, transcript, ctx, m, key)
        set_result(job_id, r)
    except Exception as e:
        set_error(job_id, e)


@app.post("/agent/analyze")
async def analyze(req: AnalysisRequest, background_tasks: BackgroundTasks):
    job_id = str(uuid.uuid4())
    create_job(job_id)

    background_tasks.add_task(
        background,
        job_id,
        req.transcript,
        req.research_context,
        req.model,
        req.api_key
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
