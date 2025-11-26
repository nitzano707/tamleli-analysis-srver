import uuid
import json
import asyncio
from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx


# ============================================================
# FASTAPI + CORS
# ============================================================
app = FastAPI(title="Braun & Clarke Qualitative Agent")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # בפרודקשן תעדכן!
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# JOB STORE (In-Memory)
# ============================================================
JOBS = {}


def create_job(job_id):
    JOBS[job_id] = {
        "status": "running",
        "progress": 0,
        "result": None,
        "error": None
    }


def update_progress(job_id, val):
    if job_id in JOBS:
        JOBS[job_id]["progress"] = int(val)


def set_result(job_id, result):
    if job_id in JOBS:
        JOBS[job_id]["status"] = "done"
        JOBS[job_id]["progress"] = 100
        JOBS[job_id]["result"] = result


def set_error(job_id, err):
    if job_id in JOBS:
        JOBS[job_id]["status"] = "error"
        JOBS[job_id]["error"] = str(err)


# ============================================================
# REQUEST MODEL
# ============================================================
class AnalysisRequest(BaseModel):
    transcript: list
    research_context: dict = {}
    model: str = "gpt"
    api_key: str | None = None


# ============================================================
# PROVIDERS
# ============================================================
async def call_gpt(prompt, api_key):
    headers = {"Authorization": f"Bearer {api_key}"}
    body = {
        "model": "gpt-4.1",
        "messages": [{"role": "user", "content": prompt}]
    }
    async with httpx.AsyncClient(timeout=300) as client:
        r = await client.post("https://api.openai.com/v1/chat/completions",
                              headers=headers, json=body)
    data = r.json()
    return data["choices"][0]["message"]["content"]


async def model_call(prompt, model, api_key):
    return await call_gpt(prompt, api_key)


# ============================================================
# PROMPTS — Braun & Clarke (6 Phases)
# ============================================================

def p_clean_transcript(transcript):
    return f"""
נקה ותאם את התמלול הבא. אם קיימות מספר גרסאות, בחר בגרסה החדשה ביותר.
עליך להחזיר מערך JSON של מקטעים בצורה:
[{{"id":1,"text":"...","speaker":"...","start":0.0,"end":3.2}}, ...]

תמלול:
{json.dumps(transcript, ensure_ascii=False)}
"""


def p_initial_codes(segment, context):
    return f"""
בצע קידוד פתוח (Initial Coding) לפי Braun & Clarke על המקטע הבא.
אסור לדלג על משפטים. עבור כל משפט החזר:
{{
 "sentence": "...",
 "code": "..."
}}

הקשר מחקרי:
{json.dumps(context, ensure_ascii=False, indent=2)}

טקסט:
{segment}
"""


def p_generate_themes(codes):
    return f"""
צור תימות ראשוניות (Initial Themes) בהתאם לקודים הבאים:
{json.dumps(codes, ensure_ascii=False)}

החזר JSON מבנה:
[
 {{"theme":"...", "codes":["..."]}}
]
"""


def p_review_themes(themes, transcript):
    return f"""
בצע סקירה (Reviewing Themes).
ודא שכל תימה נתמכת בציטוטים מדויקים מהתמלול.

Themes:
{json.dumps(themes, ensure_ascii=False)}

Transcript:
{json.dumps(transcript, ensure_ascii=False)}

החזר JSON מבנה:
[
 {{
   "theme":"...",
   "codes":[...],
   "quotes":[{{"text":"...","start":0.0,"end":3.1,"speaker":"..."}}]
 }}
]
"""


def p_define_themes(reviewed):
    return f"""
בצע הגדרה ומתן שם (Defining & Naming Themes) לכל תימה.

להחזיר:
[
 {{
   "theme":"...",
   "definition":"...",
   "codes":[...],
   "quotes":[...]
 }}
]
"""


def p_report(final):
    return f"""
כתוב דו"ח מחקרי לפי Braun & Clarke על בסיס התימות הבאות:
{json.dumps(final, ensure_ascii=False)}

החזר JSON:
{{
 "summary":"...",
 "implications":"...",
 "themes": [...]
}}
"""


def p_matrix(final):
    return f"""
בנה מטריצה מסכמת הכוללת:
- תימה
- הגדרה
- קודים
- ציטוטים
- פרשנות

Data:
{json.dumps(final, ensure_ascii=False)}

החזר JSON:
[
 {{
   "theme":"...",
   "definition":"...",
   "codes":["..."],
   "quotes":[...],
   "interpretation":"..."
 }}
]
"""


# ============================================================
# PIPELINE (6 PHASES)
# ============================================================
async def run_pipeline(job_id, transcript_raw, context, model, api_key):

    # ------------------ 1. Familiarization ------------------
    update_progress(job_id, 5)
    clean_raw = await model_call(p_clean_transcript(transcript_raw), model, api_key)
    transcript = json.loads(clean_raw)

    # ------------------ 2. Initial Coding -------------------
    all_codes = []
    total = len(transcript)
    for i, seg in enumerate(transcript, start=1):
        c_raw = await model_call(p_initial_codes(seg["text"], context), model, api_key)
        codes = json.loads(c_raw)
        all_codes.append({
            "segment_id": seg["id"],
            "codes": codes
        })
        update_progress(job_id, 5 + (i / total) * 30)

    # ------------------ 3. Generate Themes ------------------
    update_progress(job_id, 40)
    th_raw = await model_call(p_generate_themes(all_codes), model, api_key)
    initial_themes = json.loads(th_raw)

    # ------------------ 4. Review Themes --------------------
    update_progress(job_id, 55)
    rv_raw = await model_call(p_review_themes(initial_themes, transcript), model, api_key)
    reviewed = json.loads(rv_raw)

    # ------------------ 5. Define & Name Themes -------------
    update_progress(job_id, 70)
    df_raw = await model_call(p_define_themes(reviewed), model, api_key)
    defined = json.loads(df_raw)

    # ------------------ 6. Final Report ---------------------
    update_progress(job_id, 85)
    rpt_raw = await model_call(p_report(defined), model, api_key)
    report = json.loads(rpt_raw)

    # ------------------ MATRIX ------------------------------
    update_progress(job_id, 95)
    mx_raw = await model_call(p_matrix(defined), model, api_key)
    matrix = json.loads(mx_raw)

    return {
        "clean_transcript": transcript,
        "codes": all_codes,
        "themes_initial": initial_themes,
        "themes_reviewed": reviewed,
        "themes_defined": defined,
        "report": report,
        "matrix": matrix
    }


# ============================================================
# BACKGROUND TASK WRAPPER
# ============================================================
async def background_task(job_id, transcript, ctx, model, api_key):
    try:
        result = await run_pipeline(job_id, transcript, ctx, model, api_key)
        set_result(job_id, result)
    except Exception as e:
        set_error(job_id, e)


# ============================================================
# API ENDPOINTS
# ============================================================
@app.post("/agent/analyze")
async def analyze(req: AnalysisRequest, background_tasks: BackgroundTasks):
    job_id = str(uuid.uuid4())
    create_job(job_id)

    background_tasks.add_task(
        background_task,
        job_id,
        req.transcript,
        req.research_context,
        req.model,
        req.api_key
    )
    return {"job_id": job_id, "status": "processing"}


@app.get("/agent/status/{job_id}")
async def status(job_id: str):
    if job_id not in JOBS:
        return {"error": "not found"}
    return JOBS[job_id]


@app.get("/ping")
async def ping():
    return {"status": "ok"}


# ============================================================
# LOCAL
# ============================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000)
