import uuid
import json
import asyncio
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx

app = FastAPI(title="Qualitative Agent – GPT-5.1 (Async Task)")

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
    model: str = "gpt"
    api_key: str | None = None


# ============================================================
# GPT-5.1 CALL
# ============================================================
async def call_gpt_51(prompt, api_key):
    url = "https://api.openai.com/v1/responses"
    headers = {"Authorization": f"Bearer {api_key}"}
    body = {
        "model": "gpt-5.1",
        "input": prompt
    }

    async with httpx.AsyncClient(timeout=40.0) as client:
        try:
            r = await client.post(url, headers=headers, json=body)
        except Exception as e:
            print("HTTPX ERROR:", e)
            return "{}"

    try:
        return r.json()["output_text"]
    except:
        print("PARSE ERROR:", r.text)
        return "{}"


async def model_call(prompt, model, api_key):
    return await call_gpt_51(prompt, api_key)


# ============================================================
# HELPER – extract latest transcript
# ============================================================
def extract_latest_version(raw):
    if isinstance(raw, dict):
        if "versions" in raw:
            return raw["versions"][-1]["segments"]
        if "segments" in raw:
            return raw["segments"]
    return raw


# ============================================================
# PROMPTS (פשוטים כרגע כדי לא יתקע בשלב ראשון)
# ============================================================
def p_initial_codes(text):
    return f"""
החזר רק JSON של קודים בסיסיים.

[{{"sentence":"{text}", "code":"קוד ראשוני"}}]]
"""


def p_themes(codes):
    return f"""
הפק תימות פשוטות מתוך הקודים הבאים:
{json.dumps(codes, ensure_ascii=False)}

החזר JSON:
[{{"theme":"תימה ראשונית","codes":["..."]}}]
"""


# ============================================================
# PIPELINE – גרסה עובדת
# ============================================================
async def run_pipeline(job_id, transcript_raw, ctx, model, api_key):

    update_progress(job_id, 5)
    transcript = extract_latest_version(transcript_raw)

    # INITIAL CODING
    all_codes = []
    total = len(transcript)

    for i, seg in enumerate(transcript, start=1):
        txt = seg.get("text", "")
        raw = await model_call(p_initial_codes(txt), model, api_key)

        try:
            codes = json.loads(raw)
        except:
            codes = [{"sentence": txt, "code": "קוד"}]

        all_codes.append({"segment": seg, "codes": codes})
        update_progress(job_id, 5 + (i / total) * 30)

    # THEMES
    update_progress(job_id, 40)
    th_raw = await model_call(p_themes(all_codes), model, api_key)
    try:
        themes = json.loads(th_raw)
    except:
        themes = [{"theme": "תימה", "codes": []}]

    update_progress(job_id, 80)

    # FINAL RESULT
    return {
        "clean_transcript": transcript,
        "codes": all_codes,
        "themes": themes,
    }


# ============================================================
# ASYNC BACKGROUND EXECUTION — FIX FOR RENDER
# ============================================================
async def background_run(job_id, transcript, ctx, m, key):
    print("BACKGROUND STARTED")
    try:
        result = await run_pipeline(job_id, transcript, ctx, m, key)
        set_result(job_id, result)
    except Exception as e:
        print("PIPELINE ERROR:", e)
        set_error(job_id, str(e))


# ============================================================
# API
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
