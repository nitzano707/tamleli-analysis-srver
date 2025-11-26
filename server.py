import uuid
import json
import asyncio
from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx


# ─────────────────────────────────────────────
# הגדרות FastAPI + CORS
# ─────────────────────────────────────────────
app = FastAPI(title="Qualitative Analysis Agent")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],     # בפרודקשן תעדכן!
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────
# JOB STORE — שמירת עבודות בזיכרון (קל משקל)
# אפשר להחליף ל־Redis בעתיד
# ─────────────────────────────────────────────
JOBS = {}


def create_job(job_id):
    JOBS[job_id] = {
        "status": "running",
        "progress": 0,
        "result": None,
        "error": None
    }


def update_progress(job_id, progress):
    if job_id in JOBS:
        JOBS[job_id]["progress"] = progress


def set_result(job_id, result):
    if job_id in JOBS:
        JOBS[job_id]["status"] = "done"
        JOBS[job_id]["progress"] = 100
        JOBS[job_id]["result"] = result


def set_error(job_id, error):
    if job_id in JOBS:
        JOBS[job_id]["status"] = "error"
        JOBS[job_id]["error"] = str(error)


# ─────────────────────────────────────────────
# מודל בקשה
# ─────────────────────────────────────────────
class AnalysisRequest(BaseModel):
    transcript: list
    research_context: dict = {}
    model: str = "gemini"      # gpt / claude / gemini
    api_key: str | None = None


# ─────────────────────────────────────────────
# AI PROVIDERS — unified call
# ─────────────────────────────────────────────
async def call_gpt(prompt, api_key):
    """ GPT Provider """
    headers = {"Authorization": f"Bearer {api_key}"}
    body = {
        "model": "gpt-4.1",   # תוכל לשנות למתקדם יותר
        "messages": [{"role": "user", "content": prompt}]
    }
    async with httpx.AsyncClient(timeout=200) as client:
        r = await client.post("https://api.openai.com/v1/chat/completions",
                              headers=headers, json=body)
    return json.loads(r.text)["choices"][0]["message"]["content"]


async def call_claude(prompt, api_key):
    """ Claude Provider """
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json"
    }
    body = {
        "model": "claude-3-5-sonnet-20241022",
        "max_tokens": 4000,
        "messages": [{"role": "user", "content": prompt}]
    }
    async with httpx.AsyncClient(timeout=200) as client:
        r = await client.post("https://api.anthropic.com/v1/messages",
                              headers=headers, json=body)
    return r.json()["content"][0]["text"]


async def call_gemini_pro(prompt, api_key):
    """ Gemini Pro (Cloud) """
    async with httpx.AsyncClient(timeout=200) as client:
        r = await client.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key={api_key}",
            json={"contents": [{"parts": [{"text": prompt}]}]}
        )
    data = r.json()
    return data["candidates"][0]["content"]["parts"][0]["text"]


async def call_gemini_free(prompt):
    """
    Gemini Flash דרך Cloud API חינמי.
    אין צורך ב־API key (זה fallback בלבד).
    """
    # מודול חינמי של גוגל לעיתים דורש auth
    # לכן נשתמש בסטנד-אין עם dummy anonymous key
    url = (
        "https://generativelanguage.googleapis.com/"
        "v1beta/models/gemini-1.5-flash:generateContent?key=dummy"
    )
    async with httpx.AsyncClient(timeout=200) as client:
        r = await client.post(url,
                              json={"contents": [{"parts": [{"text": prompt}]}]})

    # במידה וקיבלנו תשובת שגיאה כלשהי — ננסה לא להתרסק
    try:
        return r.json()["candidates"][0]["content"]["parts"][0]["text"]
    except:
        raise Exception("Gemini החינמי חסם את הבקשה")


# ─────────────────────────────────────────────
# Unified Provider
# ─────────────────────────────────────────────
async def model_call(prompt, model, api_key):
    # אם יש API key → מודל פרימיום
    if api_key:
        if model == "gpt":
            return await call_gpt(prompt, api_key)
        if model == "claude":
            return await call_claude(prompt, api_key)
        if model == "gemini":
            return await call_gemini_pro(prompt, api_key)

    # אחרת → fallback חינמי
    return await call_gemini_free(prompt)


# ─────────────────────────────────────────────
# פרומפטים
# ─────────────────────────────────────────────
def p_open_coding(text, context):
    return f"""
בצע קידוד פתוח (Open Coding) על הקטע הבא.
עליך לזהות קודים ברורים בלבד.
חובה להתייחס לכל משפט, ללא דילוג.

הקשר מחקרי:
{json.dumps(context, ensure_ascii=False, indent=2)}

הטקסט:
{text}

החזר JSON תקני:
{{"codes":[{{"sentence":"...","code":"..."}}]}}
"""


def p_coverage(text, codes_json):
    return f"""
בדיקת כיסוי. ודא שכל משפט בטקסט מכוסה בקוד.

טקסט:
{text}

קידוד:
{json.dumps(codes_json, ensure_ascii=False)}

החזר:
{{"status":"ok"}} או
{{"status":"missing","sentences":["..."]}}
"""


def p_cross(text, codes_json):
    return f"""
Cross-check:
האם הקידוד מתייחס לכל המשפטים באופן מלא?

טקסט:
{text}

קידוד:
{json.dumps(codes_json, ensure_ascii=False)}

החזר:
{{"cross_ok": true}} או
{{"cross_ok": false}}
"""


def p_axial(open_coding, context):
    return f"""
בצע Axial Coding על בסיס כל הקודים.
חפש צירים (קטגוריות) מאחדות בין קודים.

הקשר מחקרי:
{json.dumps(context,ensure_ascii=False,indent=2)}

קידוד פתוח:
{json.dumps(open_coding,ensure_ascii=False)}

החזר JSON תקני.
"""


def p_themes(axial, context):
    return f"""
זהה תימות (Themes) על בסיס הקידוד הצירי.

הקשר מחקרי:
{json.dumps(context,ensure_ascii=False,indent=2)}

Axial:
{json.dumps(axial, ensure_ascii=False)}

JSON בלבד.
"""


def p_quotes(themes, transcript):
    return f"""
התאם לכל קוד ציטוט מדויק מקטעי התמלול.

תמות:
{json.dumps(themes, ensure_ascii=False)}

תמלול:
{json.dumps(transcript, ensure_ascii=False)}

החזר JSON.
"""


def p_interpret(themes, quotes):
    return f"""
כתוב פרשנות עומק מחקרית לכל תימה.

Themes:
{json.dumps(themes, ensure_ascii=False)}

Quotes:
{json.dumps(quotes, ensure_ascii=False)}

החזר JSON תקני בלבד.
"""


def p_matrix(themes, quotes, interpretations):
    return f"""
בנה מטריצה מסכמת הכוללת:
- תימה
- קודים
- ציטוטים
- פרשנות

Themes:
{json.dumps(themes,ensure_ascii=False)}

Quotes:
{json.dumps(quotes,ensure_ascii=False)}

Interpretations:
{json.dumps(interpretations,ensure_ascii=False)}

החזר JSON.
"""


# ─────────────────────────────────────────────
# Strict Enforcement — per segment
# ─────────────────────────────────────────────
async def enforce_segment(text, context, model, api_key):

    for attempt in range(3):

        # 1. open coding
        oc_prompt = p_open_coding(text, context)
        oc_raw = await model_call(oc_prompt, model, api_key)
        try:
            oc = json.loads(oc_raw)
        except:
            continue

        # 2. coverage
        cov_prompt = p_coverage(text, oc)
        cov_raw = await model_call(cov_prompt, model, api_key)
        try:
            cov = json.loads(cov_raw)
        except:
            continue

        if cov.get("status") == "missing":
            continue

        # 3. cross-check
        cross_prompt = p_cross(text, oc)
        cross_raw = await model_call(cross_prompt, model, api_key)
        try:
            cross = json.loads(cross_raw)
        except:
            continue

        if cross.get("cross_ok") is True:
            return oc

    raise Exception("אכיפה נכשלה על מקטע")


# ─────────────────────────────────────────────
# FULL ANALYSIS PIPELINE
# ─────────────────────────────────────────────
async def run_pipeline(job_id, transcript, context, model, api_key):
    update_progress(job_id, 5)

    # --- שלב 1: קידוד פתוח עם אכיפה ---
    open_coding = []
    total = len(transcript)

    for i, seg in enumerate(transcript, start=1):
        oc = await enforce_segment(seg["text"], context, model, api_key)
        open_coding.append({"segment_id": seg["id"], "codes": oc})

        update_progress(job_id, int(5 + (i / total) * 40))

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

    # --- שלב 5: Interpretation ---
    update_progress(job_id, 85)
    interp_raw = await model_call(
        p_interpret(themes, quotes), model, api_key
    )
    interpretations = json.loads(interp_raw)

    # --- שלב 6: Matrix ---
    update_progress(job_id, 95)
    matrix_raw = await model_call(
        p_matrix(themes, quotes, interpretations),
        model,
        api_key
    )
    matrix = json.loads(matrix_raw)

    # תוצאה סופית
    return {
        "openCoding": open_coding,
        "axial": axial,
        "themes": themes,
        "quotes": quotes,
        "interpretations": interpretations,
        "matrix": matrix
    }


# ─────────────────────────────────────────────
# BACKGROUND TASK RUNNER
# ─────────────────────────────────────────────
async def background_analysis(job_id, transcript, context, model, api_key):
    try:
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
        req.api_key
    )

    return {"job_id": job_id, "status": "processing"}


@app.get("/agent/status/{job_id}")
async def status(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        return {"error": "job not found"}
    return job


# ─────────────────────────────────────────────
# הפעלה מקומית
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000)
