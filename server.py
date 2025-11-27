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

app = FastAPI(title="Qualitative Analysis Agent - Braun & Clarke")

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
    model: str = "gemini"
    api_key: str | None = None


# ============================================================
# JSON EXTRACTION HELPER
# ============================================================
def extract_json(raw_text: str):
    """מחלץ JSON מתגובת המודל"""
    text = raw_text.strip()
    
    code_block_pattern = r'```(?:json|JSON)?\s*\n?([\s\S]*?)\n?```'
    match = re.search(code_block_pattern, text)
    if match:
        text = match.group(1).strip()
    
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
    async with httpx.AsyncClient(timeout=180) as client:
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
        "max_tokens": 8000,
        "messages": [{"role": "user", "content": prompt}],
    }
    async with httpx.AsyncClient(timeout=180) as client:
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
        "generationConfig": {"temperature": 0.3, "maxOutputTokens": 8000}
    }
    async with httpx.AsyncClient(timeout=180) as client:
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
    else:
        return await call_gemini(prompt, api_key)


# ============================================================
# TRANSCRIPT EXTRACTION & FILTERING
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


def filter_intro_segments(segments: list) -> list:
    """
    מסנן מקטעים שאינם חלק מתוכן הראיון עצמו:
    - הצגות עצמיות
    - פתיחות טכניות
    - סגירות
    """
    intro_patterns = [
        r'שלום.*שמי',
        r'היי.*קוראים לי',
        r'אני.*המראיין',
        r'אני.*החוקר',
        r'תודה שהסכמת',
        r'תודה שבאת',
        r'נתחיל.*הראיון',
        r'לפני שנתחיל',
        r'אני מקליט',
        r'האם אפשר להקליט',
        r'בוא נתחיל',
        r'תספר.*על עצמך',
        r'ספר.*קצת על עצמך',
        r'תציג.*את עצמך',
        r'מה השם שלך',
        r'בן כמה אתה',
        r'מאיפה אתה',
        r'תודה רבה על.*הראיון',
        r'זהו.*סיימנו',
        r'תודה על הזמן',
        r'נסיים כאן',
    ]
    
    filtered = []
    skip_intro = True  # מדלג על פתיחות
    
    for seg in segments:
        text = seg.get("text", "").strip()
        if not text:
            continue
            
        # בדיקה אם זה מקטע פתיחה/סגירה
        is_intro = any(re.search(p, text, re.IGNORECASE) for p in intro_patterns)
        
        # אם מצאנו תוכן ממשי, מפסיקים לדלג
        if not is_intro and len(text) > 20:
            skip_intro = False
        
        if skip_intro and is_intro:
            continue
            
        # מדלג על משפטים קצרים מאוד בהתחלה
        if skip_intro and len(text) < 15:
            continue
            
        filtered.append(seg)
    
    return filtered


# ============================================================
# PROMPTS - Braun & Clarke (משופרים)
# ============================================================
def p_filter_content(segments):
    """פרומפט לסינון תוכן לא רלוונטי"""
    text = json.dumps(segments, ensure_ascii=False)
    return f"""
אתה מנתח מחקר איכותני מומחה.

קיבלת תמלול ראיון. עליך לסנן ולהחזיר רק את המקטעים שמכילים תוכן מהותי לניתוח.

יש להסיר:
1. הצגות עצמיות של המרואיין או המראיין
2. שאלות טכניות (הקלטה, זמן וכו')
3. פתיחות וסגירות פורמליות
4. small talk לא רלוונטי
5. משפטים קצרים כמו "כן", "אוקיי", "בסדר" בלבד

יש לשמור:
1. תשובות מהותיות של המרואיין
2. שאלות תוכניות של המראיין
3. סיפורים, חוויות, דעות
4. כל תוכן בעל ערך לניתוח

התמלול:
{text}

החזר JSON בלבד - מערך של המקטעים הרלוונטיים בלבד, באותו פורמט:
[{{"speaker": "...", "text": "...", "start": ..., "end": ...}}]
"""


def p_initial_coding(segments):
    """קידוד פתוח"""
    text = "\n\n".join([
        f"[מקטע {i+1}] {s.get('speaker', 'דובר')}:\n\"{s.get('text', '')}\""
        for i, s in enumerate(segments)
    ])
    return f"""
אתה מנתח מחקר איכותני מומחה בשיטת Braun & Clarke.

בצע קידוד פתוח (Initial Coding) על התמלול הבא.
עבור כל מקטע, זהה קודים סמנטיים המתארים את התוכן, הרגשות, והמשמעויות.

כללים:
- קודים צריכים להיות תמציתיים (2-5 מילים)
- קודים צריכים לשקף את תוכן הדברים, לא רק לתאר
- זהה גם קודים תיאוריים וגם קודים פרשניים
- התעלם ממקטעים טכניים או חסרי תוכן

התמלול:
{text}

החזר JSON בלבד:
[
  {{
    "segment_index": 1,
    "speaker": "מרואיין",
    "text": "הטקסט המקורי",
    "codes": ["קוד 1", "קוד 2", "קוד 3"]
  }}
]
"""


def p_initial_themes(codes):
    """יצירת תימות ראשוניות"""
    return f"""
אתה מנתח מחקר איכותני מומחה בשיטת Braun & Clarke.

על בסיס הקודים הבאים, צור תימות ראשוניות (Initial Themes).
קבץ קודים דומים לתימות רחבות יותר.

כללים:
- כל תימה צריכה לכלול לפחות 2-3 קודים קשורים
- תימות צריכות להיות משמעותיות ולא טריוויאליות
- הוסף תיאור קצר לכל תימה
- ציין את הקודים השייכים לכל תימה

הקודים:
{json.dumps(codes, ensure_ascii=False, indent=2)}

החזר JSON בלבד:
[
  {{
    "theme": "שם התימה",
    "description": "תיאור קצר של התימה",
    "codes": ["קוד 1", "קוד 2"],
    "frequency": 5
  }}
]
"""


def p_review_themes(themes, codes):
    """סקירה וחידוד תימות"""
    return f"""
אתה מנתח מחקר איכותני מומחה בשיטת Braun & Clarke.

בצע סקירה וחידוד של התימות (Reviewing Themes).

בדוק:
1. האם כל תימה מגובשת פנימית?
2. האם התימות נבדלות זו מזו מספיק?
3. האם יש תימות שכדאי למזג?
4. האם יש תימות שכדאי לפצל?
5. האם כל הקודים משויכים נכון?

תימות נוכחיות:
{json.dumps(themes, ensure_ascii=False, indent=2)}

קודים מקוריים:
{json.dumps(codes, ensure_ascii=False, indent=2)}

החזר JSON עם תימות מחודדות:
[
  {{
    "theme": "שם מחודד",
    "description": "תיאור מעודכן",
    "codes": ["קוד 1", "קוד 2"],
    "rationale": "הסבר קצר לשינויים שבוצעו"
  }}
]
"""


def p_define_themes(themes, segments):
    """הגדרה סופית של תימות עם ציטוטים"""
    return f"""
אתה מנתח מחקר איכותני מומחה בשיטת Braun & Clarke.

הגדר את התימות הסופיות (Define & Name Themes).
לכל תימה:
1. תן שם ברור וממצה
2. כתוב הגדרה אקדמית מדויקת
3. הוסף 2-4 ציטוטים תומכים מהתמלול
4. הסבר את המשמעות התיאורטית

תימות:
{json.dumps(themes, ensure_ascii=False, indent=2)}

תמלול מקורי:
{json.dumps(segments, ensure_ascii=False, indent=2)}

החזר JSON בלבד:
[
  {{
    "theme": "שם סופי אקדמי",
    "definition": "הגדרה אקדמית מפורטת של התימה (2-3 משפטים)",
    "codes": ["קוד 1", "קוד 2"],
    "quotes": [
      {{
        "text": "ציטוט מדויק מהתמלול",
        "speaker": "מרואיין",
        "context": "הקשר קצר"
      }}
    ],
    "theoretical_significance": "משמעות תיאורטית"
  }}
]
"""


def p_report(themes_defined, segments):
    """דו"ח מחקרי מסכם"""
    return f"""
אתה מנתח מחקר איכותני מומחה בשיטת Braun & Clarke.

כתוב דו"ח ממצאים מחקרי מקצועי.

הדו"ח צריך לכלול:
1. תקציר מנהלים (Executive Summary)
2. סקירת התימות המרכזיות
3. ממצאים עיקריים לכל תימה
4. קשרים בין התימות
5. השלכות מחקריות ומעשיות
6. מגבלות הניתוח

מספר מקטעים שנותחו: {len(segments)}

תימות מוגדרות:
{json.dumps(themes_defined, ensure_ascii=False, indent=2)}

החזר JSON בלבד:
{{
  "executive_summary": "תקציר מנהלים (3-4 משפטים)",
  "methodology_note": "הערה מתודולוגית קצרה",
  "themes_overview": [
    {{
      "theme": "שם התימה",
      "key_findings": "ממצאים עיקריים (2-3 משפטים)",
      "prevalence": "שכיחות: גבוהה/בינונית/נמוכה",
      "significance": "משמעות הממצא"
    }}
  ],
  "theme_relationships": "תיאור הקשרים בין התימות",
  "implications": {{
    "theoretical": "השלכות תיאורטיות",
    "practical": "השלכות מעשיות"
  }},
  "limitations": "מגבלות הניתוח",
  "future_research": "כיווני מחקר עתידיים"
}}
"""


def p_matrix(themes_defined, codes):
    """מטריצת תימות מסכמת"""
    return f"""
צור מטריצת תימות אקדמית מסכמת.

לכל תימה כלול:
- שם התימה
- מספר הקודים
- מספר הציטוטים
- תובנה מרכזית
- רמת שכיחות

תימות:
{json.dumps(themes_defined, ensure_ascii=False, indent=2)}

סה"כ קודים:
{json.dumps(codes, ensure_ascii=False, indent=2)}

החזר JSON בלבד:
[
  {{
    "theme": "שם התימה",
    "codes_count": 5,
    "quotes_count": 3,
    "prevalence": "גבוהה/בינונית/נמוכה",
    "key_insight": "תובנה מרכזית במשפט אחד",
    "sub_themes": ["תת-תימה 1", "תת-תימה 2"]
  }}
]
"""


# ============================================================
# PIPELINE - Full Braun & Clarke
# ============================================================
async def run_pipeline(job_id, transcript_raw, ctx, model, api_key):
    try:
        update_progress(job_id, 2)
        segments = extract_transcript(transcript_raw)
        
        if not segments:
            raise Exception("לא נמצאו מקטעים בתמלול")
        
        logger.info(f"Starting analysis with {len(segments)} segments")

        # שלב 0: סינון ראשוני בצד שרת
        update_progress(job_id, 5)
        segments = filter_intro_segments(segments)
        logger.info(f"After local filtering: {len(segments)} segments")

        # שלב 1: סינון תוכן באמצעות AI
        update_progress(job_id, 8)
        logger.info("Step 0: AI Content Filtering")
        raw = await model_call(p_filter_content(segments), model, api_key)
        filtered_segments = extract_json(raw)
        if filtered_segments and len(filtered_segments) > 0:
            segments = filtered_segments
            logger.info(f"After AI filtering: {len(segments)} segments")
        update_progress(job_id, 15)

        # שלב 2: קידוד פתוח
        logger.info("Step 1: Initial Coding")
        raw = await model_call(p_initial_coding(segments), model, api_key)
        codes = extract_json(raw)
        if not codes:
            codes = [{"segment_index": i+1, "text": s.get("text", ""), "codes": ["קוד כללי"]} 
                     for i, s in enumerate(segments)]
        update_progress(job_id, 30)

        # שלב 3: תימות ראשוניות
        logger.info("Step 2: Initial Themes")
        raw = await model_call(p_initial_themes(codes), model, api_key)
        themes_initial = extract_json(raw)
        if not themes_initial:
            themes_initial = [{"theme": "תימה כללית", "codes": [], "description": ""}]
        update_progress(job_id, 45)

        # שלב 4: סקירת תימות
        logger.info("Step 3: Reviewing Themes")
        raw = await model_call(p_review_themes(themes_initial, codes), model, api_key)
        themes_reviewed = extract_json(raw)
        if not themes_reviewed:
            themes_reviewed = themes_initial
        update_progress(job_id, 58)

        # שלב 5: הגדרת תימות סופיות
        logger.info("Step 4: Define & Name Themes")
        raw = await model_call(p_define_themes(themes_reviewed, segments), model, api_key)
        themes_defined = extract_json(raw)
        if not themes_defined:
            themes_defined = [{"theme": t.get("theme", "תימה"), "definition": "", "codes": t.get("codes", []), "quotes": []} 
                             for t in themes_reviewed]
        update_progress(job_id, 72)

        # שלב 6: דו"ח סופי
        logger.info("Step 5: Final Report")
        raw = await model_call(p_report(themes_defined, segments), model, api_key)
        report = extract_json(raw)
        if not report:
            report = {
                "executive_summary": "לא נוצר סיכום",
                "themes_overview": [],
                "implications": {"theoretical": "", "practical": ""},
                "limitations": ""
            }
        update_progress(job_id, 88)

        # שלב 7: מטריצה
        logger.info("Step 6: Matrix")
        raw = await model_call(p_matrix(themes_defined, codes), model, api_key)
        matrix = extract_json(raw)
        if not matrix:
            matrix = []
        update_progress(job_id, 98)

        logger.info("Pipeline completed successfully")
        
        # סטטיסטיקות
        stats = {
            "total_segments": len(segments),
            "total_codes": sum(len(c.get("codes", [])) for c in codes),
            "total_themes": len(themes_defined),
            "analysis_model": model
        }
        
        return {
            "statistics": stats,
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
    
    return {"job_id": job_id, "status": "running"}


@app.get("/agent/status/{job_id}")
async def status(job_id: str):
    return JOBS.get(job_id, {"error": "not found"})


@app.get("/ping")
async def ping():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000)
