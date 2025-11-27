import uuid
import json
import asyncio
import re
import logging
import time
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
# MODEL CONFIGURATIONS (נובמבר 2025)
# ============================================================
MODEL_CONFIG = {
    # Google Gemini
    "gemini": {
        "api_name": "gemini-2.0-flash",
        "provider": "google",
        "url_template": "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}",
        "max_retries": 5,
        "base_delay": 30,  # המתנה בסיסית בין קריאות
        "retry_delay": 60,  # המתנה לאחר 429
    },
    "gemini-pro": {
        "api_name": "gemini-1.5-pro",
        "provider": "google",
        "url_template": "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}",
        "max_retries": 5,
        "base_delay": 30,
        "retry_delay": 60,
    },
    "gemini-flash": {
        "api_name": "gemini-1.5-flash",
        "provider": "google",
        "url_template": "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}",
        "max_retries": 5,
        "base_delay": 20,
        "retry_delay": 45,
    },
    # OpenAI
    "gpt": {
        "api_name": "gpt-4o",
        "provider": "openai",
        "url": "https://api.openai.com/v1/chat/completions",
        "max_retries": 5,
        "base_delay": 5,
        "retry_delay": 30,
    },
    "gpt-mini": {
        "api_name": "gpt-4o-mini",
        "provider": "openai",
        "url": "https://api.openai.com/v1/chat/completions",
        "max_retries": 5,
        "base_delay": 3,
        "retry_delay": 20,
    },
    # Anthropic Claude
    "claude": {
        "api_name": "claude-sonnet-4-20250514",
        "provider": "anthropic",
        "url": "https://api.anthropic.com/v1/messages",
        "max_retries": 4,
        "base_delay": 5,
        "retry_delay": 30,
    },
    "claude-haiku": {
        "api_name": "claude-haiku-4-20250514",
        "provider": "anthropic",
        "url": "https://api.anthropic.com/v1/messages",
        "max_retries": 4,
        "base_delay": 3,
        "retry_delay": 20,
    },
}

# ============================================================
# JOB STORE - משופר עם מידע על המתנה
# ============================================================
JOBS = {}


def create_job(job_id):
    JOBS[job_id] = {
        "status": "running",
        "progress": 0,
        "step_info": "",
        "wait_until": None,  # timestamp עד מתי ממתינים
        "result": None,
        "error": None
    }


def update_progress(job_id, v, step_info=""):
    if job_id in JOBS:
        JOBS[job_id]["progress"] = int(v)
        JOBS[job_id]["step_info"] = step_info
        JOBS[job_id]["wait_until"] = None


def set_waiting(job_id, seconds):
    """מעדכן שהמערכת ממתינה בגלל rate limit"""
    if job_id in JOBS:
        JOBS[job_id]["status"] = "waiting"
        JOBS[job_id]["wait_until"] = time.time() + seconds
        JOBS[job_id]["step_info"] = f"ממתין {seconds} שניות בגלל הגבלת קריאות..."


def set_running(job_id):
    if job_id in JOBS:
        JOBS[job_id]["status"] = "running"
        JOBS[job_id]["wait_until"] = None


def set_result(job_id, r):
    JOBS[job_id]["status"] = "done"
    JOBS[job_id]["progress"] = 100
    JOBS[job_id]["result"] = r
    JOBS[job_id]["wait_until"] = None


def set_error(job_id, err):
    JOBS[job_id]["status"] = "error"
    JOBS[job_id]["error"] = str(err)
    JOBS[job_id]["wait_until"] = None


# ============================================================
# REQUEST MODEL
# ============================================================
class AnalysisRequest(BaseModel):
    transcript: dict | list
    research_context: dict = {}
    model: str = "gemini"
    api_key: str | None = None


# ============================================================
# JSON EXTRACTION HELPER - משופר
# ============================================================
def extract_json(raw_text: str):
    """מחלץ JSON מתגובת המודל - עם טיפול משופר"""
    if not raw_text:
        return None
        
    text = raw_text.strip()
    
    # נסה למצוא JSON בתוך code block
    code_block_pattern = r'```(?:json|JSON)?\s*\n?([\s\S]*?)\n?```'
    match = re.search(code_block_pattern, text)
    if match:
        text = match.group(1).strip()
    
    # אם לא מתחיל ב-{ או [, חפש את ההתחלה
    if not text.startswith('{') and not text.startswith('['):
        for i, char in enumerate(text):
            if char in '{[':
                text = text[i:]
                break
    
    # נסה לחתוך עד סוף ה-JSON
    if text.startswith('{'):
        # מצא את ה-} האחרון
        last_brace = text.rfind('}')
        if last_brace > 0:
            text = text[:last_brace + 1]
    elif text.startswith('['):
        last_bracket = text.rfind(']')
        if last_bracket > 0:
            text = text[:last_bracket + 1]
    
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON: {e}")
        logger.error(f"Text was: {text[:500]}")
        
        # נסיון אחרון - תיקון JSON נפוץ
        try:
            # החלף גרשיים בודדים בכפולים
            fixed = text.replace("'", '"')
            return json.loads(fixed)
        except:
            pass
        
        return None


# ============================================================
# SMART RATE LIMIT HANDLER
# ============================================================
async def smart_api_call(func, job_id, config, *args, **kwargs):
    """
    קריאת API חכמה עם:
    - Exponential backoff
    - עדכון UI בזמן המתנה
    - ניסיונות חוזרים חכמים
    """
    max_retries = config.get("max_retries", 5)
    base_delay = config.get("base_delay", 10)
    retry_delay = config.get("retry_delay", 60)
    
    for attempt in range(max_retries):
        try:
            set_running(job_id)
            result = await func(*args, **kwargs)
            
            # המתנה בסיסית בין קריאות למניעת rate limit
            if base_delay > 0:
                logger.info(f"Waiting {base_delay}s before next call...")
                await asyncio.sleep(base_delay)
            
            return result
            
        except httpx.HTTPStatusError as e:
            status_code = e.response.status_code
            
            if status_code == 429:
                # Rate limit - המתנה ארוכה
                wait_time = retry_delay * (2 ** attempt)  # Exponential backoff
                
                # בדוק Retry-After header
                retry_after = e.response.headers.get("Retry-After")
                if retry_after:
                    try:
                        wait_time = max(wait_time, int(retry_after))
                    except ValueError:
                        pass
                
                # הגבל ל-5 דקות מקסימום
                wait_time = min(wait_time, 300)
                
                logger.warning(f"Rate limited! Waiting {wait_time}s (attempt {attempt + 1}/{max_retries})")
                set_waiting(job_id, wait_time)
                await asyncio.sleep(wait_time)
                
            elif status_code == 503:
                # Server overload
                wait_time = 30 * (attempt + 1)
                logger.warning(f"Server overload, waiting {wait_time}s")
                set_waiting(job_id, wait_time)
                await asyncio.sleep(wait_time)
                
            elif status_code in [401, 403]:
                raise Exception(f"שגיאת הרשאה ({status_code}): בדוק שמפתח ה-API תקין")
                
            elif status_code == 404:
                raise Exception(f"המודל לא נמצא ({status_code}): ייתכן שאין לך גישה למודל זה")
                
            else:
                raise Exception(f"שגיאת API ({status_code}): {str(e)}")
                
        except httpx.TimeoutException:
            wait_time = 10 * (attempt + 1)
            logger.warning(f"Timeout, waiting {wait_time}s")
            if attempt < max_retries - 1:
                await asyncio.sleep(wait_time)
            else:
                raise Exception("תם הזמן המוקצב לתגובה מהשרת. נסה שוב.")
                
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            logger.error(f"Error on attempt {attempt + 1}: {e}")
            await asyncio.sleep(5)
    
    raise Exception(f"נכשל לאחר {max_retries} ניסיונות")


# ============================================================
# AI PROVIDERS
# ============================================================
async def call_gpt(prompt: str, api_key: str, model_id: str = "gpt") -> str:
    config = MODEL_CONFIG.get(model_id, MODEL_CONFIG["gpt"])
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    body = {
        "model": config["api_name"],
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,
        "max_tokens": 16000,  # הגדלנו!
    }
    async with httpx.AsyncClient(timeout=300) as client:  # timeout ארוך יותר
        r = await client.post(config["url"], headers=headers, json=body)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]


async def call_claude(prompt: str, api_key: str, model_id: str = "claude") -> str:
    config = MODEL_CONFIG.get(model_id, MODEL_CONFIG["claude"])
    
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    body = {
        "model": config["api_name"],
        "max_tokens": 16000,  # הגדלנו!
        "messages": [{"role": "user", "content": prompt}],
    }
    async with httpx.AsyncClient(timeout=300) as client:
        r = await client.post(config["url"], headers=headers, json=body)
        r.raise_for_status()
        return r.json()["content"][0]["text"]


async def call_gemini(prompt: str, api_key: str, model_id: str = "gemini") -> str:
    config = MODEL_CONFIG.get(model_id, MODEL_CONFIG["gemini"])
    
    url = config["url_template"].format(
        model=config["api_name"],
        api_key=api_key
    )
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.3,
            "maxOutputTokens": 16000  # הגדלנו!
        }
    }
    async with httpx.AsyncClient(timeout=300) as client:
        r = await client.post(url, json=payload)
        r.raise_for_status()
        response = r.json()
        
        if "candidates" in response and len(response["candidates"]) > 0:
            return response["candidates"][0]["content"]["parts"][0]["text"]
        elif "error" in response:
            raise Exception(f"Gemini API error: {response['error'].get('message', 'Unknown error')}")
        else:
            raise Exception(f"Unexpected Gemini response: {json.dumps(response)[:200]}")


async def model_call(prompt: str, model: str, api_key: str, job_id: str) -> str:
    """קריאה למודל AI עם ניהול חכם של rate limits"""
    if not api_key:
        raise Exception("חסר API Key")
    
    config = MODEL_CONFIG.get(model, MODEL_CONFIG.get("gemini"))
    provider = config["provider"]
    
    logger.info(f"Calling {model} (provider: {provider})")
    
    if provider == "openai":
        call_func = call_gpt
    elif provider == "anthropic":
        call_func = call_claude
    else:
        call_func = call_gemini
    
    return await smart_api_call(
        call_func,
        job_id,
        config,
        prompt, api_key, model
    )


# ============================================================
# TRANSCRIPT EXTRACTION & FILTERING
# ============================================================
def extract_transcript(raw):
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
    intro_patterns = [
        r'שלום.*שמי', r'היי.*קוראים לי', r'אני.*המראיין',
        r'תודה שהסכמת', r'נתחיל.*הראיון', r'לפני שנתחיל',
        r'אני מקליט', r'בוא נתחיל', r'תספר.*על עצמך',
        r'תודה רבה על.*הראיון', r'זהו.*סיימנו', r'נסיים כאן',
    ]
    
    filtered = []
    skip_intro = True
    
    for seg in segments:
        text = seg.get("text", "").strip()
        if not text or len(text) < 5:
            continue
        is_intro = any(re.search(p, text, re.IGNORECASE) for p in intro_patterns)
        if not is_intro and len(text) > 20:
            skip_intro = False
        if skip_intro and (is_intro or len(text) < 15):
            continue
        filtered.append(seg)
    
    return filtered


# ============================================================
# COMBINED PROMPTS - פחות קריאות, יותר תוכן!
# ============================================================

def p_coding_and_themes(segments):
    """פרומפט משולב: סינון + קידוד + תימות ראשוניות - קריאה אחת!"""
    text = "\n\n".join([
        f"[{i+1}] {s.get('speaker', 'דובר')}: \"{s.get('text', '')}\""
        for i, s in enumerate(segments[:50])  # הגבלה ל-50 מקטעים
    ])
    
    return f"""
אתה מנתח מחקר איכותני מומחה בשיטת Braun & Clarke (2006).

קיבלת תמלול ראיון. בצע ניתוח מקיף בשלב אחד:

## שלב 1: זיהוי מקטעים רלוונטיים
סנן והתעלם מ: הצגות עצמיות, שאלות טכניות, פתיחות/סגירות, small talk.

## שלב 2: קידוד פתוח (Initial Coding)
לכל מקטע רלוונטי, צור קודים סמנטיים (2-5 מילים כל קוד).

## שלב 3: זיהוי תימות ראשוניות
קבץ את הקודים לתימות משמעותיות (3-6 תימות).

התמלול:
{text}

החזר JSON בלבד במבנה הבא:
{{
  "codes": [
    {{
      "segment_index": 1,
      "text": "טקסט מקוצר של המקטע",
      "codes": ["קוד 1", "קוד 2"],
      "speaker": "דובר"
    }}
  ],
  "initial_themes": [
    {{
      "theme": "שם התימה",
      "description": "תיאור קצר",
      "codes": ["קוד 1", "קוד 2", "קוד 3"]
    }}
  ]
}}
"""


def p_define_and_report(themes, codes, segments):
    """פרומפט משולב: הגדרת תימות + דו"ח + מטריצה - קריאה אחת!"""
    
    # הכן ציטוטים אפשריים
    quotes_text = "\n".join([
        f"[{i+1}] {s.get('speaker', 'דובר')}: \"{s.get('text', '')[:200]}\""
        for i, s in enumerate(segments[:30])
    ])
    
    return f"""
אתה מנתח מחקר איכותני מומחה בשיטת Braun & Clarke (2006).

בצע ניתוח סופי מקיף:

## תימות לעיבוד:
{json.dumps(themes, ensure_ascii=False, indent=2)}

## קודים שזוהו:
{json.dumps(codes[:30], ensure_ascii=False, indent=2)}

## ציטוטים מהתמלול (בחר מתאימים):
{quotes_text}

## משימות:

### 1. הגדר תימות סופיות
לכל תימה: שם אקדמי, הגדרה מפורטת, ציטוטים תומכים מהתמלול.

### 2. כתוב דו"ח ממצאים
תקציר מנהלים, סקירת תימות, קשרים, השלכות.

### 3. צור מטריצת סיכום
לכל תימה: מספר קודים, ציטוטים, תובנה מרכזית.

החזר JSON בלבד:
{{
  "themes_defined": [
    {{
      "theme": "שם התימה האקדמי",
      "definition": "הגדרה מפורטת (2-3 משפטים)",
      "codes": ["קוד 1", "קוד 2"],
      "quotes": [
        {{
          "text": "ציטוט מדויק",
          "speaker": "מרואיין",
          "context": "הקשר"
        }}
      ],
      "theoretical_significance": "משמעות תיאורטית"
    }}
  ],
  "report": {{
    "executive_summary": "תקציר מנהלים מקיף (4-5 משפטים)",
    "methodology_note": "הערה מתודולוגית",
    "themes_overview": [
      {{
        "theme": "שם",
        "key_findings": "ממצאים עיקריים",
        "prevalence": "גבוהה/בינונית/נמוכה",
        "significance": "משמעות"
      }}
    ],
    "theme_relationships": "קשרים בין תימות",
    "implications": {{
      "theoretical": "השלכות תיאורטיות",
      "practical": "השלכות מעשיות"
    }},
    "limitations": "מגבלות",
    "future_research": "כיווני מחקר"
  }},
  "matrix": [
    {{
      "theme": "שם",
      "codes_count": 5,
      "quotes_count": 3,
      "prevalence": "גבוהה",
      "key_insight": "תובנה מרכזית",
      "sub_themes": ["תת-תימה"]
    }}
  ]
}}
"""


# ============================================================
# OPTIMIZED PIPELINE - רק 2 קריאות API!
# ============================================================
async def run_pipeline(job_id, transcript_raw, ctx, model, api_key):
    try:
        update_progress(job_id, 5, "טוען תמלול...")
        segments = extract_transcript(transcript_raw)
        
        if not segments:
            raise Exception("לא נמצאו מקטעים בתמלול")
        
        logger.info(f"Starting optimized analysis with {len(segments)} segments")

        # סינון מקומי
        update_progress(job_id, 10, "מסנן תוכן לא רלוונטי...")
        segments = filter_intro_segments(segments)
        logger.info(f"After filtering: {len(segments)} segments")

        if len(segments) == 0:
            raise Exception("לא נשאר תוכן לניתוח לאחר סינון")

        # ========== קריאה 1: קידוד + תימות ראשוניות ==========
        update_progress(job_id, 20, "שלב 1/2: קידוד וזיהוי תימות ראשוניות...")
        logger.info("Step 1: Coding and Initial Themes (combined)")
        
        raw1 = await model_call(p_coding_and_themes(segments), model, api_key, job_id)
        result1 = extract_json(raw1)
        
        if not result1:
            logger.error("Failed to parse step 1 result")
            # יצירת ברירת מחדל
            result1 = {
                "codes": [{"segment_index": i+1, "text": s.get("text", "")[:100], "codes": ["קוד כללי"], "speaker": s.get("speaker", "")} 
                         for i, s in enumerate(segments[:20])],
                "initial_themes": [{"theme": "תימה כללית", "description": "תימה שזוהתה בניתוח", "codes": ["קוד כללי"]}]
            }
        
        codes = result1.get("codes", [])
        themes_initial = result1.get("initial_themes", [])
        
        logger.info(f"Found {len(codes)} coded segments, {len(themes_initial)} initial themes")
        update_progress(job_id, 50, "שלב 1 הושלם. ממתין לפני שלב 2...")

        # ========== קריאה 2: הגדרה + דו"ח + מטריצה ==========
        update_progress(job_id, 55, "שלב 2/2: הגדרת תימות, דו\"ח ומטריצה...")
        logger.info("Step 2: Define, Report and Matrix (combined)")
        
        raw2 = await model_call(p_define_and_report(themes_initial, codes, segments), model, api_key, job_id)
        result2 = extract_json(raw2)
        
        if not result2:
            logger.error("Failed to parse step 2 result")
            # יצירת ברירת מחדל
            result2 = {
                "themes_defined": [
                    {
                        "theme": t.get("theme", "תימה"),
                        "definition": t.get("description", "תימה שזוהתה בניתוח"),
                        "codes": t.get("codes", []),
                        "quotes": [],
                        "theoretical_significance": ""
                    }
                    for t in themes_initial
                ],
                "report": {
                    "executive_summary": f"נותחו {len(segments)} מקטעים וזוהו {len(themes_initial)} תימות מרכזיות.",
                    "methodology_note": "ניתוח תמטי לפי Braun & Clarke (2006)",
                    "themes_overview": [
                        {"theme": t.get("theme", ""), "key_findings": t.get("description", ""), "prevalence": "בינונית", "significance": ""}
                        for t in themes_initial
                    ],
                    "theme_relationships": "נמצאו קשרים בין התימות",
                    "implications": {"theoretical": "", "practical": ""},
                    "limitations": "מגבלות הניתוח האוטומטי",
                    "future_research": ""
                },
                "matrix": [
                    {"theme": t.get("theme", ""), "codes_count": len(t.get("codes", [])), "quotes_count": 0, "prevalence": "בינונית", "key_insight": t.get("description", ""), "sub_themes": []}
                    for t in themes_initial
                ]
            }

        update_progress(job_id, 95, "מסכם תוצאות...")

        # הרכב תוצאה סופית
        themes_defined = result2.get("themes_defined", [])
        report = result2.get("report", {})
        matrix = result2.get("matrix", [])
        
        # וודא שיש executive_summary
        if not report.get("executive_summary"):
            report["executive_summary"] = f"ניתוח תמטי של {len(segments)} מקטעים העלה {len(themes_defined)} תימות מרכזיות."
        
        stats = {
            "total_segments": len(segments),
            "total_codes": sum(len(c.get("codes", [])) for c in codes),
            "total_themes": len(themes_defined),
            "analysis_model": model,
            "api_calls": 2  # רק 2 קריאות!
        }
        
        logger.info(f"Pipeline completed: {stats}")
        
        return {
            "statistics": stats,
            "clean_transcript": segments,
            "codes": codes,
            "themes_initial": themes_initial,
            "themes_reviewed": themes_initial,  # אותו דבר בגרסה המצומצמת
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
    job = JOBS.get(job_id, {"error": "not found"})
    
    # חשב זמן המתנה נותר
    if job.get("wait_until"):
        remaining = max(0, int(job["wait_until"] - time.time()))
        job["wait_remaining"] = remaining
    else:
        job["wait_remaining"] = 0
    
    return job


@app.get("/ping")
async def ping():
    return {"status": "ok"}


@app.get("/models")
async def get_models():
    return {
        "models": [
            {"id": mid, "name": cfg["api_name"], "provider": cfg["provider"]}
            for mid, cfg in MODEL_CONFIG.items()
        ]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000)
