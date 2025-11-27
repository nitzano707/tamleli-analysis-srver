import uuid
import json
import asyncio
import re
import logging
import time
from typing import Union, Optional, List, Dict

from fastapi import FastAPI, BackgroundTasks, Response
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
    "gemini": {
        "api_name": "gemini-2.0-flash",
        "display_name": "Gemini 2.0 Flash",
        "provider": "google",
        "url_template": "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}",
        "max_retries": 5,
        "base_delay": 30,
        "retry_delay": 60,
        "max_segments_per_chunk": 35,  # ~35 segments per API call
    },
    "gemini-pro": {
        "api_name": "gemini-1.5-pro",
        "display_name": "Gemini 1.5 Pro",
        "provider": "google",
        "url_template": "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}",
        "max_retries": 5,
        "base_delay": 30,
        "retry_delay": 60,
        "max_segments_per_chunk": 40,
    },
    "gpt": {
        "api_name": "gpt-4o",
        "display_name": "GPT-4o",
        "provider": "openai",
        "url": "https://api.openai.com/v1/chat/completions",
        "max_retries": 5,
        "base_delay": 5,
        "retry_delay": 30,
        "max_segments_per_chunk": 30,
    },
    "gpt-mini": {
        "api_name": "gpt-4o-mini",
        "display_name": "GPT-4o Mini",
        "provider": "openai",
        "url": "https://api.openai.com/v1/chat/completions",
        "max_retries": 5,
        "base_delay": 3,
        "retry_delay": 20,
        "max_segments_per_chunk": 35,
    },
    "claude": {
        "api_name": "claude-sonnet-4-20250514",
        "display_name": "Claude Sonnet 4",
        "provider": "anthropic",
        "url": "https://api.anthropic.com/v1/messages",
        "max_retries": 4,
        "base_delay": 5,
        "retry_delay": 30,
        "max_segments_per_chunk": 35,
    },
    "claude-haiku": {
        "api_name": "claude-haiku-4-20250514",
        "display_name": "Claude Haiku 4",
        "provider": "anthropic",
        "url": "https://api.anthropic.com/v1/messages",
        "max_retries": 4,
        "base_delay": 3,
        "retry_delay": 20,
        "max_segments_per_chunk": 40,
    },
}

# ============================================================
# JOB STORE
# ============================================================
JOBS = {}


def create_job(job_id):
    """יוצר job חדש או מאפס job קיים"""
    JOBS[job_id] = {
        "status": "running",
        "progress": 0,
        "step_info": "",
        "wait_until": None,
        "result": None,
        "error": None
    }


def update_progress(job_id, v, step_info=""):
    if job_id not in JOBS:
        create_job(job_id)
    JOBS[job_id]["progress"] = int(v)
    JOBS[job_id]["step_info"] = step_info
    JOBS[job_id]["wait_until"] = None


def set_waiting(job_id, seconds):
    if job_id not in JOBS:
        create_job(job_id)
    JOBS[job_id]["status"] = "waiting"
    JOBS[job_id]["wait_until"] = time.time() + seconds
    JOBS[job_id]["step_info"] = f"ממתין {seconds} שניות בגלל הגבלת קריאות..."


def set_running(job_id):
    if job_id not in JOBS:
        create_job(job_id)
    JOBS[job_id]["status"] = "running"
    JOBS[job_id]["wait_until"] = None


def set_result(job_id, r):
    if job_id not in JOBS:
        create_job(job_id)
    JOBS[job_id]["status"] = "done"
    JOBS[job_id]["progress"] = 100
    JOBS[job_id]["result"] = r
    JOBS[job_id]["wait_until"] = None


def set_error(job_id, err):
    if job_id not in JOBS:
        create_job(job_id)
    JOBS[job_id]["status"] = "error"
    JOBS[job_id]["error"] = str(err)
    JOBS[job_id]["wait_until"] = None


# ============================================================
# REQUEST MODELS
# ============================================================
class ResearchContext(BaseModel):
    research_question: str = ""
    study_context: str = ""
    participant_info: str = ""
    additional_notes: str = ""


class AnalysisRequest(BaseModel):
    transcript: dict | list
    research_context: ResearchContext | dict = {}
    model: str = "gemini"
    api_key: str | None = None


# ============================================================
# JSON EXTRACTION HELPER
# ============================================================
def extract_json(raw_text: str):
    if not raw_text:
        return None
        
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
    
    if text.startswith('{'):
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
        try:
            fixed = text.replace("'", '"')
            return json.loads(fixed)
        except:
            pass
        return None


# ============================================================
# TIME FORMATTING
# ============================================================
def format_timestamp(seconds: float) -> str:
    """המרת שניות לפורמט MM:SS או HH:MM:SS"""
    if seconds is None:
        return ""
    try:
        seconds = float(seconds)
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        
        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{secs:02d}"
        return f"{minutes:02d}:{secs:02d}"
    except:
        return ""


# ============================================================
# SMART RATE LIMIT HANDLER
# ============================================================
async def smart_api_call(func, job_id, config, *args, **kwargs):
    max_retries = config.get("max_retries", 5)
    base_delay = config.get("base_delay", 10)
    retry_delay = config.get("retry_delay", 60)
    
    for attempt in range(max_retries):
        try:
            set_running(job_id)
            result = await func(*args, **kwargs)
            
            if base_delay > 0:
                logger.info(f"Waiting {base_delay}s before next call...")
                await asyncio.sleep(base_delay)
            
            return result
            
        except httpx.HTTPStatusError as e:
            status_code = e.response.status_code
            
            if status_code == 429:
                wait_time = retry_delay * (2 ** attempt)
                retry_after = e.response.headers.get("Retry-After")
                if retry_after:
                    try:
                        wait_time = max(wait_time, int(retry_after))
                    except ValueError:
                        pass
                wait_time = min(wait_time, 300)
                
                logger.warning(f"Rate limited! Waiting {wait_time}s (attempt {attempt + 1}/{max_retries})")
                set_waiting(job_id, wait_time)
                await asyncio.sleep(wait_time)
                
            elif status_code == 503:
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
        "max_tokens": 16000,
    }
    async with httpx.AsyncClient(timeout=300) as client:
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
        "max_tokens": 16000,
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
            "maxOutputTokens": 16000
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
# TRANSCRIPT EXTRACTION - תומך במספר פורמטים
# ============================================================
def extract_transcript(raw) -> List[Dict]:
    """
    מחלץ מקטעי תמלול מפורמטים שונים:
    - Tamleli Pro format: segments + versionHistory
    - Simple format: list of segments
    - Legacy format: versions array
    """
    segments = []
    
    if isinstance(raw, list):
        segments = raw
    elif isinstance(raw, dict):
        # Tamleli Pro format - use main segments (latest version)
        if "segments" in raw:
            segments = raw["segments"]
        # Legacy format with versions
        elif "versions" in raw:
            versions = raw["versions"]
            if versions:
                # Get latest version
                latest = sorted(versions, key=lambda v: v.get("saved_at", ""), reverse=True)[0]
                segments = latest.get("segments", latest.get("segments_snapshot", []))
        # versionHistory format
        elif "versionHistory" in raw:
            vh = raw["versionHistory"]
            if vh:
                latest = sorted(vh, key=lambda v: v.get("saved_at", ""), reverse=True)[0]
                segments = latest.get("segments_snapshot", [])
        # Utterances format
        elif "utterances" in raw:
            segments = raw["utterances"]
    
    # Ensure each segment has required fields
    processed = []
    for i, seg in enumerate(segments):
        processed.append({
            "index": i,
            "speaker": seg.get("speaker", f"דובר {i+1}"),
            "text": seg.get("text", ""),
            "start": seg.get("start"),
            "end": seg.get("end"),
        })
    
    return processed


def filter_intro_segments(segments: list) -> list:
    """מסנן מקטעי פתיחה/סגירה לא רלוונטיים"""
    intro_patterns = [
        r'שלום.*שמי', r'היי.*קוראים לי', r'אני.*המראיין',
        r'תודה שהסכמת', r'נתחיל.*הראיון', r'לפני שנתחיל',
        r'אני מקליט', r'בוא נתחיל', r'תספר.*על עצמך',
        r'תודה רבה על.*הראיון', r'זהו.*סיימנו', r'נסיים כאן',
        r'כל השמות.*בדויים', r'במסגרת.*התואר',
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


def chunk_segments(segments: List[Dict], max_per_chunk: int) -> List[List[Dict]]:
    """
    מחלק את התמלול לחלקים עם הקשר חופף
    כל חלק מקבל 2-3 מקטעים מהחלק הקודם לשמירת הקשר
    """
    if len(segments) <= max_per_chunk:
        return [segments]
    
    chunks = []
    overlap = 3  # מקטעי חפיפה בין chunks
    
    i = 0
    while i < len(segments):
        end = min(i + max_per_chunk, len(segments))
        chunk = segments[i:end]
        chunks.append(chunk)
        i = end - overlap  # חפיפה
        if i < 0:
            i = end
    
    return chunks


# ============================================================
# PROMPTS WITH QUOTE ENFORCEMENT
# ============================================================

def format_research_context(ctx: dict) -> str:
    parts = []
    if ctx.get("research_question"):
        parts.append(f"שאלת המחקר: {ctx['research_question']}")
    if ctx.get("study_context"):
        parts.append(f"הקשר המחקר: {ctx['study_context']}")
    if ctx.get("additional_notes"):
        parts.append(f"הערות נוספות: {ctx['additional_notes']}")
    return "\n".join(parts) if parts else ""


def p_coding_and_themes(segments: List[Dict], research_ctx: dict, chunk_info: str = ""):
    """פרומפט לקידוד - עם אכיפת ציטוטים וחותמות זמן"""
    
    context_section = format_research_context(research_ctx)
    
    # Format segments with timestamps
    text = "\n\n".join([
        f"[מקטע {s.get('index', i)+1}] [{format_timestamp(s.get('start'))} - {format_timestamp(s.get('end'))}] {s.get('speaker', 'דובר')}:\n\"{s.get('text', '')}\""
        for i, s in enumerate(segments)
    ])
    
    return f"""
אתה מנתח מחקר איכותני מומחה בשיטת Braun & Clarke (2006).

{"=" * 50}
הקשר המחקר:
{context_section if context_section else "לא סופק הקשר ספציפי - בצע ניתוח כללי"}
{chunk_info}
{"=" * 50}

קיבלת תמלול ראיון. בצע ניתוח מקיף תוך התמקדות בשאלת המחקר.

## הנחיות קריטיות:

### 1. קידוד פתוח
- צור קודים סמנטיים (2-5 מילים) לכל מקטע משמעותי
- קודים צריכים להיות בעברית ותמציתיים

### 2. ⚠️ חובה: ציטוטים לכל קוד!
- **לכל קוד חייבים להיות לפחות 2-3 ציטוטים מהתמלול**
- ציטוטים צריכים להיות **מדויקים** - העתק את הטקסט המקורי
- כל ציטוט חייב לכלול **חותמת זמן** (start, end)
- ניתוח ללא ציטוטים הוא ניתוח לא תקין!

### 3. תימות ראשוניות
- קבץ קודים דומים ל-3-6 תימות
- כל תימה צריכה שם ממצה ותיאור קצר

## התמלול:
{text}

## החזר JSON בלבד במבנה הבא:
{{
  "intro_info": {{
    "participant_description": "תיאור קצר של המרואיין (אם יש)",
    "interview_context": "הקשר הראיון",
    "excluded_content": "תוכן שהושמט"
  }},
  "codes": [
    {{
      "segment_index": 1,
      "text": "הטקסט המלא של המקטע",
      "start": 0.0,
      "end": 10.5,
      "speaker": "דובר",
      "codes": ["קוד 1", "קוד 2"],
      "quotes": [
        {{
          "text": "ציטוט מדויק מהמקטע",
          "start": 2.5,
          "end": 8.0,
          "timestamp": "00:02"
        }}
      ]
    }}
  ],
  "initial_themes": [
    {{
      "theme": "שם התימה המלא",
      "description": "תיאור מפורט",
      "codes": ["קוד 1", "קוד 2"],
      "relevance_to_research": "קשר לשאלת המחקר"
    }}
  ]
}}
"""


def p_define_and_report(themes, codes, segments, research_ctx: dict):
    """פרומפט להגדרת תימות ודו"ח - עם אכיפת ציטוטים"""
    
    context_section = format_research_context(research_ctx)
    participant_info = research_ctx.get("participant_info", "")
    
    # Prepare quotes from coded segments
    all_quotes = []
    for c in codes:
        for q in c.get("quotes", []):
            all_quotes.append({
                "text": q.get("text", ""),
                "start": q.get("start"),
                "end": q.get("end"),
                "timestamp": q.get("timestamp", format_timestamp(q.get("start"))),
                "speaker": c.get("speaker", ""),
                "code": c.get("codes", [])[0] if c.get("codes") else ""
            })
    
    return f"""
אתה מנתח מחקר איכותני מומחה בשיטת Braun & Clarke (2006).

{"=" * 50}
הקשר המחקר:
{context_section if context_section else "לא סופק הקשר ספציפי"}

מידע על המשתתפים:
{participant_info if participant_info else "לא סופק מידע"}
{"=" * 50}

## תימות ראשוניות לעיבוד:
{json.dumps(themes, ensure_ascii=False, indent=2)}

## קודים וציטוטים שזוהו:
{json.dumps(codes[:40], ensure_ascii=False, indent=2)}

## משימות:

### 1. הגדר תימות סופיות
לכל תימה:
- שם אקדמי מלא וממצה (לא לקצר!)
- הגדרה מפורטת (2-3 משפטים)
- רשימת קודים משויכים (מופרדים בפסיקים)

### 2. ⚠️ חובה: ציטוטים עשירים!
- **כל קוד חייב לכלול לפחות 2-4 ציטוטים**
- כל ציטוט עם חותמת זמן (timestamp)
- העתק ציטוטים **מדויקים** מהתמלול
- ניתוח עשיר בציטוטים = ניתוח איכותי!

### 3. דו"ח ממצאים
- פסקת מבוא עם רקע על המרואיין
- תקציר מנהלים שעונה על שאלת המחקר

## החזר JSON בלבד:
{{
  "intro_paragraph": "פסקת מבוא עם רקע על המרואיין והקשר הראיון",
  "themes_defined": [
    {{
      "theme": "שם התימה המלא והאקדמי",
      "definition": "הגדרה מפורטת (2-3 משפטים)",
      "relevance_to_research": "כיצד התימה עונה על שאלת המחקר",
      "codes_with_quotes": [
        {{
          "code": "שם הקוד",
          "quotes": [
            {{"text": "ציטוט מדויק 1", "speaker": "מרואיין", "timestamp": "05:23", "start": 323.0, "end": 330.0}},
            {{"text": "ציטוט מדויק 2", "speaker": "מרואיין", "timestamp": "12:45", "start": 765.0, "end": 772.0}},
            {{"text": "ציטוט מדויק 3", "speaker": "מרואיין", "timestamp": "18:30", "start": 1110.0, "end": 1118.0}}
          ]
        }}
      ],
      "theoretical_significance": "משמעות תיאורטית"
    }}
  ],
  "report": {{
    "executive_summary": "תקציר מנהלים שעונה על שאלת המחקר (4-5 משפטים)",
    "methodology_note": "הערה מתודולוגית",
    "themes_overview": [
      {{
        "theme": "שם מלא",
        "key_findings": "ממצאים עיקריים",
        "prevalence": "גבוהה/בינונית/נמוכה",
        "quotes_count": 8
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
      "theme": "שם התימה המלא",
      "definition": "הגדרה מלאה",
      "codes": [
        {{
          "code": "שם הקוד",
          "quotes": [
            {{"text": "ציטוט", "timestamp": "05:23", "start": 323.0, "end": 330.0, "speaker": "מרואיין"}}
          ]
        }}
      ],
      "prevalence": "גבוהה",
      "key_insight": "תובנה מרכזית"
    }}
  ]
}}
"""


# ============================================================
# PIPELINE WITH CHUNKING
# ============================================================
async def run_pipeline(job_id, transcript_raw, research_ctx, model, api_key):
    try:
        model_config = MODEL_CONFIG.get(model, MODEL_CONFIG["gemini"])
        model_display_name = model_config.get("display_name", model)
        max_segments = model_config.get("max_segments_per_chunk", 35)
        
        if isinstance(research_ctx, dict):
            ctx = research_ctx
        else:
            ctx = {}
        
        update_progress(job_id, 5, "טוען תמלול...")
        segments = extract_transcript(transcript_raw)
        
        if not segments:
            raise Exception("לא נמצאו מקטעים בתמלול")
        
        original_count = len(segments)
        logger.info(f"Loaded {original_count} segments")

        update_progress(job_id, 10, "מסנן תוכן...")
        segments = filter_intro_segments(segments)
        filtered_count = len(segments)
        logger.info(f"After filtering: {filtered_count} segments")

        if filtered_count == 0:
            raise Exception("לא נשאר תוכן לניתוח לאחר סינון")

        # Check if chunking is needed
        chunks = chunk_segments(segments, max_segments)
        num_chunks = len(chunks)
        logger.info(f"Split into {num_chunks} chunks (max {max_segments} segments each)")

        all_codes = []
        all_themes = []

        # ========== שלב 1: קידוד (עם חלוקה לחלקים אם צריך) ==========
        for chunk_idx, chunk in enumerate(chunks):
            chunk_info = ""
            if num_chunks > 1:
                chunk_info = f"\n[חלק {chunk_idx + 1} מתוך {num_chunks}]"
                if chunk_idx > 0:
                    chunk_info += "\n(המשך מחלק קודם - שמור על קוהרנטיות)"
            
            progress = 15 + (chunk_idx / num_chunks) * 30
            update_progress(job_id, progress, f"שלב 1: קידוד (חלק {chunk_idx + 1}/{num_chunks})...")
            
            raw1 = await model_call(
                p_coding_and_themes(chunk, ctx, chunk_info),
                model, api_key, job_id
            )
            result1 = extract_json(raw1)
            
            if result1:
                chunk_codes = result1.get("codes", [])
                chunk_themes = result1.get("initial_themes", [])
                all_codes.extend(chunk_codes)
                
                # Merge themes intelligently
                for new_theme in chunk_themes:
                    existing = next((t for t in all_themes if t.get("theme") == new_theme.get("theme")), None)
                    if existing:
                        existing["codes"] = list(set(existing.get("codes", []) + new_theme.get("codes", [])))
                    else:
                        all_themes.append(new_theme)
        
        if not all_codes:
            logger.warning("No codes found, creating defaults")
            all_codes = [{
                "segment_index": i,
                "text": s.get("text", "")[:200],
                "start": s.get("start"),
                "end": s.get("end"),
                "codes": ["קוד כללי"],
                "quotes": [{"text": s.get("text", "")[:100], "start": s.get("start"), "end": s.get("end"), "timestamp": format_timestamp(s.get("start"))}],
                "speaker": s.get("speaker", "")
            } for i, s in enumerate(segments[:20])]
        
        if not all_themes:
            all_themes = [{"theme": "תימה כללית", "description": "תימה שזוהתה", "codes": ["קוד כללי"]}]
        
        logger.info(f"Total: {len(all_codes)} coded segments, {len(all_themes)} themes")
        update_progress(job_id, 50, "שלב 1 הושלם...")

        # ========== שלב 2: הגדרה ודו"ח ==========
        update_progress(job_id, 55, "שלב 2: הגדרת תימות ודו\"ח...")
        
        raw2 = await model_call(
            p_define_and_report(all_themes, all_codes, segments, ctx),
            model, api_key, job_id
        )
        result2 = extract_json(raw2)
        
        if not result2:
            logger.error("Failed to parse step 2, creating defaults")
            result2 = create_default_result(all_themes, all_codes)

        update_progress(job_id, 95, "מסכם תוצאות...")

        # Build final result
        themes_defined = result2.get("themes_defined", [])
        report = result2.get("report", {})
        matrix = result2.get("matrix", [])
        intro_paragraph = result2.get("intro_paragraph", "")
        
        # Ensure quotes exist
        themes_defined = ensure_quotes_in_themes(themes_defined, all_codes)
        matrix = ensure_quotes_in_matrix(matrix, themes_defined)
        
        if not report.get("executive_summary"):
            report["executive_summary"] = f"ניתוח תמטי של {filtered_count} מקטעים העלה {len(themes_defined)} תימות מרכזיות."
        
        # Count quotes
        total_quotes = sum(
            sum(len(cq.get("quotes", [])) for cq in t.get("codes_with_quotes", []))
            for t in themes_defined
        )
        
        stats = {
            "total_segments": filtered_count,
            "original_segments": original_count,
            "total_codes": len(all_codes),
            "total_themes": len(themes_defined),
            "total_quotes": total_quotes,
            "analysis_model": model,
            "model_display_name": model_display_name,
            "chunks_processed": num_chunks,
            "analysis_date": time.strftime("%Y-%m-%d %H:%M")
        }
        
        logger.info(f"Pipeline completed: {stats}")
        
        return {
            "statistics": stats,
            "research_context": ctx,
            "intro_paragraph": intro_paragraph,
            "clean_transcript": segments,
            "codes": all_codes,
            "themes_initial": all_themes,
            "themes_defined": themes_defined,
            "report": report,
            "matrix": matrix
        }

    except Exception as e:
        logger.error(f"Pipeline error: {e}")
        raise


def create_default_result(themes, codes):
    """יוצר תוצאת ברירת מחדל אם הפרסינג נכשל"""
    return {
        "intro_paragraph": "",
        "themes_defined": [
            {
                "theme": t.get("theme", "תימה"),
                "definition": t.get("description", ""),
                "relevance_to_research": "",
                "codes_with_quotes": [
                    {"code": c, "quotes": []}
                    for c in t.get("codes", [])
                ],
                "theoretical_significance": ""
            }
            for t in themes
        ],
        "report": {
            "executive_summary": f"זוהו {len(themes)} תימות מרכזיות.",
            "methodology_note": "ניתוח תמטי לפי Braun & Clarke (2006)",
            "themes_overview": [],
            "theme_relationships": "",
            "implications": {"theoretical": "", "practical": ""},
            "limitations": "",
            "future_research": ""
        },
        "matrix": []
    }


def ensure_quotes_in_themes(themes_defined, all_codes):
    """מוודא שיש ציטוטים לכל קוד בתימות"""
    # Build a map of code -> quotes from all_codes
    code_quotes_map = {}
    for c in all_codes:
        for code_name in c.get("codes", []):
            if code_name not in code_quotes_map:
                code_quotes_map[code_name] = []
            # Add quotes from this segment
            for q in c.get("quotes", []):
                code_quotes_map[code_name].append({
                    "text": q.get("text", c.get("text", "")[:100]),
                    "speaker": c.get("speaker", ""),
                    "timestamp": q.get("timestamp", format_timestamp(c.get("start"))),
                    "start": q.get("start", c.get("start")),
                    "end": q.get("end", c.get("end"))
                })
            # If no quotes, use segment text
            if not c.get("quotes"):
                code_quotes_map[code_name].append({
                    "text": c.get("text", "")[:150],
                    "speaker": c.get("speaker", ""),
                    "timestamp": format_timestamp(c.get("start")),
                    "start": c.get("start"),
                    "end": c.get("end")
                })
    
    # Ensure themes have quotes
    for theme in themes_defined:
        codes_with_quotes = theme.get("codes_with_quotes", [])
        if not codes_with_quotes:
            # Create from theme's codes list
            theme["codes_with_quotes"] = [
                {"code": code, "quotes": code_quotes_map.get(code, [])}
                for code in theme.get("codes", [])
            ]
        else:
            # Ensure each code has quotes
            for cq in codes_with_quotes:
                if not cq.get("quotes"):
                    cq["quotes"] = code_quotes_map.get(cq.get("code"), [])
    
    return themes_defined


def ensure_quotes_in_matrix(matrix, themes_defined):
    """מוודא שיש ציטוטים במטריצה"""
    if not matrix:
        # Build from themes_defined
        matrix = []
        for theme in themes_defined:
            matrix.append({
                "theme": theme.get("theme", ""),
                "definition": theme.get("definition", ""),
                "codes": theme.get("codes_with_quotes", []),
                "prevalence": "בינונית",
                "key_insight": theme.get("relevance_to_research", "")
            })
    else:
        # Ensure codes have quotes
        theme_map = {t.get("theme"): t for t in themes_defined}
        for m in matrix:
            if not m.get("codes") or all(not c.get("quotes") for c in m.get("codes", [])):
                theme = theme_map.get(m.get("theme"))
                if theme:
                    m["codes"] = theme.get("codes_with_quotes", [])
    
    return matrix


# ============================================================
# BACKGROUND RUNNER
# ============================================================
async def background_run(job_id, transcript, ctx, model, api_key):
    logger.info(f"Background task started for job {job_id}")
    try:
        # Validate inputs
        if not transcript:
            raise Exception("תמלול חסר")
        if not api_key:
            raise Exception("API Key חסר")
        if not model:
            model = "gemini"
        
        result = await run_pipeline(job_id, transcript, ctx, model, api_key)
        set_result(job_id, result)
        logger.info(f"Job {job_id} completed successfully")
    except Exception as e:
        logger.error(f"Job {job_id} failed: {e}", exc_info=True)
        set_error(job_id, str(e))


# ============================================================
# API ENDPOINTS
# ============================================================
@app.post("/agent/analyze")
async def analyze(req: AnalysisRequest):
    try:
        # Validate request
        if not req.transcript:
            return {"error": "תמלול חסר", "status": "error"}
        
        if not req.api_key:
            return {"error": "API Key חסר", "status": "error"}
        
        job_id = str(uuid.uuid4())
        create_job(job_id)
        
        ctx = req.research_context
        if hasattr(ctx, 'dict'):
            ctx = ctx.dict()
        elif not isinstance(ctx, dict):
            ctx = {}
        
        asyncio.create_task(
            background_run(job_id, req.transcript, ctx, req.model, req.api_key)
        )
        
        return {"job_id": job_id, "status": "running"}
    except Exception as e:
        logger.error(f"Error in analyze endpoint: {e}")
        return {"error": str(e), "status": "error"}


@app.get("/agent/status/{job_id}")
async def status(job_id: str):
    try:
        job = JOBS.get(job_id, {"error": "not found", "status": "error"})
        
        if job.get("wait_until"):
            remaining = max(0, int(job["wait_until"] - time.time()))
            job["wait_remaining"] = remaining
        else:
            job["wait_remaining"] = 0
        
        return job
    except Exception as e:
        logger.error(f"Error getting status for {job_id}: {e}")
        return {"error": str(e), "status": "error"}


@app.get("/ping")
async def ping():
    return {"status": "ok"}
@app.head("/ping")
async def ping_head():
    # UptimeRobot שולח בקשת HEAD – מחזירים 200 בלי body
    return Response(status_code=200)



@app.get("/models")
async def get_models():
    return {
        "models": [
            {"id": mid, "name": cfg["api_name"], "display_name": cfg["display_name"], "provider": cfg["provider"]}
            for mid, cfg in MODEL_CONFIG.items()
        ]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000)
