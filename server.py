import uuid
import json
import asyncio
import re
import logging
import time
from typing import Union, Optional

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
        "base_delay": 25,
        "retry_delay": 60,
        "max_segments_per_chunk": 35,
    },
    "gemini-pro": {
        "api_name": "gemini-1.5-pro",
        "display_name": "Gemini 1.5 Pro",
        "provider": "google",
        "url_template": "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}",
        "max_retries": 5,
        "base_delay": 25,
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
        "max_segments_per_chunk": 25,
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
        "max_segments_per_chunk": 30,
    },
}

# ============================================================
# JOB STORE
# ============================================================
JOBS = {}


def create_job(job_id):
    JOBS[job_id] = {
        "status": "running",
        "progress": 0,
        "step_info": "",
        "wait_until": None,
        "result": None,
        "error": None
    }


def update_progress(job_id, v, step_info=""):
    if job_id in JOBS:
        JOBS[job_id]["progress"] = int(v)
        JOBS[job_id]["step_info"] = step_info
        JOBS[job_id]["wait_until"] = None
        JOBS[job_id]["status"] = "running"


def set_waiting(job_id, seconds):
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
    url = config["url_template"].format(model=config["api_name"], api_key=api_key)
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.3, "maxOutputTokens": 16000}
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
    return await smart_api_call(call_func, job_id, config, prompt, api_key, model)


# ============================================================
# TRANSCRIPT EXTRACTION & HELPERS
# ============================================================
def format_timestamp(seconds: float) -> str:
    if seconds is None:
        return ""
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{mins:02d}:{secs:02d}"


def extract_transcript(raw):
    if isinstance(raw, list):
        return raw
    if isinstance(raw, dict):
        if "versions" in raw and isinstance(raw["versions"], list) and len(raw["versions"]) > 0:
            latest_version = raw["versions"][-1]
            return latest_version.get("segments", [])
        if "segments" in raw:
            return raw["segments"]
        if "utterances" in raw:
            return raw["utterances"]
    return []


def filter_intro_segments(segments: list) -> tuple:
    intro_patterns = [
        r'שלום.*שמי', r'היי.*קוראים לי', r'אני.*המראיין',
        r'תודה שהסכמת', r'נתחיל.*הראיון', r'לפני שנתחיל',
        r'אני מקליט', r'בוא נתחיל', r'תספר.*על עצמך',
        r'תודה רבה על.*הראיון', r'זהו.*סיימנו', r'נסיים כאן',
        r'ביי\s*ביי', r'להתראות', r'תודה.*עזרת',
    ]
    filtered = []
    intro_segments = []
    skip_intro = True
    
    for seg in segments:
        text = seg.get("text", "").strip()
        if not text or len(text) < 5:
            continue
        is_intro = any(re.search(p, text, re.IGNORECASE) for p in intro_patterns)
        if is_intro:
            intro_segments.append(seg)
            continue
        if not is_intro and len(text) > 20:
            skip_intro = False
        if skip_intro and len(text) < 15:
            intro_segments.append(seg)
            continue
        filtered.append(seg)
    return filtered, intro_segments


def chunk_segments(segments: list, max_per_chunk: int) -> list:
    if len(segments) <= max_per_chunk:
        return [segments]
    chunks = []
    for i in range(0, len(segments), max_per_chunk):
        chunks.append(segments[i:i + max_per_chunk])
    return chunks


# ============================================================
# PROMPTS - 6 STAGES OF BRAUN & CLARKE
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


def format_segments_with_timestamps(segments: list) -> str:
    lines = []
    for i, s in enumerate(segments):
        start_time = format_timestamp(s.get("start", 0))
        end_time = format_timestamp(s.get("end", 0))
        speaker = s.get("speaker", "דובר")
        text = s.get("text", "")
        lines.append(f"[{i+1}] [{start_time}-{end_time}] {speaker}: \"{text}\"")
    return "\n\n".join(lines)


def p_stage1_2_coding(segments: list, research_ctx: dict, chunk_num: int = 1, total_chunks: int = 1) -> str:
    context_section = format_research_context(research_ctx)
    text = format_segments_with_timestamps(segments)
    chunk_note = f"\n*** חלק {chunk_num} מתוך {total_chunks} ***\n" if total_chunks > 1 else ""
    
    return f"""אתה מנתח מחקר איכותני מומחה בשיטת Braun & Clarke (2006).
{chunk_note}
{"=" * 60}
הקשר המחקר:
{context_section if context_section else "לא סופק הקשר ספציפי"}
{"=" * 60}

## משימה: שלבים 1-2 - היכרות וקידוד ראשוני

### הנחיות קריטיות:
1. קרא את התמלול בעיון והתמקד בתוכן הרלוונטי לשאלת המחקר
2. צור קודים סמנטיים (2-5 מילים בעברית)
3. **חובה: לכל קוד לפחות 2-3 ציטוטים מדויקים עם חותמות זמן!**
4. ציטוטים עשירים = ניתוח איכותי טוב

### התמלול:
{text}

### החזר JSON בלבד:
{{
  "coded_segments": [
    {{
      "segment_index": 1,
      "timestamp": "MM:SS-MM:SS",
      "speaker": "שם",
      "original_text": "הטקסט המלא",
      "codes": [
        {{
          "code": "שם הקוד",
          "quotes": [
            {{"text": "ציטוט מדויק 1", "timestamp": "MM:SS"}},
            {{"text": "ציטוט מדויק 2", "timestamp": "MM:SS"}},
            {{"text": "ציטוט מדויק 3", "timestamp": "MM:SS"}}
          ]
        }}
      ]
    }}
  ],
  "all_codes": ["רשימת כל הקודים"]
}}

**זכור: ציטוטים רבים ומדויקים הם המפתח לניתוח טוב!**
"""


def p_stage3_4_themes(codes_data: list, research_ctx: dict) -> str:
    context_section = format_research_context(research_ctx)
    
    codes_summary = []
    for seg in codes_data:
        for code_item in seg.get("codes", []):
            codes_summary.append({
                "code": code_item.get("code", ""),
                "quotes": code_item.get("quotes", []),
                "speaker": seg.get("speaker", "")
            })
    
    return f"""אתה מנתח מחקר איכותני מומחה בשיטת Braun & Clarke (2006).

{"=" * 60}
הקשר המחקר:
{context_section if context_section else "לא סופק הקשר"}
{"=" * 60}

## משימה: שלבים 3-4 - חיפוש וסקירת תימות

### שלב 3: קבץ קודים לתימות ראשוניות (3-7 תימות)
### שלב 4: סקור - בדוק קוהרנטיות, מזג/פצל לפי הצורך

### הנחיות קריטיות:
1. **העבר את כל הציטוטים מהקודים לתימות!**
2. כל תימה צריכה להיות קוהרנטית פנימית
3. הבחנה ברורה בין תימות

### קודים (כולל ציטוטים):
{json.dumps(codes_summary[:50], ensure_ascii=False, indent=2)}

### החזר JSON:
{{
  "themes": [
    {{
      "theme_name": "שם התימה",
      "description": "תיאור",
      "codes_with_quotes": [
        {{
          "code": "שם הקוד",
          "quotes": [
            {{"text": "ציטוט", "timestamp": "MM:SS", "speaker": "דובר"}}
          ]
        }}
      ],
      "coherence_note": "הסבר למה התימה קוהרנטית"
    }}
  ]
}}
"""


def p_stage5_6_report(themes_data: list, research_ctx: dict, participant_info: str) -> str:
    context_section = format_research_context(research_ctx)
    
    return f"""אתה מנתח מחקר איכותני מומחה בשיטת Braun & Clarke (2006).

{"=" * 60}
הקשר המחקר:
{context_section if context_section else "לא סופק הקשר"}

מידע על משתתפים (למבוא):
{participant_info if participant_info else "לא סופק"}
{"=" * 60}

## משימה: שלבים 5-6 - הגדרת תימות ודו"ח סופי

### שלב 5: הגדר כל תימה סופית עם שם אקדמי והגדרה מדויקת
### שלב 6: כתוב דו"ח מקיף

### הנחיות קריטיות:
1. **שמור את כל הציטוטים עם חותמות הזמן!**
2. הגדרות אקדמיות ברורות
3. תקציר שעונה על שאלת המחקר

### תימות לעיבוד:
{json.dumps(themes_data, ensure_ascii=False, indent=2)}

### החזר JSON מלא:
{{
  "intro_paragraph": "פסקת מבוא עם רקע על המרואיין והקשר",
  
  "themes_defined": [
    {{
      "theme": "שם התימה הסופי",
      "definition": "הגדרה אקדמית (2-3 משפטים)",
      "codes_with_quotes": [
        {{
          "code": "שם הקוד",
          "quotes": [
            {{"text": "ציטוט מדויק", "timestamp": "MM:SS", "speaker": "דובר"}}
          ]
        }}
      ],
      "relevance_to_research": "קשר לשאלת המחקר",
      "theoretical_significance": "משמעות תיאורטית"
    }}
  ],
  
  "report": {{
    "executive_summary": "תקציר מנהלים (5-7 משפטים)",
    "methodology_note": "ניתוח תמטי - Braun & Clarke (2006)",
    "key_findings": ["ממצא 1", "ממצא 2"],
    "theme_relationships": "קשרים בין תימות",
    "implications": {{"theoretical": "...", "practical": "..."}},
    "limitations": "מגבלות",
    "future_research": "המלצות"
  }},
  
  "matrix": [
    {{
      "theme": "שם",
      "definition": "הגדרה",
      "codes_list": ["קוד1", "קוד2"],
      "total_quotes": 10,
      "prevalence": "גבוהה/בינונית/נמוכה"
    }}
  ]
}}

**חשוב: וודא שכל הציטוטים נשמרים עם חותמות זמן!**
"""


# ============================================================
# MAIN PIPELINE
# ============================================================
async def run_pipeline(job_id, transcript_raw, research_ctx, model, api_key):
    try:
        model_config = MODEL_CONFIG.get(model, MODEL_CONFIG["gemini"])
        model_display_name = model_config.get("display_name", model)
        max_segments = model_config.get("max_segments_per_chunk", 35)
        
        ctx = research_ctx if isinstance(research_ctx, dict) else {}
        
        update_progress(job_id, 2, "טוען תמלול...")
        segments = extract_transcript(transcript_raw)
        
        if not segments:
            raise Exception("לא נמצאו מקטעים בתמלול")
        
        logger.info(f"Loaded {len(segments)} segments")
        
        update_progress(job_id, 5, "מסנן...")
        filtered_segments, intro_segments = filter_intro_segments(segments)
        
        if not filtered_segments:
            raise Exception("לא נשאר תוכן לניתוח")
        
        logger.info(f"Filtered: {len(filtered_segments)} segments")
        
        chunks = chunk_segments(filtered_segments, max_segments)
        total_chunks = len(chunks)
        logger.info(f"Chunks: {total_chunks}")
        
        # === קריאה 1: שלבים 1-2 ===
        all_coded = []
        all_codes = []
        
        for idx, chunk in enumerate(chunks):
            progress = 10 + (idx * 25 // total_chunks)
            update_progress(job_id, progress, f"שלבים 1-2: קידוד ({idx+1}/{total_chunks})...")
            
            raw = await model_call(p_stage1_2_coding(chunk, ctx, idx+1, total_chunks), model, api_key, job_id)
            result = extract_json(raw)
            
            if result:
                all_coded.extend(result.get("coded_segments", []))
                all_codes.extend(result.get("all_codes", []))
        
        all_codes = list(set(all_codes))
        logger.info(f"Coded: {len(all_coded)} segments, {len(all_codes)} codes")
        update_progress(job_id, 40, "שלבים 1-2 הושלמו...")
        
        # === קריאה 2: שלבים 3-4 ===
        update_progress(job_id, 45, "שלבים 3-4: תימות וסקירה...")
        
        raw2 = await model_call(p_stage3_4_themes(all_coded, ctx), model, api_key, job_id)
        result2 = extract_json(raw2)
        
        themes = result2.get("themes", []) if result2 else []
        logger.info(f"Themes: {len(themes)}")
        update_progress(job_id, 65, "שלבים 3-4 הושלמו...")
        
        # === קריאה 3: שלבים 5-6 ===
        update_progress(job_id, 70, "שלבים 5-6: הגדרות ודו\"ח...")
        
        raw3 = await model_call(p_stage5_6_report(themes, ctx, ctx.get("participant_info", "")), model, api_key, job_id)
        result3 = extract_json(raw3)
        
        intro_paragraph = ""
        themes_defined = []
        report = {}
        matrix = []
        
        if result3:
            intro_paragraph = result3.get("intro_paragraph", "")
            themes_defined = result3.get("themes_defined", [])
            report = result3.get("report", {})
            matrix = result3.get("matrix", [])
        
        if not report.get("executive_summary"):
            report["executive_summary"] = f"ניתוח תמטי של {len(filtered_segments)} מקטעים העלה {len(themes_defined)} תימות."
        
        update_progress(job_id, 95, "מסכם...")
        
        total_quotes = sum(
            len(q.get("quotes", []))
            for t in themes_defined
            for q in t.get("codes_with_quotes", [])
        )
        
        stats = {
            "total_segments": len(filtered_segments),
            "total_codes": len(all_codes),
            "total_themes": len(themes_defined),
            "total_quotes": total_quotes,
            "analysis_model": model,
            "model_display_name": model_display_name,
            "api_calls": 2 + total_chunks,
            "analysis_date": time.strftime("%Y-%m-%d %H:%M"),
            "methodology": "Braun & Clarke (2006) - 6 שלבים"
        }
        
        logger.info(f"Done: {stats}")
        
        return {
            "statistics": stats,
            "research_context": ctx,
            "intro_paragraph": intro_paragraph,
            "clean_transcript": filtered_segments,
            "coded_segments": all_coded,
            "all_codes": all_codes,
            "themes": themes,
            "themes_defined": themes_defined,
            "report": report,
            "matrix": matrix
        }

    except Exception as e:
        logger.error(f"Pipeline error: {e}")
        raise


async def background_run(job_id, transcript, ctx, model, api_key):
    logger.info(f"Job {job_id} started")
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
    ctx = req.research_context
    if hasattr(ctx, 'dict'):
        ctx = ctx.dict()
    elif not isinstance(ctx, dict):
        ctx = {}
    asyncio.create_task(background_run(job_id, req.transcript, ctx, req.model, req.api_key))
    return {"job_id": job_id, "status": "running"}


@app.get("/agent/status/{job_id}")
async def status(job_id: str):
    job = JOBS.get(job_id, {"error": "not found"})
    if job.get("wait_until"):
        job["wait_remaining"] = max(0, int(job["wait_until"] - time.time()))
    else:
        job["wait_remaining"] = 0
    return job


@app.get("/ping")
async def ping():
    return {"status": "ok"}


@app.head("/ping")
async def ping_head():
    return Response(status_code=200)


@app.get("/models")
async def get_models():
    return {"models": [
        {"id": mid, "name": cfg["api_name"], "display_name": cfg["display_name"], "provider": cfg["provider"]}
        for mid, cfg in MODEL_CONFIG.items()
    ]}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000)
