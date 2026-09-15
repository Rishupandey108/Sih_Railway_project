"""
api.py
=======
Railway Maintenance AI System — FastAPI Backend + Gemini LLM Copilot

Exposes the full ML system as a REST API and provides a natural-language
copilot powered by Google Gemini.

Endpoints:
  GET  /health                 — Health check
  GET  /assets/risk            — All assets with ML risk scores
  POST /predict/risk           — Predict risk for a new asset
  GET  /schedule/optimized     — Current optimized maintenance schedule
  POST /schedule/regenerate    — Re-run data synthesis + ML + optimizer
  GET  /coa/blocks             — Approved COA blocks
  GET  /eda/summary            — Statistical summary of all three systems
  POST /copilot/query          — Gemini LLM natural language query

CORS is enabled for development (all origins). Lock down in production.

Usage:
    uvicorn api:app --host 0.0.0.0 --port 8000 --reload
    # → Swagger UI at http://localhost:8000/docs
"""

import json
import os
import subprocess
import sys
import warnings
from datetime import date
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd
import google.generativeai as genai
import uvicorn

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

warnings.filterwarnings("ignore")
load_dotenv()   # reads GEMINI_API_KEY from .env file

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(__file__)
DATA_DIR   = os.path.join(BASE_DIR, "data")
MODEL_DIR  = os.path.join(BASE_DIR, "models")
STATIC_DIR = os.path.join(BASE_DIR, "static")

# ── Gemini setup ──────────────────────────────────────────────────────────────
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    _gemini_model = genai.GenerativeModel("gemini-3.6-flash")
else:
    _gemini_model = None


# ═══════════════════════════════════════════════════════════════════════════════
# FastAPI Application
# ═══════════════════════════════════════════════════════════════════════════════
app = FastAPI(
    title="Railway Maintenance AI System",
    description=(
        "End-to-end AI system for Indian Railways maintenance scheduling.\n\n"
        "Integrates TMS (Track), SMMS (Signal & Telecom), TDMS (Traction/OHE) "
        "defect data with COA block availability to produce risk-ranked, "
        "OR-Tools optimized maintenance schedules."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS — allow all origins in dev; restrict to your frontend domain in prod
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if os.path.exists(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/", include_in_schema=False)
async def serve_dashboard():
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "Railway Maintenance AI API — Web Dashboard static files missing."}



# ═══════════════════════════════════════════════════════════════════════════════
# Pydantic Request / Response Models
# ═══════════════════════════════════════════════════════════════════════════════
class AssetRiskRequest(BaseModel):
    """Schema for single-asset risk prediction."""
    asset_age_years:      float = Field(..., ge=0, le=100, example=15.0,
                                        description="Age of the asset in years")
    fault_count:          int   = Field(..., ge=0, le=50,  example=4,
                                        description="Fault/defect count in the observation period")
    last_inspection_days: int   = Field(..., ge=0, le=365, example=45,
                                        description="Days since last inspection")
    component_wear_pct:   float = Field(..., ge=0, le=100, example=65.0,
                                        description="Component wear as a percentage")
    traffic_load_mgt:     float = Field(..., ge=0, le=100, example=30.0,
                                        description="Traffic load in Million Gross Tonnes")
    zone:                 str   = Field(..., example="NR",
                                        description="Indian Railways zone code")
    source_system:        str   = Field(..., example="TMS",
                                        description="Source system: TMS | SMMS | TDMS")


class CopilotRequest(BaseModel):
    """Schema for Gemini LLM copilot queries."""
    query: str = Field(
        ...,
        example="Which track sections in NR zone need urgent attention this week?",
        description="Natural language question about the maintenance schedule or asset risk"
    )
    include_schedule_context: bool = Field(
        default=True,
        description="If true, inject current schedule summary into Gemini context"
    )


class RegenerateResponse(BaseModel):
    status:  str
    message: str
    tasks:   int


# ═══════════════════════════════════════════════════════════════════════════════
# Internal helpers — load data and models (cached at startup)
# ═══════════════════════════════════════════════════════════════════════════════
_cache: dict[str, Any] = {}


def _get_risk_model():
    """Lazy-loads and caches the risk model artifact."""
    if "risk" not in _cache:
        path = os.path.join(MODEL_DIR, "risk_model.pkl")
        if not os.path.exists(path):
            raise HTTPException(
                status_code=503,
                detail="Risk model not found. Run ai_engine.py first."
            )
        _cache["risk"] = joblib.load(path)
    return _cache["risk"]


def _get_duration_model():
    """Lazy-loads and caches the duration model artifact."""
    if "duration" not in _cache:
        path = os.path.join(MODEL_DIR, "duration_model.pkl")
        if not os.path.exists(path):
            raise HTTPException(
                status_code=503,
                detail="Duration model not found. Run ai_engine.py first."
            )
        _cache["duration"] = joblib.load(path)
    return _cache["duration"]


def _load_schedule() -> list[dict]:
    """Loads the latest optimized schedule JSON."""
    path = os.path.join(DATA_DIR, "optimized_schedule.json")
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data.get("schedule", [])


def _load_coa() -> pd.DataFrame:
    """Loads approved COA blocks."""
    path = os.path.join(DATA_DIR, "coa_data.csv")
    if not os.path.exists(path):
        raise HTTPException(status_code=503, detail="COA data missing. Run data_synthesizer.py.")
    df = pd.read_csv(path)
    return df[df["status"] == "approved"].reset_index(drop=True)


def _load_all_assets() -> pd.DataFrame:
    """Loads and unifies TMS + SMMS + TDMS with ML scores."""
    def _load_system(fname: str, system: str) -> pd.DataFrame:
        p = os.path.join(DATA_DIR, fname)
        if not os.path.exists(p):
            raise HTTPException(503, f"{fname} missing. Run data_synthesizer.py.")
        df              = pd.read_csv(p)
        df["source_system"] = system
        if "rail_age_years" in df.columns: df["asset_age_years"] = df["rail_age_years"]
        if "mast_age_years" in df.columns: df["asset_age_years"] = df["mast_age_years"]
        if "fault_count_6m" in df.columns: df["fault_count"]     = df["fault_count_6m"]
        if "fault_count_3m" in df.columns: df["fault_count"]     = df["fault_count_3m"]
        if "defect_count"   in df.columns: df["fault_count"]     = df["defect_count"]
        return df

    tms  = _load_system("tms_data.csv",  "TMS")
    smms = _load_system("smms_data.csv", "SMMS")
    tdms = _load_system("tdms_data.csv", "TDMS")

    for df_ in [tms, smms, tdms]:
        for col in ["asset_age_years","fault_count","last_inspection_days",
                    "component_wear_pct","traffic_load_mgt","zone","track_section"]:
            if col not in df_.columns:
                df_[col] = 0

    unified = pd.concat([tms, smms, tdms], ignore_index=True)

    risk_art      = _get_risk_model()
    duration_art  = _get_duration_model()
    risk_pipeline = risk_art["pipeline"]
    risk_features = risk_art["feature_cols"]
    dur_pipeline  = duration_art["pipeline"]
    dur_features  = duration_art["feature_cols"]

    X_risk                    = unified[risk_features].fillna(0)
    unified["risk_probability"] = risk_pipeline.predict_proba(X_risk)[:, 1]

    dur_input = unified[[c for c in dur_features if c != "risk_probability"]].copy().fillna(0)
    dur_input["risk_probability"] = unified["risk_probability"]
    dur_input = dur_input[dur_features]
    unified["est_maintenance_hrs"] = dur_pipeline.predict(dur_input)

    return unified


# ═══════════════════════════════════════════════════════════════════════════════
# Routes
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/health", tags=["System"])
async def health_check():
    """Returns API health, model status, and data availability."""
    risk_ready     = os.path.exists(os.path.join(MODEL_DIR, "risk_model.pkl"))
    duration_ready = os.path.exists(os.path.join(MODEL_DIR, "duration_model.pkl"))
    schedule_ready = os.path.exists(os.path.join(DATA_DIR, "optimized_schedule.json"))
    data_ready     = all(
        os.path.exists(os.path.join(DATA_DIR, f))
        for f in ["tms_data.csv", "smms_data.csv", "tdms_data.csv", "coa_data.csv"]
    )

    return {
        "status":          "healthy",
        "gemini_enabled":  bool(GEMINI_API_KEY),
        "models_ready":    risk_ready and duration_ready,
        "data_ready":      data_ready,
        "schedule_ready":  schedule_ready,
        "message": (
            "All systems operational" if (risk_ready and data_ready)
            else "Run data_synthesizer.py then ai_engine.py then optimizer.py"
        ),
    }


@app.get("/assets/risk", tags=["Assets"])
async def get_all_assets_with_risk(
    system:         Optional[str]   = None,
    zone:           Optional[str]   = None,
    min_risk:       Optional[float] = 0.0,
    top_n:          Optional[int]   = 100,
    sort_by_risk:   bool            = True,
):
    """
    Returns assets from TMS, SMMS, and TDMS with their ML risk scores.

    Query params:
      - system: Filter by source system (TMS | SMMS | TDMS)
      - zone:   Filter by railway zone (e.g. NR, SR, WR)
      - min_risk: Minimum risk probability threshold (0.0–1.0)
      - top_n:  Maximum number of results to return
      - sort_by_risk: If true, sorted by risk probability descending
    """
    df = _load_all_assets()

    if system:
        df = df[df["source_system"] == system.upper()]
    if zone:
        df = df[df["zone"] == zone.upper()]
    if min_risk > 0:
        df = df[df["risk_probability"] >= min_risk]
    if sort_by_risk:
        df = df.sort_values("risk_probability", ascending=False)

    df = df.head(top_n)

    keep_cols = ["asset_id", "source_system", "zone", "track_section",
                 "asset_age_years", "fault_count", "last_inspection_days",
                 "component_wear_pct", "traffic_load_mgt",
                 "risk_probability", "est_maintenance_hrs", "risk_label"]
    out_cols  = [c for c in keep_cols if c in df.columns]

    return {
        "total":   len(df),
        "assets":  df[out_cols].fillna(0).round(4).to_dict(orient="records"),
    }


@app.post("/predict/risk", tags=["Assets"])
async def predict_asset_risk(asset: AssetRiskRequest):
    """
    Predicts risk label and probability for a single new asset.
    Also returns the estimated maintenance duration.

    Use this endpoint during field inspections to get real-time risk assessment
    for any asset by entering its measured parameters.
    """
    risk_art     = _get_risk_model()
    duration_art = _get_duration_model()

    risk_pipeline  = risk_art["pipeline"]
    risk_features  = risk_art["feature_cols"]
    dur_pipeline   = duration_art["pipeline"]
    dur_features   = duration_art["feature_cols"]

    record = asset.model_dump()

    # Build risk input
    risk_input    = pd.DataFrame([{k: record.get(k, 0) for k in risk_features}])
    risk_prob     = float(risk_pipeline.predict_proba(risk_input)[0, 1])
    risk_label    = int(risk_prob >= 0.5)

    # Build duration input (includes risk_probability as stacked feature)
    dur_record    = {k: record.get(k, 0) for k in dur_features}
    dur_record["risk_probability"] = risk_prob
    dur_input     = pd.DataFrame([dur_record])
    est_hours     = float(dur_pipeline.predict(dur_input)[0])

    # Risk band labelling
    if risk_prob >= 0.85:
        risk_band = "CRITICAL"
    elif risk_prob >= 0.65:
        risk_band = "HIGH"
    elif risk_prob >= 0.45:
        risk_band = "MEDIUM"
    else:
        risk_band = "LOW"

    return {
        "input":                   record,
        "risk_label":              risk_label,
        "risk_probability":        round(risk_prob, 4),
        "risk_band":               risk_band,
        "estimated_maintenance_hrs": round(est_hours, 2),
        "recommendation": (
            "Schedule immediate maintenance block"   if risk_band == "CRITICAL"
            else "Plan maintenance in next 2 weeks"  if risk_band == "HIGH"
            else "Monitor — schedule within 1 month" if risk_band == "MEDIUM"
            else "Routine inspection schedule"
        ),
    }


@app.get("/schedule/optimized", tags=["Schedule"])
async def get_optimized_schedule(
    system:   Optional[str] = None,
    zone:     Optional[str] = None,
    date_from: Optional[str] = None,
    date_to:   Optional[str] = None,
):
    """
    Returns the current optimized maintenance schedule produced by OR-Tools.

    Query params:
      - system:     Filter by TMS | SMMS | TDMS
      - zone:       Filter by railway zone
      - date_from:  ISO date string (YYYY-MM-DD)
      - date_to:    ISO date string (YYYY-MM-DD)
    """
    schedule = _load_schedule()
    if not schedule:
        raise HTTPException(
            status_code=404,
            detail="No schedule found. Run optimizer.py to generate one."
        )

    # Apply filters
    if system:
        schedule = [s for s in schedule if s.get("source_system") == system.upper()]
    if zone:
        schedule = [s for s in schedule if s.get("zone") == zone.upper()]
    if date_from:
        schedule = [s for s in schedule if s.get("block_date", "") >= date_from]
    if date_to:
        schedule = [s for s in schedule if s.get("block_date", "") <= date_to]

    # Aggregate summary
    by_system = {}
    by_zone   = {}
    for task in schedule:
        s = task.get("source_system", "?")
        z = task.get("zone", "?")
        by_system[s] = by_system.get(s, 0) + 1
        by_zone[z]   = by_zone.get(z,   0) + 1

    return {
        "total_tasks":          len(schedule),
        "by_system":            by_system,
        "by_zone":              dict(sorted(by_zone.items(), key=lambda x: -x[1])[:10]),
        "schedule":             schedule,
    }


@app.post("/schedule/regenerate", tags=["Schedule"])
async def regenerate_schedule(background_tasks: BackgroundTasks):
    """
    Re-runs the full pipeline: data synthesis → ML training → optimization.

    This endpoint triggers a background process. The response returns
    immediately. Check /schedule/optimized after ~3–5 minutes.

    WARNING: This will overwrite existing data and models.
    """
    def _run_pipeline():
        """Runs each script in sequence as a subprocess."""
        scripts = [
            ("data_synthesizer.py", "Data Synthesis"),
            ("ai_engine.py",        "ML Training"),
            ("optimizer.py",        "Optimization"),
        ]
        for script, label in scripts:
            script_path = os.path.join(BASE_DIR, script)
            result = subprocess.run(
                [sys.executable, script_path],
                capture_output=True, text=True, cwd=BASE_DIR
            )
            if result.returncode != 0:
                print(f"ERROR in {label}: {result.stderr}")
                return

        # Clear model cache so next request reloads fresh models
        _cache.clear()

    background_tasks.add_task(_run_pipeline)

    return {
        "status":  "accepted",
        "message": "Pipeline started in background. Check /schedule/optimized in ~5 minutes.",
    }


@app.get("/coa/blocks", tags=["COA"])
async def get_coa_blocks(
    zone:       Optional[str] = None,
    block_type: Optional[str] = None,
    top_n:      int = 100,
):
    """
    Returns approved COA corridor blocks available for maintenance scheduling.

    Each block represents a pre-cleared maintenance window from the
    Control Office Application — a period where train traffic is suspended
    or routed around the section to allow engineering work.
    """
    df = _load_coa()
    if zone:
        df = df[df["zone"] == zone.upper()]
    if block_type:
        df = df[df["block_type"] == block_type]
    df = df.head(top_n)
    return {
        "total":  len(df),
        "blocks": df.fillna("").to_dict(orient="records"),
    }


@app.get("/eda/summary", tags=["EDA"])
async def get_eda_summary():
    """
    Returns statistical summaries for TMS, SMMS, and TDMS datasets.
    Useful for dashboards and understanding data distributions.
    """
    summaries = {}
    system_files = {
        "TMS":  "tms_data.csv",
        "SMMS": "smms_data.csv",
        "TDMS": "tdms_data.csv",
    }
    for system, fname in system_files.items():
        path = os.path.join(DATA_DIR, fname)
        if not os.path.exists(path):
            summaries[system] = {"error": f"{fname} not found"}
            continue
        df = pd.read_csv(path)
        risk_col = "risk_label" if "risk_label" in df.columns else None
        num_df   = df.select_dtypes(include=[np.number])
        summary  = {
            "total_records":    len(df),
            "high_risk_count":  int(df[risk_col].sum()) if risk_col else "N/A",
            "high_risk_pct":    round(float(df[risk_col].mean()) * 100, 2) if risk_col else "N/A",
            "feature_stats":    num_df.describe().round(3).to_dict(),
        }
        summaries[system] = summary

    return summaries


@app.post("/copilot/query", tags=["Copilot"])
async def copilot_query(request: CopilotRequest):
    """
    Answers natural language questions about the maintenance schedule,
    asset risks, and COA blocks using Google Gemini.

    Example queries:
      - "Which NR zone sections need urgent maintenance this week?"
      - "How many TDMS assets are critical risk in SCR zone?"
      - "What is the total maintenance load for the next 7 days?"
      - "Explain why OHE tension loss causes high risk."
    """
    if not _gemini_model:
        raise HTTPException(
            status_code=503,
            detail=(
                "Gemini API key not configured. "
                "Add GEMINI_API_KEY to your .env file. "
                "Get a free key at: https://aistudio.google.com/app/apikey"
            )
        )

    # ── Build context for Gemini ──────────────────────────────────────────────
    context_parts = [
        "You are an expert AI copilot for the Indian Railways Maintenance AI System.",
        "You have access to data from three subsystems:",
        "  • TMS  — Track Management System (track geometry, rail wear, ultrasonic defects)",
        "  • SMMS — Signalling Maintenance & Management System (signals, point machines, track circuits)",
        "  • TDMS — Traction Distribution Management System (25kV OHE, contact wire, boosters)",
        "  • COA  — Control Office Application (approved block windows for maintenance)",
        "",
        "The system uses XGBoost to predict asset risk (0–1 probability) and",
        "Random Forest to estimate maintenance duration. OR-Tools CP-SAT then",
        "schedules high-risk assets into COA-approved maintenance windows.",
        "",
    ]

    if request.include_schedule_context:
        schedule = _load_schedule()
        if schedule:
            # Summarise schedule for context (avoid token overflow)
            by_sys    = {}
            by_zone   = {}
            critical  = [t for t in schedule if t.get("risk_probability", 0) >= 0.85]
            for task in schedule:
                s = task.get("source_system", "?")
                z = task.get("zone", "?")
                by_sys[s]  = by_sys.get(s, 0)  + 1
                by_zone[z] = by_zone.get(z, 0) + 1

            context_parts += [
                f"CURRENT SCHEDULE SUMMARY ({len(schedule)} tasks):",
                f"  By system: {json.dumps(by_sys)}",
                f"  Top zones: {json.dumps(dict(sorted(by_zone.items(), key=lambda x:-x[1])[:5]))}",
                f"  Critical tasks (risk ≥ 85%): {len(critical)}",
                "  (Next 5 critical tasks):",
            ]
            for t in critical[:5]:
                context_parts.append(
                    f"    - {t['asset_id']} ({t['source_system']}) | "
                    f"Section: {t['track_section']} | "
                    f"Risk: {t['risk_probability']*100:.1f}% | "
                    f"Block: {t['block_date']} {t['block_start_time']}"
                )
        else:
            context_parts.append(
                "No schedule is currently available. "
                "Run optimizer.py to generate one."
            )

    context_parts += ["", f"USER QUESTION: {request.query}"]
    full_prompt = "\n".join(context_parts)

    # ── Call Gemini ───────────────────────────────────────────────────────────
    try:
        response = _gemini_model.generate_content(full_prompt)
        answer   = response.text
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Gemini API error: {str(e)}"
        )

    return {
        "query":    request.query,
        "answer":   answer,
        "context_injected": request.include_schedule_context,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))
    uvicorn.run("api:app", host=host, port=port, reload=True)
