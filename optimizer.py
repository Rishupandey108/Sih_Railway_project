"""
optimizer.py
=============
Railway Maintenance AI System — OR-Tools Scheduling Optimizer

Uses Google OR-Tools CP-SAT (Constraint Programming — SAT) solver to produce
an optimal maintenance schedule from the ML model outputs.

Features:
  1. Dynamic Check & Single-Pass Execution: For each high-risk asset, checks if
     a single approved COA block window in its zone fits the full repair duration.
  2. Sprint Fallback & Task Splitting: If no single window is long enough, splits
     the asset maintenance into sequential sub-tasks (sprints of max 3.5 hours).
  3. Precedence Constraints: Enforces that Sprint k+1 must occur chronologically
     after Sprint k on the timeline.
  4. Output Tagging: Scheduled tasks are tagged in data/optimized_schedule.json
     with sprint labels like `SMMS-SR-01004 (Sprint 1/2)`.

Usage:
    python optimizer.py
"""

import json
import os
import warnings
from datetime import date, datetime

import joblib
import numpy as np
import pandas as pd
from ortools.sat.python import cp_model
from rich.console import Console
from rich.table import Table

warnings.filterwarnings("ignore")

# Force UTF-8 output on Windows (prevents cp1252 UnicodeEncodeError with Rich)
import sys
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

console = Console()

BASE_DIR   = os.path.dirname(__file__)
DATA_DIR   = os.path.join(BASE_DIR, "data")
MODEL_DIR  = os.path.join(BASE_DIR, "models")

# ── Scheduling constants ──────────────────────────────────────────────────────
RISK_THRESHOLD     = 0.60    # Only schedule assets with risk_prob > 60%
MAX_CREW_PER_BLOCK  = 3      # Maximum number of tasks maintained per block window
MAX_SPRINT_HOURS   = 3.5    # Maximum duration for split sprint sub-tasks
TOP_N_ASSETS       = 150     # Consider top N high-risk assets
TOP_N_BLOCKS       = 300     # Consider approved blocks in next 30 days


# ═══════════════════════════════════════════════════════════════════════════════
# Step 1: Load data and generate risk scores using trained models
# ═══════════════════════════════════════════════════════════════════════════════
def load_assets_with_risk() -> pd.DataFrame:
    """
    Loads TMS + SMMS + TDMS data, runs the risk model, and returns a unified
    DataFrame of high-risk assets annotated with risk_probability and
    est_maintenance_hrs.
    """
    console.print("[cyan]Loading assets and scoring with ML models...[/cyan]")

    risk_artifact     = joblib.load(os.path.join(MODEL_DIR, "risk_model.pkl"))
    duration_artifact = joblib.load(os.path.join(MODEL_DIR, "duration_model.pkl"))

    risk_pipeline     = risk_artifact["pipeline"]
    risk_features     = risk_artifact["feature_cols"]
    duration_pipeline = duration_artifact["pipeline"]
    duration_features = duration_artifact["feature_cols"]

    def _load_and_align(csv_name: str, system: str) -> pd.DataFrame:
        df = pd.read_csv(os.path.join(DATA_DIR, csv_name))
        df["source_system"] = system

        if "rail_age_years"  in df.columns: df["asset_age_years"] = df["rail_age_years"]
        if "mast_age_years"  in df.columns: df["asset_age_years"] = df["mast_age_years"]
        if "fault_count_6m"  in df.columns: df["fault_count"]     = df["fault_count_6m"]
        if "fault_count_3m"  in df.columns: df["fault_count"]     = df["fault_count_3m"]
        if "defect_count"    in df.columns: df["fault_count"]     = df["defect_count"]
        if "last_inspection_days" not in df.columns: df["last_inspection_days"] = 30
        if "component_wear_pct"   not in df.columns: df["component_wear_pct"]   = 50.0
        if "traffic_load_mgt"     not in df.columns: df["traffic_load_mgt"]     = 20.0
        if "asset_age_years"      not in df.columns: df["asset_age_years"]      = 10.0
        if "fault_count"          not in df.columns: df["fault_count"]          = 0

        return df

    tms  = _load_and_align("tms_data.csv",  "TMS")
    smms = _load_and_align("smms_data.csv", "SMMS")
    tdms = _load_and_align("tdms_data.csv", "TDMS")
    all_assets = pd.concat([tms, smms, tdms], ignore_index=True)

    for col in ["zone", "track_section"]:
        if col not in all_assets.columns:
            all_assets[col] = "UNKNOWN"

    X_risk          = all_assets[risk_features].copy()
    risk_probs      = risk_pipeline.predict_proba(X_risk)[:, 1]
    all_assets["risk_probability"] = risk_probs

    dur_input = all_assets[
        [c for c in duration_features if c != "risk_probability"]
    ].copy()
    dur_input["risk_probability"] = risk_probs
    dur_input = dur_input[[c for c in duration_features]]
    all_assets["est_maintenance_hrs"] = duration_pipeline.predict(dur_input)

    high_risk = (
        all_assets[all_assets["risk_probability"] >= RISK_THRESHOLD]
        .sort_values("risk_probability", ascending=False)
        .head(TOP_N_ASSETS)
        .reset_index(drop=True)
    )

    console.print(
        f"[green]✓ High-risk assets (prob ≥ {RISK_THRESHOLD}):[/green] "
        f"{len(high_risk)} of {len(all_assets):,} total"
    )
    return high_risk


# ═══════════════════════════════════════════════════════════════════════════════
# Step 2: Load approved COA blocks
# ═══════════════════════════════════════════════════════════════════════════════
def load_approved_blocks() -> pd.DataFrame:
    """Loads approved COA blocks in the next 30 days."""
    coa = pd.read_csv(os.path.join(DATA_DIR, "coa_data.csv"))
    coa["block_date"] = pd.to_datetime(coa["block_date"])

    today   = pd.Timestamp(date.today())
    horizon = today + pd.Timedelta(days=30)

    approved = (
        coa[
            (coa["status"] == "approved") &
            (coa["block_date"] >= today) &
            (coa["block_date"] <= horizon)
        ]
        .head(TOP_N_BLOCKS)
        .reset_index(drop=True)
    )

    console.print(
        f"[green]✓ Approved COA blocks (next 30 days):[/green] {len(approved)}"
    )
    return approved


# ═══════════════════════════════════════════════════════════════════════════════
# Task Preparation & Sprint Splitting
# ═══════════════════════════════════════════════════════════════════════════════
def prepare_tasks_with_sprint_splitting(assets: pd.DataFrame, blocks: pd.DataFrame) -> list[dict]:
    """
    Implements conditional task splitting (sprint scheduling) for high-risk assets:
    1. Dynamic Check: Checks if any single approved COA block window in asset's zone
       fits est_duration (and matches block rules).
    2. Single-Pass Execution: If full window exists, keep as single execution block.
    3. Sprint Fallback: If no single window is long enough, split task into sequential
       sub-tasks (max 3.5 hours per sprint: Sprint 1, Sprint 2, etc.).
    """
    tasks = []
    blocks_by_zone = {}
    for _, block in blocks.iterrows():
        z = str(block["zone"])
        if z not in blocks_by_zone:
            blocks_by_zone[z] = []
        blocks_by_zone[z].append(block)

    unsplit_count = 0
    split_asset_count = 0
    total_sprint_tasks = 0

    for _, asset in assets.iterrows():
        asset_id = str(asset["asset_id"])
        zone     = str(asset["zone"])
        sys_type = str(asset["source_system"])
        est_hrs  = float(asset["est_maintenance_hrs"])

        zone_blocks = blocks_by_zone.get(zone, [])

        has_full_window = False
        for b in zone_blocks:
            b_dur  = float(b["block_duration_hrs"])
            b_type = str(b["block_type"])

            if b_type == "power_block" and sys_type != "TDMS":
                continue
            if b_type == "engineering_block" and sys_type == "TDMS":
                continue

            if b_dur >= est_hrs:
                has_full_window = True
                break

        if has_full_window:
            unsplit_count += 1
            total_sprint_tasks += 1
            tasks.append({
                "task_id":             asset_id,
                "sprint_label":        asset_id,
                "parent_asset_id":     asset_id,
                "source_system":       asset["source_system"],
                "track_section":       asset.get("track_section", "N/A"),
                "zone":                asset.get("zone", "N/A"),
                "risk_probability":    float(asset["risk_probability"]),
                "est_maintenance_hrs": round(est_hrs, 2),
                "sprint_index":        1,
                "total_sprints":       1,
                "is_split":            False,
            })
        else:
            split_asset_count += 1
            rem = est_hrs
            durations = []
            while rem > MAX_SPRINT_HOURS:
                durations.append(MAX_SPRINT_HOURS)
                rem -= MAX_SPRINT_HOURS
            if rem > 0.05:
                durations.append(round(rem, 2))

            n_sprints = len(durations)
            total_sprint_tasks += n_sprints

            for k, dur in enumerate(durations):
                sprint_idx   = k + 1
                sprint_label = f"{asset_id} (Sprint {sprint_idx}/{n_sprints})"
                tasks.append({
                    "task_id":             sprint_label,
                    "sprint_label":        sprint_label,
                    "parent_asset_id":     asset_id,
                    "source_system":       asset["source_system"],
                    "track_section":       asset.get("track_section", "N/A"),
                    "zone":                asset.get("zone", "N/A"),
                    "risk_probability":    float(asset["risk_probability"]),
                    "est_maintenance_hrs": round(dur, 2),
                    "sprint_index":        sprint_idx,
                    "total_sprints":       n_sprints,
                    "is_split":            True,
                })

    console.print(
        f"   Task Splitting Summary: [bold cyan]{unsplit_count}[/bold cyan] single-pass assets, "
        f"[bold yellow]{split_asset_count}[/bold yellow] split assets "
        f"([bold green]{total_sprint_tasks}[/bold green] total scheduled task units)"
    )
    return tasks


def compute_block_timestamps(blocks: pd.DataFrame):
    """Computes integer start and end minute timestamps from epoch for each block."""
    start_mins = []
    end_mins = []
    for _, block in blocks.iterrows():
        dt_str = str(block["block_date"]).split(" ")[0]
        start_str = str(block["block_start_time"])
        end_str   = str(block["block_end_time"])

        sh, sm = map(int, start_str[:5].split(":"))
        eh, em = map(int, end_str[:5].split(":"))

        d_ordinal = datetime.strptime(dt_str, "%Y-%m-%d").toordinal()

        start_m = d_ordinal * 1440 + sh * 60 + sm
        end_m   = d_ordinal * 1440 + eh * 60 + em
        if end_m <= start_m:
            end_m += 1440

        start_mins.append(start_m)
        end_mins.append(end_m)

    return start_mins, end_mins


# ═══════════════════════════════════════════════════════════════════════════════
# Step 3: OR-Tools CP-SAT Scheduler
# ═══════════════════════════════════════════════════════════════════════════════
def run_optimizer(assets: pd.DataFrame, blocks: pd.DataFrame) -> list[dict]:
    """
    Runs the CP-SAT constraint solver to find the optimal maintenance schedule
    with conditional sprint task splitting and precedence enforcement.
    """
    console.print("[cyan]Running CP-SAT optimizer...[/cyan]")

    if len(assets) == 0 or len(blocks) == 0:
        console.print("[red]No assets or blocks to schedule.[/red]")
        return []

    tasks = prepare_tasks_with_sprint_splitting(assets, blocks)
    n_tasks  = len(tasks)
    n_blocks = len(blocks)

    block_start_mins, block_end_mins = compute_block_timestamps(blocks)

    model = cp_model.CpModel()

    # Decision variables x[i, j]: task i in block j
    x = {}
    for i in range(n_tasks):
        for j in range(n_blocks):
            x[i, j] = model.NewBoolVar(f"x[{i},{j}]")

    # Constraint 1: Each task/sprint scheduled AT MOST ONCE
    for i in range(n_tasks):
        model.Add(sum(x[i, j] for j in range(n_blocks)) <= 1)

    # Constraint 2: Crew capacity — max tasks per block
    for j in range(n_blocks):
        model.Add(sum(x[i, j] for i in range(n_tasks)) <= MAX_CREW_PER_BLOCK)

    # Constraint 3: Duration fit (task duration <= block duration)
    SCALE = 10
    for i in range(n_tasks):
        task_dur_scaled = int(tasks[i]["est_maintenance_hrs"] * SCALE)
        for j in range(n_blocks):
            block_dur_scaled = int(blocks.at[j, "block_duration_hrs"] * SCALE)
            if task_dur_scaled > block_dur_scaled:
                model.Add(x[i, j] == 0)

    # Constraint 4: Zone matching
    for i in range(n_tasks):
        task_zone = tasks[i]["zone"]
        for j in range(n_blocks):
            if task_zone != blocks.at[j, "zone"]:
                model.Add(x[i, j] == 0)

    # Constraint 5: Power / Engineering block rules
    for i in range(n_tasks):
        sys_i = tasks[i]["source_system"]
        for j in range(n_blocks):
            b_type = blocks.at[j, "block_type"]
            if b_type == "power_block" and sys_i != "TDMS":
                model.Add(x[i, j] == 0)
            if b_type == "engineering_block" and sys_i == "TDMS":
                model.Add(x[i, j] == 0)

    # Constraint 6: Sequential Sprint Execution & Precedence Constraints
    parent_groups = {}
    for i, t in enumerate(tasks):
        pid = t["parent_asset_id"]
        if pid not in parent_groups:
            parent_groups[pid] = []
        parent_groups[pid].append((i, t))

    precedence_constraints_count = 0
    for pid, group in parent_groups.items():
        if len(group) > 1:
            group.sort(key=lambda item: item[1]["sprint_index"])
            for k in range(len(group) - 1):
                t1_idx, t1 = group[k]
                t2_idx, t2 = group[k + 1]

                # 6a. Sprint k+1 can only be scheduled if Sprint k is scheduled
                model.Add(sum(x[t2_idx, j] for j in range(n_blocks)) <= sum(x[t1_idx, j] for j in range(n_blocks)))

                # 6b. Precedence constraint: Sprint k+1 block (j2) must start after Sprint k block (j1) ends
                for j1 in range(n_blocks):
                    for j2 in range(n_blocks):
                        if block_start_mins[j2] < block_end_mins[j1]:
                            model.Add(x[t1_idx, j1] + x[t2_idx, j2] <= 1)
                            precedence_constraints_count += 1

    console.print(f"   Enforced [bold cyan]{precedence_constraints_count:,}[/bold cyan] CP-SAT sprint precedence constraints")

    # Objective: Maximise risk-weighted coverage
    risk_scores_int = [
        int(tasks[i]["risk_probability"] * 1000)
        for i in range(n_tasks)
    ]

    model.Maximize(
        sum(risk_scores_int[i] * x[i, j]
            for i in range(n_tasks)
            for j in range(n_blocks))
    )

    # Solve
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 30.0
    solver.parameters.num_search_workers = 4

    status = solver.Solve(model)

    STATUS_NAMES = {
        cp_model.OPTIMAL:    "OPTIMAL",
        cp_model.FEASIBLE:   "FEASIBLE",
        cp_model.INFEASIBLE: "INFEASIBLE",
        cp_model.UNKNOWN:    "UNKNOWN",
    }
    status_name = STATUS_NAMES.get(status, str(status))
    console.print(f"   Solver status : [bold]{status_name}[/bold]")
    console.print(f"   Objective     : {solver.ObjectiveValue():,.0f}")
    console.print(f"   Wall time     : {solver.WallTime():.2f}s")

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        console.print("[red]No feasible schedule found.[/red]")
        return []

    # Extract schedule
    schedule = []
    for i in range(n_tasks):
        for j in range(n_blocks):
            if solver.BooleanValue(x[i, j]):
                t = tasks[i]
                block = blocks.iloc[j]
                schedule.append({
                    "asset_id":               t["sprint_label"],   # Tagged with sprint label e.g., SMMS-SR-01004 (Sprint 1/2)
                    "parent_asset_id":        t["parent_asset_id"],
                    "sprint_label":           t["sprint_label"],
                    "sprint_index":           t["sprint_index"],
                    "total_sprints":          t["total_sprints"],
                    "is_split":               t["is_split"],
                    "source_system":          t["source_system"],
                    "track_section":          t.get("track_section", "N/A"),
                    "zone":                   t.get("zone", "N/A"),
                    "risk_probability":       round(float(t["risk_probability"]), 4),
                    "est_maintenance_hrs":    round(float(t["est_maintenance_hrs"]), 2),
                    "block_id":               block["block_id"],
                    "block_date":             str(block["block_date"].date()),
                    "block_start_time":       block["block_start_time"],
                    "block_end_time":         block["block_end_time"],
                    "block_type":             block["block_type"],
                    "block_duration_hrs":     float(block["block_duration_hrs"]),
                    "scheduled_at":           datetime.now().isoformat(),
                })

    schedule.sort(key=lambda r: (-r["risk_probability"], r["block_date"], r["sprint_index"]))
    return schedule


# ═══════════════════════════════════════════════════════════════════════════════
# Step 4: Reporting
# ═══════════════════════════════════════════════════════════════════════════════
def print_schedule(schedule: list[dict]):
    """Prints the optimized schedule as a Rich table."""
    table = Table(
        title=f"Optimized Maintenance Schedule ({len(schedule)} task units)",
        style="bold",
        show_lines=True,
    )
    table.add_column("Asset ID / Sprint Label", style="cyan",   no_wrap=True)
    table.add_column("System",                  style="magenta")
    table.add_column("Section",                 style="green")
    table.add_column("Risk %",                  justify="right", style="red")
    table.add_column("Est. Hrs",                justify="right")
    table.add_column("Block Date",              style="yellow")
    table.add_column("Window",                  style="dim")
    table.add_column("Block Type",              style="blue")

    for row in schedule[:30]:
        risk_pct = f"{row['risk_probability']*100:.1f}%"
        window   = f"{row['block_start_time']} – {row['block_end_time']}"
        table.add_row(
            row["asset_id"],
            row["source_system"],
            row["track_section"],
            risk_pct,
            f"{row['est_maintenance_hrs']:.1f}h",
            row["block_date"],
            window,
            row["block_type"],
        )

    if len(schedule) > 30:
        table.add_row("...", f"(+{len(schedule)-30} more)", "", "", "", "", "", "")

    console.print(table)

    systems = {}
    for row in schedule:
        s = row["source_system"]
        systems[s] = systems.get(s, 0) + 1

    sprint_tasks_count = sum(1 for row in schedule if row.get("is_split"))
    single_tasks_count = sum(1 for row in schedule if not row.get("is_split"))
    unique_assets = len(set(row["parent_asset_id"] for row in schedule))

    console.print(f"\n[bold]Schedule Summary:[/bold]")
    for sys, count in sorted(systems.items()):
        console.print(f"   {sys}: {count} tasks")
    console.print(f"   Unique Assets Covered : {unique_assets}")
    console.print(f"   Single-Pass Tasks     : {single_tasks_count}")
    console.print(f"   Sprint Tasks (Split)  : {sprint_tasks_count}")
    console.print(f"   Total Scheduled Tasks : {len(schedule)}")


def save_schedule(schedule: list[dict]) -> str:
    """Saves the schedule to data/optimized_schedule.json."""
    path = os.path.join(DATA_DIR, "optimized_schedule.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({
            "generated_at":       datetime.now().isoformat(),
            "total_tasks":        len(schedule),
            "risk_threshold":     RISK_THRESHOLD,
            "max_crew_per_block": MAX_CREW_PER_BLOCK,
            "schedule":           schedule,
        }, f, indent=2, ensure_ascii=False)
    console.print(f"[green]✓ Schedule saved → {path}[/green]")
    return path


# ═══════════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    console.rule("[bold yellow]Railway Maintenance AI — Scheduling Optimizer[/bold yellow]")

    assets   = load_assets_with_risk()
    blocks   = load_approved_blocks()
    schedule = run_optimizer(assets, blocks)

    if schedule:
        print_schedule(schedule)
        save_schedule(schedule)
    else:
        console.print(
            "[yellow]No schedule generated. Check asset/block availability.[/yellow]"
        )

    console.rule("[bold green]Optimization Complete![/bold green]")


if __name__ == "__main__":
    main()
