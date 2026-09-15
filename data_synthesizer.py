"""
data_synthesizer.py
====================
Railway Maintenance AI System — Synthetic Data Generator

Generates realistic dummy data for all four Indian Railways subsystems:
  • TMS  — Track Management System        (Track/Engineering defects)
  • SMMS — Signalling Maintenance &        (Signal & Telecom defects)
            Management System
  • TDMS — Traction Distribution           (Overhead electrical/traction defects)
            Management System
  • COA  — Control Office Application      (Train timetables, goods forecasts,
                                            corridor block availability)

Design decisions:
  - Correlated features: risk_label is NOT random; it is derived from feature
    combinations that mimic real degradation physics (older asset + high traffic
    + long inspection gap = higher probability of high risk). This prevents the
    ML model from learning a trivially random target.
  - Realistic Indian Railways naming conventions are used throughout (zones,
    sections, train numbering, block types).
  - Class imbalance is intentional: ~25% high-risk assets, matching real-world
    maintenance data distributions.

Usage:
    python data_synthesizer.py
    # → writes data/tms_data.csv, smms_data.csv, tdms_data.csv, coa_data.csv
"""

import os
import random
import warnings
from datetime import date, datetime, time, timedelta

import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table

warnings.filterwarnings("ignore")
np.random.seed(42)
random.seed(42)

# Force UTF-8 output on Windows (prevents cp1252 UnicodeEncodeError with Rich)
import sys
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

console = Console()

# ── Output directory ──────────────────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DATA_DIR, exist_ok=True)

# ── Indian Railways constants ─────────────────────────────────────────────────
ZONES = ["CR", "ER", "ECR", "ECoR", "NR", "NCR", "NER", "NFR",
         "NWR", "SR", "SCR", "SER", "SECR", "SWR", "WR", "WCR"]

SECTIONS = [
    "NDLS-AGC", "AGC-JHS", "JHS-BPL", "BPL-NGP", "NGP-BZA",
    "BZA-MAS", "MAS-ED",  "ED-CBE",  "CBE-TVC", "CSTM-KYN",
    "KYN-IGP", "LTT-NED", "AWB-SC",  "SC-KZJ",  "KZJ-BZA",
    "HWH-BWN", "BWN-AZ",  "AZ-MGS",  "MGS-ALD", "ALD-CNB",
    "CNB-NDLS","NDLS-UMB","UMB-LDH", "LDH-ASR", "ASR-JUC",
    "JUC-ADI", "ADI-BRC", "BRC-ST",  "ST-BCT",  "BCT-RTM",
]


# ═══════════════════════════════════════════════════════════════════════════════
# 1.  TMS — Track Management System
#     Generates track/engineering defect records.
# ═══════════════════════════════════════════════════════════════════════════════
def generate_tms_data(n: int = 5000) -> pd.DataFrame:
    """
    Generates TMS records representing individual track asset observations.

    Risk logic (domain-driven, not random):
      high_risk = (defect_count > 5) OR
                  (rail_age_years > 20 AND traffic_load_mgt > 25) OR
                  (geometry_deviation_mm > 8 AND last_inspection_days > 60) OR
                  (ultrasonic_score < 30)
    """
    console.print("[bold cyan]Generating TMS data...[/bold cyan]")

    zones       = np.random.choice(ZONES, n)
    sections    = np.random.choice(SECTIONS, n)
    asset_ids   = [f"TMS-{z}-{i:05d}" for i, z in enumerate(zones)]

    # Age distribution: most tracks are 5–30 years old
    rail_age          = np.random.gamma(shape=4, scale=4, size=n).clip(1, 45).round(1)
    # Traffic load in Million Gross Tonnes (MGT) — higher on busy corridors
    traffic_load      = np.random.lognormal(mean=3.1, sigma=0.4, size=n).clip(5, 60).round(1)
    # Days since last inspection — should be < 30 by rule but often violated
    last_inspection   = np.random.exponential(scale=35, size=n).clip(5, 180).astype(int)
    # Engineering defect count per 10 km section
    defect_count      = np.random.negative_binomial(n=2, p=0.4, size=n).clip(0, 30)
    # Ultrasonic rail testing score (0–100, lower = more internal cracks)
    ultrasonic_score  = (100 - rail_age * 1.5 - traffic_load * 0.5
                         + np.random.normal(0, 8, n)).clip(0, 100).round(1)
    # Geometry deviation in mm (track alignment quality)
    geometry_dev      = (rail_age * 0.15 + traffic_load * 0.08
                         + np.random.exponential(scale=2, size=n)).clip(0, 25).round(2)
    # Gauge deviation in mm (should be 0 for standard gauge of 1676mm BG)
    gauge_dev         = np.random.exponential(scale=1.5, size=n).clip(0, 12).round(2)
    # Rail wear in mm (vertical + lateral combined)
    rail_wear         = (rail_age * 0.08 + traffic_load * 0.04
                         + np.random.normal(0, 0.5, n)).clip(0, 10).round(2)

    # ── Domain-driven risk labelling (CRITICAL: avoids arbitrary randomness) ──
    high_risk_conditions = (
        (defect_count > 5) |
        ((rail_age > 20) & (traffic_load > 25)) |
        ((geometry_dev > 8) & (last_inspection > 60)) |
        (ultrasonic_score < 30) |
        (gauge_dev > 6)
    )
    # Add a small noise probability so the model must learn generalisation
    noise_flip   = np.random.random(n) < 0.05
    risk_label   = (high_risk_conditions ^ noise_flip).astype(int)

    priority_map = {1: "URGENT", 0: "ROUTINE"}
    # Within high-risk, further prioritise by defect count
    priority = np.where(
        (risk_label == 1) & (defect_count > 10), "CRITICAL",
        np.where(risk_label == 1, "URGENT", "ROUTINE")
    )

    df = pd.DataFrame({
        "asset_id":              asset_ids,
        "track_section":         sections,
        "zone":                  zones,
        "rail_age_years":        rail_age,
        "traffic_load_mgt":      traffic_load,
        "last_inspection_days":  last_inspection,
        "defect_count":          defect_count,
        "ultrasonic_score":      ultrasonic_score,
        "geometry_deviation_mm": geometry_dev,
        "gauge_deviation_mm":    gauge_dev,
        "rail_wear_mm":          rail_wear,
        "risk_label":            risk_label,        # TARGET for classification
        "maintenance_priority":  priority,
    })

    path = os.path.join(DATA_DIR, "tms_data.csv")
    df.to_csv(path, index=False)
    _print_summary("TMS", df, path, risk_label)
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  SMMS — Signalling Maintenance & Management System
#     Generates signal & telecom asset defect records.
# ═══════════════════════════════════════════════════════════════════════════════
def generate_smms_data(n: int = 3000) -> pd.DataFrame:
    """
    Generates SMMS records for signalling and telecom assets.

    Asset types follow Indian Railways S&T department classifications.
    Risk logic:
      high_risk = (fault_count_6m > 3) OR
                  (component_wear_pct > 70) OR
                  (cable_insulation_resistance < 1.0 MΩ) OR
                  (signal_aspect_failures > 2)
    """
    console.print("[bold cyan]Generating SMMS data...[/bold cyan]")

    ASSET_TYPES = [
        "colour_light_signal", "point_machine", "track_circuit",
        "axle_counter", "relay_room", "OFC_cable", "BPAC_panel",
        "data_logger", "telecom_tower", "level_crossing_gate"
    ]

    zones       = np.random.choice(ZONES, n)
    sections    = np.random.choice(SECTIONS, n)
    asset_types = np.random.choice(ASSET_TYPES, n, p=[
        0.20, 0.15, 0.15, 0.12, 0.08, 0.10, 0.05, 0.05, 0.05, 0.05
    ])
    asset_ids   = [f"SMMS-{z}-{i:05d}" for i, z in enumerate(zones)]

    install_year   = np.random.randint(1990, 2023, n)
    age_years      = 2025 - install_year

    fault_count_6m = np.random.negative_binomial(n=1, p=0.5, size=n).clip(0, 20)
    last_serviced  = np.random.exponential(scale=40, size=n).clip(5, 365).astype(int)
    # Component wear accelerates with age and fault history
    component_wear = (age_years * 1.8 + fault_count_6m * 3
                      + np.random.normal(0, 5, n)).clip(0, 100).round(1)
    # Cable insulation resistance in MΩ — below 1.0 is dangerous
    cable_resist   = (10 - age_years * 0.2 + np.random.normal(0, 1, n)).clip(0.1, 10).round(2)
    # Number of signal aspect (aspect = displayed light) failures
    aspect_fails   = np.random.negative_binomial(n=1, p=0.7, size=n).clip(0, 15)
    # Battery backup hours remaining
    battery_backup = (48 - age_years * 0.5 + np.random.normal(0, 3, n)).clip(0, 72).round(1)

    # Estimated maintenance hours — regression target
    est_hrs = (1.5 + component_wear * 0.05 + fault_count_6m * 0.3
               + np.random.normal(0, 0.5, n)).clip(0.5, 24).round(1)

    # ── Risk labelling ────────────────────────────────────────────────────────
    high_risk_conditions = (
        (fault_count_6m > 3) |
        (component_wear > 70) |
        (cable_resist < 1.0) |
        (aspect_fails > 2) |
        (battery_backup < 8)
    )
    noise_flip  = np.random.random(n) < 0.05
    risk_label  = (high_risk_conditions ^ noise_flip).astype(int)

    df = pd.DataFrame({
        "asset_id":                    asset_ids,
        "asset_type":                  asset_types,
        "track_section":               sections,
        "zone":                        zones,
        "installation_year":           install_year,
        "asset_age_years":             age_years,
        "fault_count_6m":              fault_count_6m,
        "last_serviced_days":          last_serviced,
        "component_wear_pct":          component_wear,
        "cable_insulation_resistance": cable_resist,      # MΩ
        "signal_aspect_failure_count": aspect_fails,
        "battery_backup_hrs":          battery_backup,
        "est_maintenance_hrs":         est_hrs,           # TARGET for regression
        "risk_label":                  risk_label,        # TARGET for classification
    })

    path = os.path.join(DATA_DIR, "smms_data.csv")
    df.to_csv(path, index=False)
    _print_summary("SMMS", df, path, risk_label)
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  TDMS — Traction Distribution Management System
#     Generates Overhead Equipment (OHE) & traction asset defect records.
# ═══════════════════════════════════════════════════════════════════════════════
def generate_tdms_data(n: int = 3000) -> pd.DataFrame:
    """
    Generates TDMS records for 25kV AC overhead traction assets.

    Asset types follow Indian Railways TRD (Traction Rolling stock Dept.) categories.
    Risk logic:
      high_risk = (contact_wire_wear > 3.5mm remnant wear limit) OR
                  (tension_loss_pct > 15%) OR
                  (earthing_resistance > 10Ω) OR
                  (fault_count_3m > 4) OR
                  (stagger_deviation > 75mm — OHE standard is ±200mm)
    """
    console.print("[bold cyan]Generating TDMS data...[/bold cyan]")

    ASSET_TYPES = [
        "OHE_mast", "dropper", "stagger_wire", "feeder_cable",
        "booster_transformer", "switching_post", "sub_station_25kV",
        "auto_transformer", "return_conductor", "section_insulator"
    ]

    zones       = np.random.choice(ZONES, n)
    sections    = np.random.choice(SECTIONS, n)
    asset_types = np.random.choice(ASSET_TYPES, n, p=[
        0.25, 0.15, 0.10, 0.12, 0.08, 0.08, 0.07, 0.05, 0.05, 0.05
    ])
    asset_ids  = [f"TDMS-{z}-{i:05d}" for i, z in enumerate(zones)]

    mast_age   = np.random.gamma(shape=3, scale=6, size=n).clip(1, 50).round(1)
    last_insp  = np.random.exponential(scale=45, size=n).clip(5, 200).astype(int)

    # Contact wire wear in mm — new = 5.0mm, condemn limit = 1.5mm
    # Wear increases with age and traffic load
    traffic    = np.random.lognormal(mean=3.0, sigma=0.4, size=n).clip(5, 60).round(1)
    contact_wear = (5.0 - mast_age * 0.06 - traffic * 0.01
                    + np.random.normal(0, 0.3, n)).clip(1.0, 5.0).round(2)

    # OHE tension loss percentage (target: 0, wire slackens over time)
    tension_loss = (mast_age * 0.4 + np.random.exponential(scale=3, size=n)).clip(0, 30).round(1)

    # Stagger deviation in mm — should be within ±200mm, critical >±300mm
    stagger_dev  = np.random.exponential(scale=40, size=n).clip(0, 300).round(1)

    # Earth continuity resistance in Ohms — should be < 1Ω, >10Ω is dangerous
    earth_resist = np.random.exponential(scale=3, size=n).clip(0.1, 50).round(2)

    fault_count_3m = np.random.negative_binomial(n=1, p=0.6, size=n).clip(0, 15)

    # Estimated maintenance hours (OHE work is slow — needs block)
    est_hrs = (2.0 + mast_age * 0.05 + tension_loss * 0.1 + fault_count_3m * 0.4
               + np.random.normal(0, 0.7, n)).clip(1.0, 36).round(1)

    # ── Risk labelling ────────────────────────────────────────────────────────
    high_risk_conditions = (
        (contact_wear < 2.5) |          # < 2.5mm approaching condemn limit
        (tension_loss > 15) |
        (earth_resist > 10) |
        (fault_count_3m > 4) |
        (stagger_dev > 200)
    )
    noise_flip = np.random.random(n) < 0.05
    risk_label = (high_risk_conditions ^ noise_flip).astype(int)

    df = pd.DataFrame({
        "asset_id":              asset_ids,
        "asset_type":            asset_types,
        "track_section":         sections,
        "zone":                  zones,
        "mast_age_years":        mast_age,
        "traffic_load_mgt":      traffic,
        "last_inspection_days":  last_insp,
        "contact_wire_wear_mm":  contact_wear,
        "tension_loss_pct":      tension_loss,
        "stagger_deviation_mm":  stagger_dev,
        "earthing_resistance_ohm": earth_resist,
        "fault_count_3m":        fault_count_3m,
        "est_maintenance_hrs":   est_hrs,               # TARGET for regression
        "risk_label":            risk_label,            # TARGET for classification
    })

    path = os.path.join(DATA_DIR, "tdms_data.csv")
    df.to_csv(path, index=False)
    _print_summary("TDMS", df, path, risk_label)
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  COA — Control Office Application
#     Generates corridor block availability records (engineering blocks,
#     timetabled train windows, and goods train forecasts).
# ═══════════════════════════════════════════════════════════════════════════════
def generate_coa_data(
    tms_df: pd.DataFrame,
    smms_df: pd.DataFrame,
    tdms_df: pd.DataFrame,
    n_blocks: int = 2000
) -> pd.DataFrame:
    """
    Generates COA block availability records.

    In real Indian Railways COA:
      - Engineering blocks are pre-approved maintenance windows
      - Typically granted between 00:00–06:00 (traffic lull)
      - Block duration: 2–4 hours for track/OHE work
      - Each block is tied to a section and must not clash with train paths
      - Traffic density and goods train forecast inform block grant decisions

    The optimizer.py will query this table to find valid maintenance slots.
    """
    console.print("[bold cyan]Generating COA data...[/bold cyan]")

    BLOCK_TYPES = ["engineering_block", "emergency_block", "power_block", "caution_block"]
    # Power blocks (TDMS/OHE work) and engineering blocks (TMS/SMMS work)

    # Generate dates covering the next 6 months
    start_date = date.today()
    block_ids  = [f"COA-{i:05d}" for i in range(n_blocks)]
    sections   = np.random.choice(SECTIONS, n_blocks)
    zones      = [ZONES[SECTIONS.index(s) % len(ZONES)] for s in sections]
    block_dates = [
        start_date + timedelta(days=int(np.random.randint(0, 180)))
        for _ in range(n_blocks)
    ]

    # Most maintenance blocks are in the pre-dawn window 00:00–06:00
    # Some afternoon blocks exist (12:00–14:00) on low-traffic days
    block_windows = []
    for _ in range(n_blocks):
        window_type = np.random.choice(["night", "afternoon"], p=[0.75, 0.25])
        if window_type == "night":
            start_hr   = np.random.randint(0, 3)       # 00:00–02:59
            # Night blocks: 3–8 hours for major OHE/track work
            duration   = np.random.choice([3, 4, 5, 6, 8], p=[0.15, 0.25, 0.30, 0.20, 0.10])
        else:
            start_hr   = np.random.randint(12, 14)     # 12:00–13:59
            # Afternoon blocks: 2–4 hours for lighter inspection tasks
            duration   = np.random.choice([2, 3, 4],   p=[0.40, 0.40, 0.20])
        end_hr = min(start_hr + duration, 23)
        block_windows.append((
            time(start_hr, 0),
            time(end_hr, 59),
            duration
        ))

    block_start   = [bw[0].strftime("%H:%M") for bw in block_windows]
    block_end     = [bw[1].strftime("%H:%M") for bw in block_windows]
    block_dur     = [bw[2] for bw in block_windows]

    block_types   = np.random.choice(BLOCK_TYPES, n_blocks, p=[0.50, 0.10, 0.30, 0.10])

    # Traffic density (trains per hour on the section during block)
    # Lower density → block more likely to be approved
    traffic_density    = np.random.choice([0, 1, 2, 3, 4, 5], n_blocks,
                                          p=[0.30, 0.30, 0.20, 0.10, 0.07, 0.03])
    # Goods train forecast in the block window (0 = no goods train clashing)
    goods_forecast     = np.random.choice([0, 1, 2, 3], n_blocks, p=[0.40, 0.35, 0.15, 0.10])

    # Linked asset IDs (randomly pull from each dataset)
    all_assets = (
        list(tms_df["asset_id"].sample(min(500, len(tms_df)))) +
        list(smms_df["asset_id"].sample(min(400, len(smms_df)))) +
        list(tdms_df["asset_id"].sample(min(400, len(tdms_df))))
    )
    linked_assets = [
        ",".join(random.sample(all_assets, k=random.randint(1, 3)))
        for _ in range(n_blocks)
    ]

    # Approval status — higher traffic density → more likely pending/rejected
    statuses = []
    for td, gf in zip(traffic_density, goods_forecast):
        if td >= 4 or gf >= 3:
            statuses.append(np.random.choice(["pending", "rejected"], p=[0.6, 0.4]))
        elif td == 0 and gf == 0:
            statuses.append("approved")
        else:
            statuses.append(np.random.choice(["approved", "pending"], p=[0.75, 0.25]))

    df = pd.DataFrame({
        "block_id":               block_ids,
        "section":                sections,
        "zone":                   zones,
        "block_date":             [d.isoformat() for d in block_dates],
        "block_start_time":       block_start,
        "block_end_time":         block_end,
        "block_duration_hrs":     block_dur,
        "block_type":             block_types,
        "linked_asset_ids":       linked_assets,
        "traffic_density_per_hr": traffic_density,
        "goods_train_forecast":   goods_forecast,
        "status":                 statuses,
    })

    # Sort by date then start time (natural planning order)
    df = df.sort_values(["block_date", "block_start_time"]).reset_index(drop=True)

    path = os.path.join(DATA_DIR, "coa_data.csv")
    df.to_csv(path, index=False)

    console.print(f"[green]✓ COA:[/green] {len(df)} blocks | "
                  f"Approved: {(df.status=='approved').sum()} | "
                  f"Pending: {(df.status=='pending').sum()} | "
                  f"Rejected: {(df.status=='rejected').sum()}")
    console.print(f"   Saved → {path}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Utility
# ─────────────────────────────────────────────────────────────────────────────
def _print_summary(name: str, df: pd.DataFrame, path: str, risk_label: np.ndarray):
    """Prints a Rich summary table for a generated dataset."""
    high = int(risk_label.sum())
    low  = len(risk_label) - high
    pct  = high / len(risk_label) * 100
    console.print(
        f"[green]✓ {name}:[/green] {len(df)} records | "
        f"High-risk: {high} ({pct:.1f}%) | Low-risk: {low} ({100-pct:.1f}%)"
    )
    console.print(f"   Saved → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
def main():
    console.rule("[bold yellow]Railway Maintenance AI — Data Synthesizer[/bold yellow]")

    tms_df  = generate_tms_data(n=5000)
    smms_df = generate_smms_data(n=3000)
    tdms_df = generate_tdms_data(n=3000)
    coa_df  = generate_coa_data(tms_df, smms_df, tdms_df, n_blocks=2000)

    console.rule("[bold green]All datasets generated successfully![/bold green]")

    # Print a final summary table
    table = Table(title="Dataset Summary", style="bold")
    table.add_column("System", style="cyan")
    table.add_column("Records", justify="right")
    table.add_column("File", style="dim")
    for name, df, fname in [
        ("TMS",  tms_df,  "tms_data.csv"),
        ("SMMS", smms_df, "smms_data.csv"),
        ("TDMS", tdms_df, "tdms_data.csv"),
        ("COA",  coa_df,  "coa_data.csv"),
    ]:
        table.add_row(name, str(len(df)), f"data/{fname}")
    console.print(table)


if __name__ == "__main__":
    main()
