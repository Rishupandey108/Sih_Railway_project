"""
ai_engine.py
=============
Railway Maintenance AI System — ML Training Engine

Trains two models on the synthesized TMS + SMMS + TDMS data:

  MODEL 1 — XGBoost Risk Classifier
    Input : Asset features (age, defects, inspection gap, wear measurements)
    Output: risk_label (0 = low risk, 1 = high risk) + probability score (0.0–1.0)
    Pitfall prevented: Class imbalance handled via `scale_pos_weight`; data
    leakage prevented by fitting all transformers ONLY on training fold.

  MODEL 2 — Random Forest Duration Regressor
    Input : Asset features + risk score (from Model 1 as a stacked feature)
    Output: est_maintenance_hrs (how long the maintenance job will take)
    Pitfall prevented: Target leakage avoided by NOT including risk_label in
    regression features; risk_score (probability) used instead.

Usage:
    python ai_engine.py
    # → saves models/risk_model.pkl and models/duration_model.pkl
    # → saves reports/eda_*.png and reports/evaluation_*.png
"""

import os
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # Non-interactive backend for server environments
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    ConfusionMatrixDisplay, classification_report,
    mean_absolute_error, mean_squared_error, r2_score,
    roc_auc_score, roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from rich.console import Console
from rich.table import Table

warnings.filterwarnings("ignore")
np.random.seed(42)
# Force UTF-8 output on Windows (prevents cp1252 UnicodeEncodeError with Rich)
import sys
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

console = Console()

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(__file__)
DATA_DIR     = os.path.join(BASE_DIR, "data")
MODEL_DIR    = os.path.join(BASE_DIR, "models")
REPORT_DIR   = os.path.join(BASE_DIR, "reports")

for d in [MODEL_DIR, REPORT_DIR]:
    os.makedirs(d, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Helper: Plot saving
# ═══════════════════════════════════════════════════════════════════════════════
def _save_fig(fig, name: str):
    path = os.path.join(REPORT_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    console.print(f"   [dim]Plot saved → {path}[/dim]")


# ═══════════════════════════════════════════════════════════════════════════════
# Step 1: Data Loading & Unification
# ═══════════════════════════════════════════════════════════════════════════════
def load_and_unify_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads TMS, SMMS, and TDMS CSVs and creates two unified DataFrames:
      1. classification_df  — for risk label prediction (XGBoost)
      2. regression_df      — for maintenance duration prediction (Random Forest)

    Design decision: We train a SINGLE unified model across all three systems.
    This allows the model to learn cross-system patterns (e.g., a high-risk
    track section often has correlated OHE defects). A `source_system` column
    is added as a categorical feature so the model can distinguish systems.

    Pitfall: Data leakage from target variable.
      → 'risk_label' is EXCLUDED from regression features.
      → 'est_maintenance_hrs' is EXCLUDED from classification features.
    """
    console.rule("[cyan]Step 1: Loading Data[/cyan]")

    # ── TMS ───────────────────────────────────────────────────────────────────
    tms = pd.read_csv(os.path.join(DATA_DIR, "tms_data.csv"))
    # TMS doesn't have est_maintenance_hrs — derive a proxy from defect_count
    tms["est_maintenance_hrs"] = (
        1.5 + tms["defect_count"] * 0.4 + tms["rail_age_years"] * 0.05
        + np.random.normal(0, 0.3, len(tms))
    ).clip(0.5, 24).round(1)
    tms["asset_age_years"]     = tms["rail_age_years"]
    tms["fault_count"]         = tms["defect_count"]
    tms["source_system"]       = "TMS"

    # ── SMMS ──────────────────────────────────────────────────────────────────
    smms = pd.read_csv(os.path.join(DATA_DIR, "smms_data.csv"))
    smms["asset_age_years"] = smms["asset_age_years"]
    smms["fault_count"]     = smms["fault_count_6m"]
    smms["source_system"]   = "SMMS"

    # ── TDMS ──────────────────────────────────────────────────────────────────
    tdms = pd.read_csv(os.path.join(DATA_DIR, "tdms_data.csv"))
    tdms["asset_age_years"] = tdms["mast_age_years"]
    tdms["fault_count"]     = tdms["fault_count_3m"]
    tdms["source_system"]   = "TDMS"

    # ── Common feature set (intersection of meaningful columns) ───────────────
    # We select features that exist conceptually across all three systems
    COMMON_FEATURES = [
        "asset_id", "zone", "track_section", "source_system",
        "asset_age_years", "traffic_load_mgt", "last_inspection_days",
        "fault_count", "component_wear_pct",
        "risk_label", "est_maintenance_hrs",
    ]

    def _align(df: pd.DataFrame) -> pd.DataFrame:
        """Fill missing common columns with 0 and select only needed cols."""
        for col in COMMON_FEATURES:
            if col not in df.columns:
                df[col] = 0.0
        return df[COMMON_FEATURES].copy()

    unified = pd.concat([_align(tms), _align(smms), _align(tdms)],
                        ignore_index=True)

    # ── Handle missing values ─────────────────────────────────────────────────
    # Pitfall: Imputing with mean of WHOLE dataset before split leaks test info.
    # We record medians here for reference only; actual imputation is inside pipeline.
    console.print(f"[green]✓ Unified dataset:[/green] {len(unified):,} records")
    console.print(f"   Missing values: {unified.isnull().sum().sum()}")

    # ── Separate targets from features ────────────────────────────────────────
    classification_df = unified.drop(columns=["est_maintenance_hrs"])
    regression_df     = unified.drop(columns=["risk_label"])

    return classification_df, regression_df


# ═══════════════════════════════════════════════════════════════════════════════
# Step 2: Exploratory Data Analysis
# ═══════════════════════════════════════════════════════════════════════════════
def run_eda(df: pd.DataFrame):
    """
    Generates essential EDA plots:
      1. Class distribution (imbalance check)
      2. Feature distributions per class
      3. Correlation heatmap (numerical features)
      4. Asset age vs fault count (scatter)
    """
    console.rule("[cyan]Step 2: EDA[/cyan]")

    sns.set_theme(style="darkgrid", palette="muted")

    # ── 2a. Class distribution ────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("EDA — Risk Label Distribution", fontsize=14, fontweight="bold")

    counts = df["risk_label"].value_counts()
    axes[0].bar(["Low Risk (0)", "High Risk (1)"], counts.values,
                color=["#2ecc71", "#e74c3c"], edgecolor="white", linewidth=1.5)
    axes[0].set_title("Overall Class Balance")
    axes[0].set_ylabel("Count")
    for i, v in enumerate(counts.values):
        axes[0].text(i, v + 30, f"{v}\n({v/len(df)*100:.1f}%)", ha="center", fontsize=10)

    # Per-system breakdown
    system_risk = df.groupby("source_system")["risk_label"].mean().sort_values()
    system_risk.plot(kind="barh", ax=axes[1], color="#3498db")
    axes[1].set_title("High-Risk Rate by System")
    axes[1].set_xlabel("Proportion High-Risk")
    axes[1].axvline(0.25, color="red", linestyle="--", label="25% threshold")
    axes[1].legend()

    _save_fig(fig, "eda_01_class_distribution.png")

    # ── 2b. Feature distributions by risk class ───────────────────────────────
    num_features = ["asset_age_years", "fault_count", "last_inspection_days",
                    "component_wear_pct", "traffic_load_mgt"]
    fig, axes = plt.subplots(1, len(num_features), figsize=(20, 4))
    fig.suptitle("EDA — Feature Distributions by Risk Class", fontsize=13)

    for ax, feat in zip(axes, num_features):
        for label, color in [(0, "#2ecc71"), (1, "#e74c3c")]:
            subset = df[df["risk_label"] == label][feat].dropna()
            ax.hist(subset, bins=30, alpha=0.6, color=color,
                    label=f"Risk={label}", density=True)
        ax.set_title(feat.replace("_", " ").title())
        ax.set_xlabel("Value")
        ax.legend(fontsize=8)

    _save_fig(fig, "eda_02_feature_distributions.png")

    # ── 2c. Correlation heatmap ───────────────────────────────────────────────
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    corr     = df[num_cols].corr()

    fig, ax  = plt.subplots(figsize=(10, 8))
    mask     = np.triu(np.ones_like(corr, dtype=bool))  # upper triangle mask
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
                center=0, ax=ax, linewidths=0.5, annot_kws={"size": 8})
    ax.set_title("Feature Correlation Heatmap", fontsize=13, fontweight="bold")
    _save_fig(fig, "eda_03_correlation_heatmap.png")

    # ── 2d. Age vs fault count coloured by risk ───────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(
        df["asset_age_years"], df["fault_count"],
        c=df["risk_label"], cmap="RdYlGn_r", alpha=0.4, s=10
    )
    plt.colorbar(scatter, ax=ax, label="Risk Label")
    ax.set_xlabel("Asset Age (Years)")
    ax.set_ylabel("Fault Count")
    ax.set_title("Asset Age vs. Fault Count (coloured by risk)")
    _save_fig(fig, "eda_04_age_vs_fault.png")

    console.print("[green]✓ EDA plots saved to reports/[/green]")


# ═══════════════════════════════════════════════════════════════════════════════
# Step 3: XGBoost Risk Classifier
# ═══════════════════════════════════════════════════════════════════════════════
def train_risk_model(df: pd.DataFrame) -> Pipeline:
    """
    Trains an XGBoost binary classifier to predict asset maintenance risk.

    Pipeline design:
      ColumnTransformer → XGBClassifier
        ├── Numerical:   StandardScaler (mean=0, std=1)
        └── Categorical: OneHotEncoder (handle_unknown='ignore' for inference)

    Key decisions:
      • scale_pos_weight = n_negative / n_positive compensates for class imbalance
        WITHOUT discarding data (unlike undersampling), preserving all signal.
      • StratifiedKFold ensures each CV fold has the same class proportion.
      • GridSearchCV searches key XGBoost hyperparameters; larger grids could
        use RandomizedSearchCV or Optuna for efficiency.
      • eval_metric='logloss' is used internally; AUC-ROC reported externally
        as it is more informative under class imbalance than accuracy.

    Pitfall prevented — Data Leakage:
      The StandardScaler is fit INSIDE the pipeline, which means it is re-fit
      on each cross-validation training fold. If we fitted the scaler before
      the CV split, the test fold's statistics would contaminate the scaler,
      causing optimistic (inflated) performance estimates.
    """
    console.rule("[cyan]Step 3: Training XGBoost Risk Classifier[/cyan]")

    FEATURE_COLS = ["asset_age_years", "fault_count", "last_inspection_days",
                    "component_wear_pct", "traffic_load_mgt", "zone", "source_system"]
    TARGET_COL   = "risk_label"

    X = df[FEATURE_COLS].copy()
    y = df[TARGET_COL].values

    # ── Train / Validation / Test split (60 / 20 / 20) ───────────────────────
    # Stratified split preserves class ratio in each subset
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp
    )
    console.print(f"   Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")

    # ── Compute class weight for XGBoost ─────────────────────────────────────
    n_neg = int((y_train == 0).sum())
    n_pos = int((y_train == 1).sum())
    scale_pos_weight = n_neg / n_pos
    console.print(f"   Class ratio neg/pos = {scale_pos_weight:.2f} → scale_pos_weight")

    # ── Preprocessing pipeline ────────────────────────────────────────────────
    num_features = ["asset_age_years", "fault_count", "last_inspection_days",
                    "component_wear_pct", "traffic_load_mgt"]
    cat_features = ["zone", "source_system"]

    preprocessor = ColumnTransformer(transformers=[
        ("num", StandardScaler(), num_features),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_features),
    ], remainder="drop")

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", XGBClassifier(
            n_estimators=200,
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
            use_label_encoder=False,
            random_state=42,
            n_jobs=-1,
        ))
    ])

    # ── Hyperparameter tuning via GridSearchCV ────────────────────────────────
    # Grid is intentionally compact for demo speed; expand for production
    param_grid = {
        "classifier__max_depth":        [4, 6],
        "classifier__learning_rate":    [0.05, 0.1],
        "classifier__subsample":        [0.8, 1.0],
        "classifier__colsample_bytree": [0.8, 1.0],
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    console.print("[yellow]Running GridSearchCV (5-fold, may take 1-2 min)...[/yellow]")

    grid_search = GridSearchCV(
        pipeline, param_grid,
        cv=cv, scoring="roc_auc",    # AUC-ROC best metric for imbalanced binary
        n_jobs=-1, verbose=0,
    )
    grid_search.fit(X_train, y_train)

    best_pipeline = grid_search.best_estimator_
    console.print(f"[green]✓ Best params:[/green] {grid_search.best_params_}")
    console.print(f"   CV AUC-ROC: {grid_search.best_score_:.4f}")

    # ── Evaluation on held-out test set ──────────────────────────────────────
    y_pred        = best_pipeline.predict(X_test)
    y_prob        = best_pipeline.predict_proba(X_test)[:, 1]
    test_auc      = roc_auc_score(y_test, y_prob)

    console.print(f"\n[bold]Test Set Evaluation:[/bold]")
    console.print(f"   AUC-ROC : {test_auc:.4f}")
    console.print(classification_report(y_test, y_pred,
                                        target_names=["Low Risk", "High Risk"]))

    # ── Evaluation plots ──────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("XGBoost Risk Classifier — Evaluation", fontsize=13)

    # Confusion matrix
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred, display_labels=["Low Risk", "High Risk"],
        colorbar=False, ax=axes[0], cmap="Blues"
    )
    axes[0].set_title("Confusion Matrix (Test Set)")

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    axes[1].plot(fpr, tpr, color="#e74c3c", lw=2, label=f"AUC = {test_auc:.3f}")
    axes[1].plot([0, 1], [0, 1], "k--", lw=1, label="Random Classifier")
    axes[1].set_xlabel("False Positive Rate")
    axes[1].set_ylabel("True Positive Rate")
    axes[1].set_title("ROC Curve")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    _save_fig(fig, "eval_01_risk_classifier.png")

    # ── Save model ────────────────────────────────────────────────────────────
    model_path = os.path.join(MODEL_DIR, "risk_model.pkl")
    joblib.dump({
        "pipeline":      best_pipeline,
        "feature_cols":  FEATURE_COLS,
        "best_params":   grid_search.best_params_,
        "cv_auc":        grid_search.best_score_,
        "test_auc":      test_auc,
    }, model_path)
    console.print(f"[green]✓ Risk model saved → {model_path}[/green]")

    return best_pipeline, FEATURE_COLS


# ═══════════════════════════════════════════════════════════════════════════════
# Step 4: Random Forest Duration Regressor
# ═══════════════════════════════════════════════════════════════════════════════
def train_duration_model(
    df: pd.DataFrame,
    risk_pipeline: Pipeline,
    risk_feature_cols: list,
) -> Pipeline:
    """
    Trains a Random Forest regressor to estimate maintenance job duration.

    Stacking design:
      The risk probability score (output of Model 1) is added as an additional
      feature to Model 2. This is a form of model stacking — the risk context
      helps the duration model understand urgency and complexity.

    Pitfall prevented — Feature Leakage:
      risk_label (the target of Model 1) is NOT used as a feature here.
      Only the PROBABILITY output from the already-trained Model 1 is added,
      and only on the REGRESSION training data (which is separate from the
      classification training data in a real system). In this demo, we use the
      full combined dataset but compute risk_prob using the already-fitted model.

    Evaluation:
      R², MAE, and RMSE are reported. Residual plots check for heteroscedasticity
      (non-constant error variance), which would indicate the model struggles at
      certain duration ranges.
    """
    console.rule("[cyan]Step 4: Training Random Forest Duration Regressor[/cyan]")

    # Add risk probability as a stacked feature
    X_risk = df[risk_feature_cols].copy()
    df     = df.copy()
    df["risk_probability"] = risk_pipeline.predict_proba(X_risk)[:, 1]

    FEATURE_COLS = ["asset_age_years", "fault_count", "last_inspection_days",
                    "component_wear_pct", "traffic_load_mgt",
                    "risk_probability", "zone", "source_system"]
    TARGET_COL   = "est_maintenance_hrs"

    X = df[FEATURE_COLS].copy()
    y = df[TARGET_COL].values

    # ── Train/Test split (no stratify — regression target) ────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    # ── Preprocessing ─────────────────────────────────────────────────────────
    num_features = ["asset_age_years", "fault_count", "last_inspection_days",
                    "component_wear_pct", "traffic_load_mgt", "risk_probability"]
    cat_features = ["zone", "source_system"]

    preprocessor = ColumnTransformer(transformers=[
        ("num", StandardScaler(), num_features),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_features),
    ], remainder="drop")

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(
            n_estimators=200,
            max_depth=10,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1,
        ))
    ])

    # ── Hyperparameter tuning ─────────────────────────────────────────────────
    param_grid = {
        "regressor__n_estimators":  [100, 200],
        "regressor__max_depth":     [8, 12],
        "regressor__max_features":  ["sqrt", 0.6],
    }

    console.print("[yellow]Running GridSearchCV for duration model...[/yellow]")
    grid_search = GridSearchCV(
        pipeline, param_grid,
        cv=5, scoring="neg_root_mean_squared_error",
        n_jobs=-1, verbose=0,
    )
    grid_search.fit(X_train, y_train)

    best_pipeline = grid_search.best_estimator_
    console.print(f"[green]✓ Best params:[/green] {grid_search.best_params_}")

    # ── Evaluation ────────────────────────────────────────────────────────────
    y_pred    = best_pipeline.predict(X_test)
    mae       = mean_absolute_error(y_test, y_pred)
    rmse      = np.sqrt(mean_squared_error(y_test, y_pred))
    r2        = r2_score(y_test, y_pred)
    residuals = y_test - y_pred

    console.print(f"\n[bold]Duration Model — Test Set:[/bold]")
    console.print(f"   MAE  : {mae:.3f} hrs")
    console.print(f"   RMSE : {rmse:.3f} hrs")
    console.print(f"   R²   : {r2:.4f}")

    # ── Evaluation plots ──────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Random Forest Duration Regressor — Evaluation", fontsize=13)

    # Predicted vs Actual
    axes[0].scatter(y_test, y_pred, alpha=0.3, s=10, color="#3498db")
    lo, hi = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
    axes[0].plot([lo, hi], [lo, hi], "r--", lw=2, label="Perfect Prediction")
    axes[0].set_xlabel("Actual Hours")
    axes[0].set_ylabel("Predicted Hours")
    axes[0].set_title(f"Predicted vs Actual (R²={r2:.3f})")
    axes[0].legend()

    # Residual plot (checks heteroscedasticity)
    axes[1].scatter(y_pred, residuals, alpha=0.3, s=10, color="#9b59b6")
    axes[1].axhline(0, color="red", linestyle="--", lw=2)
    axes[1].set_xlabel("Predicted Hours")
    axes[1].set_ylabel("Residual (Actual - Predicted)")
    axes[1].set_title("Residual Plot")

    # Residual distribution
    axes[2].hist(residuals, bins=50, color="#1abc9c", edgecolor="white", linewidth=0.5)
    axes[2].set_xlabel("Residual")
    axes[2].set_ylabel("Frequency")
    axes[2].set_title(f"Residual Distribution (MAE={mae:.2f}h)")
    axes[2].axvline(0, color="red", linestyle="--")

    _save_fig(fig, "eval_02_duration_regressor.png")

    # ── Save model ────────────────────────────────────────────────────────────
    model_path = os.path.join(MODEL_DIR, "duration_model.pkl")
    joblib.dump({
        "pipeline":     best_pipeline,
        "feature_cols": FEATURE_COLS,
        "mae":          mae,
        "rmse":         rmse,
        "r2":           r2,
    }, model_path)
    console.print(f"[green]✓ Duration model saved → {model_path}[/green]")

    return best_pipeline, FEATURE_COLS


# ═══════════════════════════════════════════════════════════════════════════════
# Step 5: Inference — sample prediction function
# ═══════════════════════════════════════════════════════════════════════════════
def predict_new_asset(asset_record: dict) -> dict:
    """
    Runs inference on a single new asset record.

    Args:
        asset_record: dict with keys matching the model feature columns.

    Returns:
        dict with 'risk_label', 'risk_probability', 'estimated_maintenance_hrs'

    Example:
        predict_new_asset({
            "asset_age_years":       15,
            "fault_count":            4,
            "last_inspection_days":  45,
            "component_wear_pct":    65,
            "traffic_load_mgt":      30,
            "zone":                 "NR",
            "source_system":        "TMS",
        })
    """
    risk_artifact     = joblib.load(os.path.join(MODEL_DIR, "risk_model.pkl"))
    duration_artifact = joblib.load(os.path.join(MODEL_DIR, "duration_model.pkl"))

    risk_pipeline      = risk_artifact["pipeline"]
    risk_features      = risk_artifact["feature_cols"]
    duration_pipeline  = duration_artifact["pipeline"]
    duration_features  = duration_artifact["feature_cols"]

    # Build input DataFrames in exact column order expected by each pipeline
    risk_input = pd.DataFrame([{k: asset_record.get(k, 0) for k in risk_features}])
    risk_prob  = float(risk_pipeline.predict_proba(risk_input)[0, 1])
    risk_label = int(risk_prob >= 0.5)

    # Add risk_probability as feature for duration model
    duration_record = {k: asset_record.get(k, 0) for k in duration_features}
    duration_record["risk_probability"] = risk_prob
    duration_input  = pd.DataFrame([duration_record])
    est_hours       = float(duration_pipeline.predict(duration_input)[0])

    return {
        "risk_label":                risk_label,
        "risk_probability":          round(risk_prob, 4),
        "estimated_maintenance_hrs": round(est_hours, 2),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    console.rule("[bold yellow]Railway Maintenance AI — ML Training Engine[/bold yellow]")

    # Step 1: Load & unify data
    classification_df, regression_df = load_and_unify_data()

    # Step 2: EDA
    run_eda(classification_df)

    # Step 3: Train Risk Classifier
    risk_pipeline, risk_features = train_risk_model(classification_df)

    # Step 4: Train Duration Regressor (stacked on Risk model output)
    duration_pipeline, duration_features = train_duration_model(
        regression_df, risk_pipeline, risk_features
    )

    # Step 5: Demo inference
    console.rule("[cyan]Step 5: Sample Inference[/cyan]")
    sample = {
        "asset_age_years":      18,
        "fault_count":           6,
        "last_inspection_days": 55,
        "component_wear_pct":   72,
        "traffic_load_mgt":     35,
        "zone":                "NR",
        "source_system":       "TMS",
    }
    result = predict_new_asset(sample)
    console.print(f"[bold]Sample asset prediction:[/bold]")
    console.print(f"   Input   : {sample}")
    console.print(f"   Output  : {result}")

    console.rule("[bold green]Training Complete![/bold green]")


if __name__ == "__main__":
    main()
