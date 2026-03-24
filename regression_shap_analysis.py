"""
Logistic Regression with SHAP Interpretation for Bank Customer Data.
Uses feature selection (L1) to reduce variables; LinearExplainer for SHAP.
"""

import os
import re
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix

warnings.filterwarnings("ignore")

DESKTOP = Path(os.path.expanduser("~")) / "Desktop"
OUTPUT_DIR = Path(__file__).parent / "shap_results"
SUPPORTED_EXTENSIONS = [".csv", ".xlsx", ".xls", ".data", ".data-numeric", ".txt"]

# Number of features to keep after selection (most important only)
N_FEATURES_SELECT = 15


def find_data_on_desktop() -> Path | None:
    candidates = []
    for path in DESKTOP.rglob("*"):
        if path.suffix.lower() in [".csv", ".xlsx", ".xls"]:
            candidates.append(path)
        elif path.suffix in [".data", ".txt"] or path.name.endswith(".data-numeric"):
            candidates.append(path)
    if not candidates:
        return None
    # Prefer larger datasets (more columns/rows) - peek first row/col count
    def size_key(p):
        try:
            if p.suffix.lower() == ".csv":
                df = pd.read_csv(p, nrows=5)
            elif p.suffix.lower() in [".xlsx", ".xls"]:
                df = pd.read_excel(p, nrows=5)
            else:
                df = pd.read_csv(p, sep=r"\s+", header=None, nrows=5)
            return (df.shape[1], len(pd.read_csv(p, nrows=0).columns) if p.suffix.lower() == ".csv" and df.shape[1] > 0 else df.shape[1])
        except Exception:
            return (0, 0)
    try:
        candidates.sort(key=lambda p: (pd.read_excel(p).shape[1] if p.suffix.lower() in [".xlsx", ".xls"] else (pd.read_csv(p, nrows=1).shape[1] if p.suffix.lower() == ".csv" else 10), 0), reverse=True)
    except Exception:
        pass
    # Sort by number of columns (desc), then by path
    def cols_rows(p):
        try:
            if p.suffix.lower() in [".xlsx", ".xls"]:
                d = pd.read_excel(p)
            elif p.suffix.lower() == ".csv":
                d = pd.read_csv(p)
            else:
                d = pd.read_csv(p, sep=r"\s+", header=None)
            return (d.shape[1], d.shape[0])
        except Exception:
            return (0, 0)
    candidates.sort(key=cols_rows, reverse=True)
    return candidates[0]


def load_german_data_numeric(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=r"\s+", header=None)
    df.columns = [f"X{i}" for i in range(len(df.columns) - 1)] + ["target"]
    df["target"] = df["target"].replace({2: 0})
    return df


def load_german_data_raw(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=r"\s+", header=None)
    col_names = [
        "checking_status", "duration", "credit_history", "purpose", "credit_amount",
        "savings", "employment", "installment_rate", "personal_status", "other_parties",
        "residence_since", "property_magnitude", "age", "other_plans", "housing",
        "num_credits", "job", "num_dependents", "phone", "foreign_worker", "target"
    ]
    if len(df.columns) == 21:
        df.columns = col_names
    else:
        df.columns = [f"X{i}" for i in range(len(df.columns) - 1)] + ["target"]
    df["target"] = df["target"].replace({2: 0})
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = pd.Categorical(df[col]).codes
    return df


def load_csv_or_excel(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    return pd.read_excel(path)


def load_data(path: Path) -> pd.DataFrame:
    name_lower = path.name.lower()
    if "german" in name_lower and ("numeric" in name_lower or path.suffix == ".data-numeric"):
        return load_german_data_numeric(path)
    if "german" in name_lower and path.suffix == ".data":
        return load_german_data_raw(path)
    return load_csv_or_excel(path)


def prepare_data(df: pd.DataFrame, target_col: str | None = None) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    df = df.dropna()
    if target_col and target_col in df.columns:
        y = df[target_col]
        X = df.drop(columns=[target_col])
    else:
        for name in ["target", "Target", "class", "outcome", "y", "label"]:
            if name in df.columns:
                target_col = name
                break
        if target_col is None:
            target_col = df.columns[-1]
        y = df[target_col]
        X = df.drop(columns=[target_col])
    X = X.select_dtypes(include=[np.number])
    if X.empty:
        raise ValueError("No numeric features found. Encode categorical columns first.")
    feature_names = X.columns.tolist()
    return X, y, feature_names


def select_features_l1(X_train: np.ndarray, y_train, feature_names: list[str], n_select: int) -> tuple[np.ndarray, np.ndarray, list[str], list[int]]:
    """Select top n_select features using L1 (Lasso) logistic regression."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    model_l1 = LogisticRegression(penalty="l1", solver="saga", max_iter=2000, random_state=42, C=0.5)
    model_l1.fit(X_scaled, y_train)
    coef = np.abs(model_l1.coef_).ravel()
    top_idx = np.argsort(coef)[-n_select:][::-1]
    selected_names = [feature_names[i] for i in top_idx]
    X_train_sel = X_train[:, top_idx]
    X_scaled_sel = X_scaled[:, top_idx]
    return X_train_sel, X_scaled_sel, selected_names, top_idx.tolist()


def fit_logistic(X_train, y_train, X_test, y_test):
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    model = LogisticRegression(penalty="l2", solver="lbfgs", max_iter=2000, random_state=42, C=1.0)
    model.fit(X_train_s, y_train)
    y_pred = model.predict(X_test_s)
    acc = accuracy_score(y_test, y_pred)
    try:
        auc = roc_auc_score(y_test, model.predict_proba(X_test_s)[:, 1])
        print(f"  Accuracy: {acc:.4f}  |  AUC-ROC: {auc:.4f}")
    except Exception:
        print(f"  Accuracy: {acc:.4f}")
    return model, scaler


def run_shap_linear(model, X_background, X_explain, feature_names: list[str], output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    # Explainer uses LinearExplainer for logistic regression (exact SHAP for linear models)
    masker = shap.maskers.Independent(X_background)
    explainer = shap.LinearExplainer(model, masker, feature_names=feature_names)
    shap_values = explainer(X_explain)

    sv = shap_values.values
    if len(sv.shape) == 3:
        sv = sv[:, :, -1]
    mean_abs_shap = np.abs(sv).mean(axis=0)

    shap.summary_plot(shap_values, X_explain, feature_names=feature_names, show=False)
    plt.gcf().savefig(output_dir / "shap_summary_beeswarm.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: shap_summary_beeswarm.png")

    shap.summary_plot(shap_values, X_explain, feature_names=feature_names, plot_type="bar", show=False)
    plt.gcf().savefig(output_dir / "shap_importance_bar.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: shap_importance_bar.png")

    top_indices = np.argsort(mean_abs_shap)[-4:][::-1]
    for i in range(min(4, len(top_indices))):
        idx = int(top_indices[i])
        fname = feature_names[idx]
        safe_name = re.sub(r'[<>:"/\\|?*]', "_", fname).strip()[:80]
        try:
            shap.dependence_plot(idx, sv, X_explain, feature_names=feature_names, interaction_index=None, show=False)
            plt.gcf().savefig(output_dir / f"shap_dependence_{safe_name}.png", dpi=150, bbox_inches="tight")
            plt.close()
            print(f"  Saved: shap_dependence_{safe_name}.png")
        except Exception as e:
            print(f"  Skipped dependence plot for {fname}: {e}")

    shap_df = pd.DataFrame(sv, columns=[f"SHAP_{n}" for n in feature_names])
    shap_df.to_csv(output_dir / "shap_values.csv", index=False)
    print(f"  Saved: shap_values.csv")

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": mean_abs_shap
    }).sort_values("mean_abs_shap", ascending=False)
    importance_df.to_csv(output_dir / "shap_feature_importance.csv", index=False)
    print(f"  Saved: shap_feature_importance.csv")

    return explainer, shap_values


def main():
    print("=" * 60)
    print("Logistic Regression with SHAP (feature selection applied)")
    print("=" * 60)

    data_path = find_data_on_desktop()
    if data_path is None:
        print("\nNo data file found on Desktop.")
        manual = input("Enter full path to data file (or Enter to exit): ").strip()
        if not manual:
            sys.exit(1)
        data_path = Path(manual)

    if not data_path.exists():
        print(f"File not found: {data_path}")
        sys.exit(1)

    try:
        print(f"\nData file: {data_path}")
    except UnicodeEncodeError:
        print("\nData file: [path with non-ASCII characters]")

    try:
        df = load_data(data_path)
        print(f"Loaded: {len(df)} rows, {len(df.columns)} columns")
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)

    try:
        X, y, feature_names = prepare_data(df)
        print(f"Initial features: {len(feature_names)} | Target: {y.name}")
        print(f"Target distribution:\n{y.value_counts()}")
    except Exception as e:
        print(f"Error preparing data: {e}")
        sys.exit(1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    n_select = min(N_FEATURES_SELECT, X_train.shape[1])
    print(f"\nFeature selection: keeping top {n_select} features (L1 logistic regression)")

    X_train_sel, X_train_scaled_sel, selected_names, selected_idx = select_features_l1(
        X_train.values, y_train, feature_names, n_select
    )
    X_test_sel = X_test.values[:, selected_idx]
    print(f"Selected features: {selected_names}")

    print("\nFitting logistic regression (L2) on selected features...")
    model, scaler = fit_logistic(X_train_sel, y_train, X_test_sel, y_test)

    X_train_final = scaler.transform(X_train_sel)
    X_test_final = scaler.transform(X_test_sel)

    print("\nComputing SHAP values (LinearExplainer)...")
    explainer, shap_values = run_shap_linear(
        model, X_train_final, X_test_final, selected_names, OUTPUT_DIR
    )

    # Save model summary and selected feature list for report
    summary_path = OUTPUT_DIR / "model_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"Model: Logistic Regression (L2)\n")
        f.write(f"Features selected: {n_select}\n")
        f.write(f"Selected: {selected_names}\n")
        f.write(f"Classification report:\n{classification_report(y_test, model.predict(X_test_final))}\n")
        f.write(f"Confusion matrix:\n{confusion_matrix(y_test, model.predict(X_test_final))}\n")
    print(f"  Saved: model_summary.txt")

    print("\n" + "=" * 60)
    print(f"Done. Results saved to: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
