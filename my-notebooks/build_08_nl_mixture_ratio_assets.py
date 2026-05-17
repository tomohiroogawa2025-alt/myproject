from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("outputs/.matplotlib_cache").resolve()))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import font_manager
from scipy.signal import savgol_filter
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


DATA_DIR = Path("../data")
TRAIN_PATH = DATA_DIR / "train.csv"
RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_DIR = Path("outputs/reports/08_nl_mixture_ratio_estimation") / RUN_ID
FIG_DIR = OUT_DIR / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
SPECIES_NUMBER_COL = "species number"
SPECIES_NAME_COL = "樹種"
SAMPLE_COL = "sample number"
MOISTURE_COL = "含水率"
MOISTURE_TOLERANCE = 3.0
MAX_MATCHED_PAIRS = 300
MIX_RATIOS = np.linspace(0, 1, 11)
SPECTRAL_RANGE_NM = (900.0, 1700.0)
MOISTURE_BINS = [0, 15, 30, 60, 120, np.inf]
MOISTURE_BIN_LABELS = ["0-15", "15-30", "30-60", "60-120", "120+"]

WOOD_TYPE_MAPPING = {
    1: "conifer",
    3: "broadleaf",
    4: "broadleaf",
    5: "broadleaf",
    8: "conifer",
    11: "broadleaf",
    12: "broadleaf",
    13: "broadleaf",
    14: "conifer",
    15: "conifer",
    16: "conifer",
    17: "conifer",
    19: "broadleaf",
}
NL_LABEL_MAPPING = {"conifer": "N", "broadleaf": "L"}


def configure_plot_font() -> str:
    candidates = [
        "Meiryo",
        "Meiriyo",
        "Hiragino Sans",
        "Hiragino Maru Gothic Pro",
        "Hiragino Mincho ProN",
        "Yu Gothic",
        "YuGothic",
        "Noto Sans CJK JP",
        "Noto Sans JP",
        "IPAexGothic",
        "TakaoGothic",
        "Osaka",
    ]
    available = {f.name for f in font_manager.fontManager.ttflist}
    chosen = next((font for font in candidates if font in available), "DejaVu Sans")
    rc = {
        "font.family": "sans-serif",
        "font.sans-serif": [chosen],
        "figure.figsize": (10, 5),
        "axes.unicode_minus": False,
    }
    sns.set_theme(style="whitegrid", rc=rc)
    plt.rcParams.update(rc)
    return chosen


class SNVTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        arr = np.asarray(X, dtype=float)
        mean = arr.mean(axis=1, keepdims=True)
        std = arr.std(axis=1, keepdims=True)
        std[std == 0] = 1.0
        return (arr - mean) / std


class SavitzkyGolayTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, window_length=15, polyorder=2, deriv=0):
        self.window_length = window_length
        self.polyorder = polyorder
        self.deriv = deriv

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        arr = np.asarray(X, dtype=float)
        n_features = arr.shape[1]
        win = min(self.window_length, n_features if n_features % 2 == 1 else n_features - 1)
        if win <= self.polyorder:
            return arr
        if win % 2 == 0:
            win -= 1
        return savgol_filter(arr, window_length=win, polyorder=self.polyorder, deriv=self.deriv, axis=1)


def read_csv(path: Path) -> pd.DataFrame:
    for enc in ["utf-8-sig", "utf-8", "cp932", "shift_jis"]:
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path)


def spectral_columns(df: pd.DataFrame) -> list[str]:
    cols = []
    for col in df.columns:
        try:
            float(col)
            cols.append(col)
        except ValueError:
            pass
    return cols


def wavenumber_to_nm(x):
    arr = np.asarray(x, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return 1e7 / arr


def nm_to_wavenumber(x):
    arr = np.asarray(x, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return 1e7 / arr


def format_axis(ax, unit: str):
    ax.set_xlabel("Wavenumber (cm$^{-1}$)" if unit == "cm-1" else "Wavelength (nm)")
    if unit == "cm-1":
        ax.invert_xaxis()
        secax = ax.secondary_xaxis("top", functions=(wavenumber_to_nm, nm_to_wavenumber))
        secax.set_xlabel("Wavelength (nm)")
        secax.tick_params(axis="x", labelsize=8)


def make_unique_moisture_pairs(df: pd.DataFrame) -> pd.DataFrame:
    candidates_n = df[df["nl_label"].eq("N")].copy()
    candidates_l = df[df["nl_label"].eq("L")].copy()
    candidates_n["_moisture"] = pd.to_numeric(candidates_n[MOISTURE_COL], errors="coerce")
    candidates_l["_moisture"] = pd.to_numeric(candidates_l[MOISTURE_COL], errors="coerce")
    candidates_n = candidates_n.dropna(subset=["_moisture"])
    candidates_l = candidates_l.dropna(subset=["_moisture"])

    rng = np.random.default_rng(RANDOM_STATE)
    n_indices = list(candidates_n.index)
    rng.shuffle(n_indices)
    unused_l = set(candidates_l.index)
    rows = []

    for n_idx in n_indices:
        if not unused_l:
            break
        n_moisture = float(candidates_n.loc[n_idx, "_moisture"])
        l_pool = candidates_l.loc[list(unused_l), "_moisture"]
        l_idx = (l_pool - n_moisture).abs().idxmin()
        diff = abs(float(candidates_l.loc[l_idx, "_moisture"]) - n_moisture)
        if diff <= MOISTURE_TOLERANCE:
            unused_l.remove(l_idx)
            rows.append(
                {
                    "pair_id": len(rows),
                    "n_index": int(n_idx),
                    "l_index": int(l_idx),
                    "n_sample": candidates_n.loc[n_idx, SAMPLE_COL],
                    "l_sample": candidates_l.loc[l_idx, SAMPLE_COL],
                    "n_species": candidates_n.loc[n_idx, SPECIES_NAME_COL],
                    "l_species": candidates_l.loc[l_idx, SPECIES_NAME_COL],
                    "n_moisture": n_moisture,
                    "l_moisture": float(candidates_l.loc[l_idx, "_moisture"]),
                    "moisture_abs_diff": diff,
                }
            )
    return pd.DataFrame(rows)


def build_mixture_dataset(df: pd.DataFrame, spec_cols: list[str], pairs: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    X_raw = df[spec_cols].astype(float).to_numpy()
    meta_rows = []
    spectra = []
    for _, row in pairs.iterrows():
        n_spec = X_raw[int(row["n_index"])]
        l_spec = X_raw[int(row["l_index"])]
        for n_ratio in MIX_RATIOS:
            mixed = n_ratio * n_spec + (1.0 - n_ratio) * l_spec
            spectra.append(mixed)
            meta_rows.append(
                {
                    "pair_id": int(row["pair_id"]),
                    "n_ratio": float(n_ratio),
                    "l_ratio": float(1.0 - n_ratio),
                    "n_ratio_percent": float(n_ratio * 100.0),
                    "l_ratio_percent": float((1.0 - n_ratio) * 100.0),
                    "n_sample": row["n_sample"],
                    "l_sample": row["l_sample"],
                    "n_species": row["n_species"],
                    "l_species": row["l_species"],
                    "n_moisture": row["n_moisture"],
                    "l_moisture": row["l_moisture"],
                    "moisture_abs_diff": row["moisture_abs_diff"],
                    "mixed_moisture_linear": float(n_ratio * row["n_moisture"] + (1.0 - n_ratio) * row["l_moisture"]),
                }
            )
    return pd.DataFrame(meta_rows), np.asarray(spectra, dtype=float), np.asarray([float(c) for c in spec_cols]), np.asarray([float(r) for r in MIX_RATIOS])


def select_900_1700(spec_cols: list[str], wavenumbers: np.ndarray, axis_unit: str) -> tuple[list[str], np.ndarray]:
    nm = wavenumber_to_nm(wavenumbers) if axis_unit == "cm-1" else wavenumbers
    mask = (nm >= SPECTRAL_RANGE_NM[0]) & (nm <= SPECTRAL_RANGE_NM[1])
    return [col for col, keep in zip(spec_cols, mask) if keep], mask


def make_pipeline(preprocess: str, model_name: str, n_features: int) -> Pipeline:
    steps = []
    if preprocess == "raw_scaled":
        steps.append(("scale", StandardScaler()))
    elif preprocess == "snv":
        steps.extend([("snv", SNVTransformer()), ("scale", StandardScaler())])
    elif preprocess == "snv_deriv1":
        steps.extend([("snv", SNVTransformer()), ("sg1", SavitzkyGolayTransformer(15, 2, 1)), ("scale", StandardScaler())])
    elif preprocess == "snv_deriv2":
        steps.extend([("snv", SNVTransformer()), ("sg2", SavitzkyGolayTransformer(15, 2, 2)), ("scale", StandardScaler())])
    else:
        raise ValueError(preprocess)

    if model_name.startswith("pls_"):
        n_components = min(int(model_name.split("_")[1]), n_features)
        model = PLSRegression(n_components=n_components)
    elif model_name == "ridge":
        model = Ridge(alpha=10.0)
    elif model_name == "svr_rbf":
        model = SVR(C=20.0, epsilon=2.0, gamma="scale")
    elif model_name == "random_forest":
        model = RandomForestRegressor(n_estimators=240, max_features="sqrt", min_samples_leaf=2, random_state=RANDOM_STATE, n_jobs=1)
    else:
        raise ValueError(model_name)

    steps.append(("model", model))
    return Pipeline(steps)


def evaluate_models(X: np.ndarray, y: np.ndarray, groups: np.ndarray, range_name: str) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    rows = []
    preds = {}
    cv = GroupKFold(n_splits=5)
    for preprocess in ["raw_scaled", "snv", "snv_deriv1", "snv_deriv2"]:
        for model_name in ["pls_2", "pls_5", "pls_10", "ridge"]:
            pipe = make_pipeline(preprocess, model_name, X.shape[1])
            pred = cross_val_predict(pipe, X, y, groups=groups, cv=cv, n_jobs=1).ravel()
            pred = np.clip(pred, 0, 100)
            key = f"{range_name}__{preprocess}__{model_name}"
            preds[key] = pred
            rmse = float(np.sqrt(mean_squared_error(y, pred)))
            mae = float(mean_absolute_error(y, pred))
            rows.append(
                {
                    "range": range_name,
                    "preprocess": preprocess,
                    "model": model_name,
                    "rmse_percent_point": rmse,
                    "mae_percent_point": mae,
                    "r2": float(r2_score(y, pred)),
                    "bias_percent_point": float(np.mean(pred - y)),
                    "within_5pt_rate": float(np.mean(np.abs(pred - y) <= 5.0)),
                    "within_10pt_rate": float(np.mean(np.abs(pred - y) <= 10.0)),
                }
            )
    return pd.DataFrame(rows).sort_values(["rmse_percent_point", "mae_percent_point"]), preds


def add_moisture_bin(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["moisture_bin"] = pd.cut(
        out["mixed_moisture_linear"],
        bins=MOISTURE_BINS,
        labels=MOISTURE_BIN_LABELS,
        right=False,
    )
    out["moisture_bin"] = out["moisture_bin"].astype(str)
    return out


def summarize_error_by_moisture(df: pd.DataFrame, pred_col: str, err_col: str) -> pd.DataFrame:
    rows = []
    for label in MOISTURE_BIN_LABELS:
        part = df[df["moisture_bin"].eq(label)]
        if part.empty:
            continue
        err = part[err_col].to_numpy(dtype=float)
        rows.append(
            {
                "moisture_bin": label,
                "n": int(len(part)),
                "moisture_min": float(part["mixed_moisture_linear"].min()),
                "moisture_max": float(part["mixed_moisture_linear"].max()),
                "mean_moisture": float(part["mixed_moisture_linear"].mean()),
                "rmse_percent_point": float(np.sqrt(np.mean(err**2))),
                "mae_percent_point": float(np.mean(np.abs(err))),
                "bias_percent_point": float(np.mean(err)),
                "within_5pt_rate": float(np.mean(np.abs(err) <= 5.0)),
                "within_10pt_rate": float(np.mean(np.abs(err) <= 10.0)),
                "mean_pred": float(part[pred_col].mean()),
                "mean_actual": float(part["n_ratio_percent"].mean()),
            }
        )
    return pd.DataFrame(rows)


def save_methodology():
    text = """# 08 N材/L材混合比率推定 方法論メモ

目的は、N材/L材が混ざって作られたパルプシートについて、スペクトルからN材比率またはL材比率を推定できるかを評価することである。

## 基本方針

- 分類モデルの確率は、物理的な混合比率そのものではないため、最終的には既知配合比の校正試料を作り、比率を目的変数とした回帰モデルとして構築する。
- パルプシート状態で既に混ざっている前提では、カメラによる空間分布推定ではなく、測定スペクトルから平均的な混合比率を推定するモデル設計を優先する。
- 本08では、現時点のN材/L材単独スペクトルから含水率が近いペアを作り、rawスペクトルを線形混合した疑似混合スペクトルで予備的な可能性評価を行う。
- 疑似混合は実混合パルプの散乱、繊維長、密度、叩解、配向、非線形性を完全には再現しない。したがって、ここでの結果は「モデル上、N/L差が比率情報として使えそうか」の予備評価である。

## 実試料で次に必要なこと

- N/L = 0/100, 10/90, ..., 100/0 など既知配合比のパルプシートを作る。
- ロット、含水率、シート密度、坪量、叩解条件などを変え、比率以外の変動をモデルに学習させる。
- PLS回帰を基準モデルにし、Ridge、SVR、GPR、RandomForest回帰などと比較する。
- 評価はRMSE、MAE、bias、R²、±5%以内/±10%以内の割合で行い、ロットや作製日をまたぐ外部検証を重視する。
"""
    (OUT_DIR / "methodology.md").write_text(text, encoding="utf-8")


def main():
    font = configure_plot_font()
    df = read_csv(TRAIN_PATH)
    spec_cols = spectral_columns(df)
    wavenumbers = np.asarray([float(c) for c in spec_cols])
    axis_unit = "cm-1" if np.nanmedian(wavenumbers) > 2500 else "nm"

    df["wood_type"] = df[SPECIES_NUMBER_COL].astype(int).map(WOOD_TYPE_MAPPING)
    df["nl_label"] = df["wood_type"].map(NL_LABEL_MAPPING)
    df = df.dropna(subset=["nl_label"]).copy()
    pairs = make_unique_moisture_pairs(df)
    if len(pairs) < 20:
        raise ValueError("Not enough moisture-matched N/L pairs for mixture simulation.")
    pairs = pairs.sort_values("moisture_abs_diff").head(MAX_MATCHED_PAIRS).reset_index(drop=True)
    pairs["pair_id"] = np.arange(len(pairs))

    meta, X_mix, wavenumbers, ratios = build_mixture_dataset(df, spec_cols, pairs)
    meta = add_moisture_bin(meta)
    y = meta["n_ratio_percent"].to_numpy()
    groups = meta["pair_id"].to_numpy()

    spec_cols_900_1700, range_mask = select_900_1700(spec_cols, wavenumbers, axis_unit)
    X_mix_900_1700 = X_mix[:, range_mask]
    wavenumbers_900_1700 = wavenumbers[range_mask]

    scores_full, preds_full = evaluate_models(X_mix, y, groups, "full")
    scores_900_1700, preds_900_1700 = evaluate_models(X_mix_900_1700, y, groups, "900-1700nm")
    scores = pd.concat([scores_full, scores_900_1700], ignore_index=True).sort_values("rmse_percent_point")
    pred_map = {**preds_full, **preds_900_1700}

    best = scores.iloc[0]
    best_key = f"{best['range']}__{best['preprocess']}__{best['model']}"
    meta_with_pred = meta.copy()
    meta_with_pred["pred_n_ratio_percent_best"] = pred_map[best_key]
    meta_with_pred["error_percent_point_best"] = meta_with_pred["pred_n_ratio_percent_best"] - meta_with_pred["n_ratio_percent"]

    best_by_range = {
        range_name: scores[scores["range"].eq(range_name)].iloc[0].to_dict()
        for range_name in ["full", "900-1700nm"]
    }
    range_prediction_paths = {}
    range_error_paths = {}
    range_moisture_error_paths = {}
    range_fig_paths = {}
    for range_name, range_best in best_by_range.items():
        range_key = f"{range_best['range']}__{range_best['preprocess']}__{range_best['model']}"
        range_pred_df = meta.copy()
        safe_range = range_name.replace("-", "_").replace(" ", "_")
        pred_col = "pred_n_ratio_percent"
        err_col = "error_percent_point"
        range_pred_df[pred_col] = pred_map[range_key]
        range_pred_df[err_col] = range_pred_df[pred_col] - range_pred_df["n_ratio_percent"]
        pred_path = OUT_DIR / f"mixture_ratio_predictions_{safe_range}_best.csv"
        range_pred_df.to_csv(pred_path, index=False, encoding="utf-8-sig")
        range_prediction_paths[range_name] = str(pred_path)
        error_by_ratio = (
            range_pred_df.groupby("n_ratio_percent")
            .agg(
                n=(err_col, "size"),
                mean_pred=(pred_col, "mean"),
                mae=(err_col, lambda s: float(np.mean(np.abs(s)))),
                bias=(err_col, "mean"),
                p95_abs_error=(err_col, lambda s: float(np.quantile(np.abs(s), 0.95))),
            )
            .reset_index()
        )
        error_path = OUT_DIR / f"mixture_ratio_error_by_ratio_{safe_range}.csv"
        error_by_ratio.to_csv(error_path, index=False, encoding="utf-8-sig")
        range_error_paths[range_name] = str(error_path)
        moisture_error = summarize_error_by_moisture(range_pred_df, pred_col, err_col)
        moisture_error_path = OUT_DIR / f"mixture_ratio_error_by_moisture_{safe_range}.csv"
        moisture_error.to_csv(moisture_error_path, index=False, encoding="utf-8-sig")
        range_moisture_error_paths[range_name] = str(moisture_error_path)
        range_fig_paths[range_name] = {}

    ratio_error = (
        meta_with_pred.groupby("n_ratio_percent")
        .agg(
            n=("error_percent_point_best", "size"),
            mean_pred=("pred_n_ratio_percent_best", "mean"),
            mae=("error_percent_point_best", lambda s: float(np.mean(np.abs(s)))),
            bias=("error_percent_point_best", "mean"),
            p95_abs_error=("error_percent_point_best", lambda s: float(np.quantile(np.abs(s), 0.95))),
        )
        .reset_index()
    )

    save_methodology()
    pairs.to_csv(OUT_DIR / "moisture_matched_nl_pairs.csv", index=False, encoding="utf-8-sig")
    meta.to_csv(OUT_DIR / "synthetic_mixture_metadata.csv", index=False, encoding="utf-8-sig")
    scores.to_csv(OUT_DIR / "mixture_ratio_cv_scores.csv", index=False, encoding="utf-8-sig")
    meta_with_pred.to_csv(OUT_DIR / "mixture_ratio_predictions_best.csv", index=False, encoding="utf-8-sig")
    ratio_error.to_csv(OUT_DIR / "mixture_ratio_error_by_ratio.csv", index=False, encoding="utf-8-sig")

    fig_paths = {}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    sns.histplot(pairs["moisture_abs_diff"], bins=30, ax=axes[0], color="#4C78A8")
    axes[0].set_title("N/Lペアの含水率差")
    axes[0].set_xlabel("含水率差（絶対値）")
    axes[0].set_ylabel("ペア数")
    sns.countplot(data=meta, x="n_ratio_percent", ax=axes[1], color="#59A14F")
    axes[1].set_title("疑似混合スペクトルのN材比率")
    axes[1].set_xlabel("N材比率 (%)")
    axes[1].set_ylabel("スペクトル数")
    axes[1].tick_params(axis="x", rotation=45)
    fig_paths["mixture_design"] = str(FIG_DIR / "mixture_design.png")
    fig.savefig(fig_paths["mixture_design"], dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    sns.histplot(meta["mixed_moisture_linear"], bins=35, ax=axes[0], color="#4C78A8")
    axes[0].set_title("疑似混合スペクトルの含水率域")
    axes[0].set_xlabel("線形混合後の推定含水率")
    axes[0].set_ylabel("スペクトル数")
    moisture_counts = meta["moisture_bin"].value_counts().reindex(MOISTURE_BIN_LABELS).fillna(0).reset_index()
    moisture_counts.columns = ["moisture_bin", "n"]
    sns.barplot(data=moisture_counts, x="moisture_bin", y="n", ax=axes[1], color="#59A14F")
    axes[1].set_title("含水率域ごとの疑似混合スペクトル数")
    axes[1].set_xlabel("含水率域")
    axes[1].set_ylabel("スペクトル数")
    fig_paths["moisture_distribution"] = str(FIG_DIR / "mixture_moisture_distribution.png")
    fig.savefig(fig_paths["moisture_distribution"], dpi=180, bbox_inches="tight")
    plt.close(fig)

    example_pair = pairs.sort_values("moisture_abs_diff").iloc[0]
    ex_meta, ex_spec, _, _ = build_mixture_dataset(df, spec_cols, pd.DataFrame([example_pair]))
    fig, ax = plt.subplots(figsize=(11, 5), constrained_layout=True)
    palette = sns.color_palette("viridis", len(MIX_RATIOS))
    for i, ratio in enumerate(MIX_RATIOS):
        ax.plot(wavenumbers, ex_spec[i], color=palette[i], linewidth=1.0, label=f"N {ratio * 100:.0f}%")
    ax.set_title("含水率が近いN/Lペアから作成した疑似混合スペクトル例")
    ax.set_ylabel("吸光度/信号")
    format_axis(ax, axis_unit)
    ax.legend(ncol=4, fontsize=8, title="混合比")
    fig_paths["example_mixed_spectra"] = str(FIG_DIR / "example_mixed_spectra.png")
    fig.savefig(fig_paths["example_mixed_spectra"], dpi=180, bbox_inches="tight")
    plt.close(fig)

    top_scores = scores.head(12).copy()
    top_scores["setting"] = top_scores["range"] + " / " + top_scores["preprocess"] + " / " + top_scores["model"]
    fig, ax = plt.subplots(figsize=(11, 5.5), constrained_layout=True)
    sns.barplot(data=top_scores, y="setting", x="rmse_percent_point", ax=ax, color="#4C78A8")
    ax.set_title("N材比率推定モデルのCV RMSE（上位12条件）")
    ax.set_xlabel("RMSE（percentage point）")
    ax.set_ylabel("")
    fig_paths["model_comparison"] = str(FIG_DIR / "model_comparison_rmse.png")
    fig.savefig(fig_paths["model_comparison"], dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 6), constrained_layout=True)
    sns.scatterplot(
        data=meta_with_pred,
        x="n_ratio_percent",
        y="pred_n_ratio_percent_best",
        hue="moisture_abs_diff",
        palette="viridis",
        s=18,
        edgecolor=None,
        ax=ax,
    )
    ax.plot([0, 100], [0, 100], color="black", linewidth=1, linestyle="--")
    ax.set_xlim(-3, 103)
    ax.set_ylim(-3, 103)
    ax.set_title("実際のN材比率 vs 推定N材比率（疑似混合CV）")
    ax.set_xlabel("実際のN材比率 (%)")
    ax.set_ylabel("推定N材比率 (%)")
    fig_paths["actual_vs_predicted"] = str(FIG_DIR / "actual_vs_predicted_best.png")
    fig.savefig(fig_paths["actual_vs_predicted"], dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    sns.boxplot(data=meta_with_pred, x="n_ratio_percent", y="error_percent_point_best", ax=axes[0], color="#F28E2B")
    axes[0].axhline(0, color="black", linewidth=1)
    axes[0].set_title("N材比率ごとの推定誤差")
    axes[0].set_xlabel("N材比率 (%)")
    axes[0].set_ylabel("推定誤差（percentage point）")
    axes[0].tick_params(axis="x", rotation=45)
    sns.lineplot(data=ratio_error, x="n_ratio_percent", y="mae", marker="o", ax=axes[1], color="#E15759")
    axes[1].set_title("N材比率ごとのMAE")
    axes[1].set_xlabel("N材比率 (%)")
    axes[1].set_ylabel("MAE（percentage point）")
    fig_paths["error_by_ratio"] = str(FIG_DIR / "error_by_ratio.png")
    fig.savefig(fig_paths["error_by_ratio"], dpi=180, bbox_inches="tight")
    plt.close(fig)

    for range_name, range_best in best_by_range.items():
        safe_range = range_name.replace("-", "_").replace(" ", "_")
        range_pred_df = pd.read_csv(range_prediction_paths[range_name])
        error_by_ratio = pd.read_csv(range_error_paths[range_name])

        fig, ax = plt.subplots(figsize=(6.2, 6), constrained_layout=True)
        sns.scatterplot(
            data=range_pred_df,
            x="n_ratio_percent",
            y="pred_n_ratio_percent",
            hue="moisture_abs_diff",
            palette="viridis",
            s=18,
            edgecolor=None,
            ax=ax,
        )
        ax.plot([0, 100], [0, 100], color="black", linewidth=1, linestyle="--")
        ax.set_xlim(-3, 103)
        ax.set_ylim(-3, 103)
        ax.set_title(f"{range_name}: 実際のN材比率 vs 推定N材比率")
        ax.set_xlabel("実際のN材比率 (%)")
        ax.set_ylabel("推定N材比率 (%)")
        range_fig_paths[range_name]["actual_vs_predicted"] = str(FIG_DIR / f"actual_vs_predicted_{safe_range}.png")
        fig.savefig(range_fig_paths[range_name]["actual_vs_predicted"], dpi=180, bbox_inches="tight")
        plt.close(fig)

        fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
        sns.boxplot(data=range_pred_df, x="n_ratio_percent", y="error_percent_point", ax=axes[0], color="#F28E2B")
        axes[0].axhline(0, color="black", linewidth=1)
        axes[0].set_title(f"{range_name}: N材比率ごとの推定誤差")
        axes[0].set_xlabel("N材比率 (%)")
        axes[0].set_ylabel("推定誤差（percentage point）")
        axes[0].tick_params(axis="x", rotation=45)
        sns.lineplot(data=error_by_ratio, x="n_ratio_percent", y="mae", marker="o", ax=axes[1], color="#E15759")
        axes[1].set_title(f"{range_name}: N材比率ごとのMAE")
        axes[1].set_xlabel("N材比率 (%)")
        axes[1].set_ylabel("MAE（percentage point）")
        range_fig_paths[range_name]["error_by_ratio"] = str(FIG_DIR / f"error_by_ratio_{safe_range}.png")
        fig.savefig(range_fig_paths[range_name]["error_by_ratio"], dpi=180, bbox_inches="tight")
        plt.close(fig)

        moisture_error = pd.read_csv(range_moisture_error_paths[range_name])
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
        sns.barplot(data=moisture_error, x="moisture_bin", y="rmse_percent_point", ax=axes[0], color="#4C78A8")
        axes[0].set_title(f"{range_name}: 含水率域ごとのRMSE")
        axes[0].set_xlabel("含水率域")
        axes[0].set_ylabel("RMSE（percentage point）")
        sns.barplot(data=moisture_error, x="moisture_bin", y="mae_percent_point", ax=axes[1], color="#E15759")
        axes[1].set_title(f"{range_name}: 含水率域ごとのMAE")
        axes[1].set_xlabel("含水率域")
        axes[1].set_ylabel("MAE（percentage point）")
        range_fig_paths[range_name]["error_by_moisture"] = str(FIG_DIR / f"error_by_moisture_{safe_range}.png")
        fig.savefig(range_fig_paths[range_name]["error_by_moisture"], dpi=180, bbox_inches="tight")
        plt.close(fig)

    summary = {
        "run_id": RUN_ID,
        "font": font,
        "n_source_samples": int(len(df)),
        "n_matched_pairs": int(len(pairs)),
        "moisture_tolerance": MOISTURE_TOLERANCE,
        "max_matched_pairs": MAX_MATCHED_PAIRS,
        "mean_pair_moisture_abs_diff": float(pairs["moisture_abs_diff"].mean()),
        "p90_pair_moisture_abs_diff": float(pairs["moisture_abs_diff"].quantile(0.90)),
        "n_synthetic_spectra": int(len(meta)),
        "mix_ratios_percent": [float(r * 100) for r in MIX_RATIOS],
        "moisture_bins": MOISTURE_BIN_LABELS,
        "mixed_moisture_min": float(meta["mixed_moisture_linear"].min()),
        "mixed_moisture_max": float(meta["mixed_moisture_linear"].max()),
        "mixed_moisture_mean": float(meta["mixed_moisture_linear"].mean()),
        "mixed_moisture_median": float(meta["mixed_moisture_linear"].median()),
        "n_full_features": int(X_mix.shape[1]),
        "n_900_1700_features": int(X_mix_900_1700.shape[1]),
        "selected_900_1700_nm_min": float(np.nanmin(wavenumber_to_nm(wavenumbers_900_1700) if axis_unit == "cm-1" else wavenumbers_900_1700)),
        "selected_900_1700_nm_max": float(np.nanmax(wavenumber_to_nm(wavenumbers_900_1700) if axis_unit == "cm-1" else wavenumbers_900_1700)),
        "best_model": best.to_dict(),
        "best_by_range": best_by_range,
        "figures": fig_paths,
        "range_figures": range_fig_paths,
        "outputs": {
            "methodology": str(OUT_DIR / "methodology.md"),
            "pairs": str(OUT_DIR / "moisture_matched_nl_pairs.csv"),
            "scores": str(OUT_DIR / "mixture_ratio_cv_scores.csv"),
            "predictions_best": str(OUT_DIR / "mixture_ratio_predictions_best.csv"),
            "error_by_ratio": str(OUT_DIR / "mixture_ratio_error_by_ratio.csv"),
            "range_predictions": range_prediction_paths,
            "range_error_by_ratio": range_error_paths,
            "range_error_by_moisture": range_moisture_error_paths,
        },
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(OUT_DIR)
    print(json.dumps(summary["best_model"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
