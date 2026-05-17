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
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, silhouette_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


DATA_DIR = Path("../data")
TRAIN_PATH = DATA_DIR / "train.csv"
RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_DIR = Path("outputs/reports/06_classification_possibility") / RUN_ID
FIG_DIR = OUT_DIR / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
SPECIES_NUMBER_COL = "species number"
SPECIES_NAME_COL = "樹種"
SAMPLE_COL = "sample number"
MOISTURE_COL = "含水率"

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
KL_LABEL_MAPPING = {"conifer": "N", "broadleaf": "L"}


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


def format_axis(ax, unit: str):
    ax.set_xlabel("Wavenumber (cm$^{-1}$)" if unit == "cm-1" else "Wavelength (nm)")
    if unit == "cm-1":
        ax.invert_xaxis()


def wavenumber_to_nm(x):
    arr = np.asarray(x, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return 1e7 / arr


def nm_to_wavenumber(x):
    arr = np.asarray(x, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return 1e7 / arr


def format_spectral_axis_with_wavelength(ax, unit: str):
    format_axis(ax, unit)
    if unit == "cm-1":
        secax = ax.secondary_xaxis("top", functions=(wavenumber_to_nm, nm_to_wavenumber))
        secax.set_xlabel("Wavelength (nm)")
        secax.tick_params(axis="x", labelsize=8)


def robust_ylim(ax, values, lower=0.01, upper=0.99, pad=0.08, symmetric=False):
    arr = np.asarray(values, dtype=float).ravel()
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return
    lo, hi = np.quantile(arr, [lower, upper])
    if symmetric:
        m = max(abs(lo), abs(hi))
        lo, hi = -m, m
    span = hi - lo
    if span == 0:
        span = max(abs(hi), 1.0)
    ax.set_ylim(lo - span * pad, hi + span * pad)


def apply_snv_deriv2(X: pd.DataFrame | np.ndarray) -> np.ndarray:
    return SavitzkyGolayTransformer(15, 2, 2).fit_transform(SNVTransformer().fit_transform(X))


def make_pipeline(preprocess: str, model_name: str) -> Pipeline:
    steps = []
    if preprocess == "raw_scaled":
        steps.append(("scale", StandardScaler()))
    elif preprocess == "snv":
        steps.extend([("snv", SNVTransformer()), ("scale", StandardScaler())])
    elif preprocess == "snv_deriv2":
        steps.extend(
            [
                ("snv", SNVTransformer()),
                ("sg2", SavitzkyGolayTransformer(15, 2, 2)),
                ("scale", StandardScaler()),
            ]
        )
    else:
        raise ValueError(preprocess)

    if model_name == "logreg":
        model = LogisticRegression(max_iter=3000, class_weight="balanced", random_state=RANDOM_STATE)
    elif model_name == "linear_svc":
        model = LinearSVC(C=1.0, class_weight="balanced", random_state=RANDOM_STATE)
    elif model_name == "random_forest":
        model = RandomForestClassifier(
            n_estimators=220,
            max_features="sqrt",
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE,
            n_jobs=1,
        )
    else:
        raise ValueError(model_name)
    steps.append(("model", model))
    return Pipeline(steps)


def cv_for(y: pd.Series) -> StratifiedKFold:
    min_count = int(y.value_counts().min())
    return StratifiedKFold(n_splits=min(3, min_count), shuffle=True, random_state=RANDOM_STATE)


def evaluate_models(df: pd.DataFrame, X: pd.DataFrame, task_col: str, task_name: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = []
    predictions = {}
    y = df[task_col].astype(str)
    cv = cv_for(y)
    for preprocess in ["raw_scaled", "snv", "snv_deriv2"]:
        for model_name in ["logreg", "linear_svc", "random_forest"]:
            pipe = make_pipeline(preprocess, model_name)
            scores = cross_validate(
                pipe,
                X,
                y,
                cv=cv,
                scoring={"balanced_accuracy": "balanced_accuracy", "f1_macro": "f1_macro", "accuracy": "accuracy"},
                n_jobs=1,
                error_score=np.nan,
            )
            rows.append(
                {
                    "task": task_name,
                    "preprocess": preprocess,
                    "model": model_name,
                    "balanced_accuracy": float(np.nanmean(scores["test_balanced_accuracy"])),
                    "f1_macro": float(np.nanmean(scores["test_f1_macro"])),
                    "accuracy": float(np.nanmean(scores["test_accuracy"])),
                }
            )
    score_df = pd.DataFrame(rows).sort_values("balanced_accuracy", ascending=False)
    best = score_df.iloc[0]
    best_pipe = make_pipeline(best["preprocess"], best["model"])
    pred = cross_val_predict(best_pipe, X, y, cv=cv, n_jobs=1)
    labels = sorted(y.unique())
    cm = pd.DataFrame(confusion_matrix(y, pred, labels=labels), index=labels, columns=labels)
    class_rows = []
    for label in labels:
        mask = y == label
        class_rows.append(
            {
                "task": task_name,
                "class": label,
                "n": int(mask.sum()),
                "recall": float((pred[mask] == label).mean()),
            }
        )
    return score_df, cm, pd.DataFrame(class_rows).sort_values("recall")


def pairwise_species(df: pd.DataFrame, X_proc: np.ndarray) -> pd.DataFrame:
    rows = []
    labels = sorted(df["species_display"].unique())
    for a_i, a in enumerate(labels):
        for b in labels[a_i + 1 :]:
            mask = df["species_display"].isin([a, b]).to_numpy()
            y = df.loc[mask, "species_display"].astype(str)
            if y.value_counts().min() < 3:
                continue
            X_pair = X_proc[mask]
            pipe = Pipeline(
                [
                    ("pca", PCA(n_components=min(10, X_pair.shape[0] - 1, X_pair.shape[1]), random_state=RANDOM_STATE)),
                    ("model", LogisticRegression(max_iter=1500, class_weight="balanced", random_state=RANDOM_STATE)),
                ]
            )
            pred = cross_val_predict(pipe, X_pair, y, cv=cv_for(y), n_jobs=1)
            rows.append(
                {
                    "class_a": a,
                    "class_b": b,
                    "n_a": int((y == a).sum()),
                    "n_b": int((y == b).sum()),
                    "balanced_accuracy": balanced_accuracy_score(y, pred),
                    "f1_macro": f1_score(y, pred, average="macro"),
                }
            )
    return pd.DataFrame(rows).sort_values("balanced_accuracy")


def top_feature_table(X_values: np.ndarray, y: pd.Series, wavelengths: np.ndarray, task: str, top_n=20) -> pd.DataFrame:
    f_values, p_values = f_classif(X_values, y.astype(str))
    order = np.argsort(np.nan_to_num(f_values, nan=-np.inf))[::-1][:top_n]
    return pd.DataFrame(
        {
            "task": task,
            "wavenumber_cm-1": wavelengths[order],
            "wavelength_nm": wavenumber_to_nm(wavelengths[order]),
            "f_value": f_values[order],
            "p_value": p_values[order],
        }
    )


font = configure_plot_font()
df = read_csv(TRAIN_PATH)
spec_cols = spectral_columns(df)
wavelengths = np.array([float(c) for c in spec_cols])
axis_unit = "cm-1" if np.nanmedian(wavelengths) > 2500 else "nm"

df["species_label"] = df[SPECIES_NUMBER_COL].astype(int).astype(str)
df["species_display"] = df[SPECIES_NUMBER_COL].astype(int).astype(str) + "_" + df[SPECIES_NAME_COL].astype(str)
df["wood_type"] = df[SPECIES_NUMBER_COL].astype(int).map(WOOD_TYPE_MAPPING)
df["kl_label"] = df["wood_type"].map(KL_LABEL_MAPPING)
X = df[spec_cols].astype(float)
X_snv = SNVTransformer().fit_transform(X)
X_snv_deriv2 = apply_snv_deriv2(X)

species_mapping = df[[SPECIES_NUMBER_COL, SPECIES_NAME_COL, "wood_type", "kl_label"]].drop_duplicates().sort_values(SPECIES_NUMBER_COL)
species_counts = df["species_display"].value_counts().rename_axis("species").reset_index(name="n")
kl_counts = df["kl_label"].value_counts().rename_axis("kl").reset_index(name="n")
species_mapping.to_csv(OUT_DIR / "species_kl_mapping.csv", index=False, encoding="utf-8-sig")
species_counts.to_csv(OUT_DIR / "species_counts.csv", index=False, encoding="utf-8-sig")
kl_counts.to_csv(OUT_DIR / "kl_counts.csv", index=False, encoding="utf-8-sig")

model_scores = []
confusion_paths = {}
class_recalls = []
for task_name, task_col in [("species", "species_display"), ("kl", "kl_label")]:
    scores, cm, recalls = evaluate_models(df, X, task_col, task_name)
    model_scores.append(scores)
    class_recalls.append(recalls)
    cm.to_csv(OUT_DIR / f"{task_name}_confusion_matrix.csv", encoding="utf-8-sig")
    confusion_paths[task_name] = str(OUT_DIR / f"{task_name}_confusion_matrix.csv")
model_scores = pd.concat(model_scores, ignore_index=True)
class_recalls = pd.concat(class_recalls, ignore_index=True)
model_scores.to_csv(OUT_DIR / "classification_cv_scores.csv", index=False, encoding="utf-8-sig")
class_recalls.to_csv(OUT_DIR / "class_recalls.csv", index=False, encoding="utf-8-sig")

pairwise_df = pairwise_species(df, X_snv_deriv2)
pairwise_df.to_csv(OUT_DIR / "species_pairwise_separability.csv", index=False, encoding="utf-8-sig")

top_features = pd.concat(
    [
        top_feature_table(X_snv_deriv2, df["species_display"], wavelengths, "species"),
        top_feature_table(X_snv_deriv2, df["kl_label"], wavelengths, "kl"),
    ],
    ignore_index=True,
)
top_features.to_csv(OUT_DIR / "top_discriminative_wavenumbers.csv", index=False, encoding="utf-8-sig")

k_mask = df["kl_label"].eq("N").to_numpy()
l_mask = df["kl_label"].eq("L").to_numpy()
k_deriv = X_snv_deriv2[k_mask]
l_deriv = X_snv_deriv2[l_mask]
k_mean = k_deriv.mean(axis=0)
l_mean = l_deriv.mean(axis=0)
k_std = k_deriv.std(axis=0, ddof=1)
l_std = l_deriv.std(axis=0, ddof=1)
pooled_std = np.sqrt(((k_deriv.shape[0] - 1) * k_std**2 + (l_deriv.shape[0] - 1) * l_std**2) / (k_deriv.shape[0] + l_deriv.shape[0] - 2))
pooled_std_safe = np.where(pooled_std == 0, np.nan, pooled_std)
standardized_diff = (l_mean - k_mean) / pooled_std_safe
variance_df = pd.DataFrame(
    {
        "wavenumber_cm-1": wavelengths,
        "wavelength_nm": wavenumber_to_nm(wavelengths),
        "k_mean_snv_deriv2": k_mean,
        "l_mean_snv_deriv2": l_mean,
        "k_std_snv_deriv2": k_std,
        "l_std_snv_deriv2": l_std,
        "pooled_std_snv_deriv2": pooled_std,
        "l_minus_k": l_mean - k_mean,
        "standardized_l_minus_k": standardized_diff,
        "abs_standardized_l_minus_k": np.abs(standardized_diff),
    }
)
variance_df.sort_values("abs_standardized_l_minus_k", ascending=False).to_csv(
    OUT_DIR / "kl_standardized_difference_by_wavenumber.csv", index=False, encoding="utf-8-sig"
)

fig_paths = {}

fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)
sns.barplot(data=species_counts, y="species", x="n", ax=axes[0], color="#4C78A8")
axes[0].set_title("樹種別サンプル数")
axes[0].set_xlabel("サンプル数")
axes[0].set_ylabel("")
sns.barplot(data=kl_counts, x="kl", y="n", ax=axes[1], color="#59A14F")
axes[1].set_title("N/L材サンプル数")
axes[1].set_xlabel("N/L")
axes[1].set_ylabel("サンプル数")
fig_paths["counts"] = str(FIG_DIR / "class_counts.png")
fig.savefig(fig_paths["counts"], dpi=180, bbox_inches="tight")
plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
for label, idx in df.groupby("kl_label").groups.items():
    axes[0].plot(wavelengths, X.loc[idx].mean(axis=0), label=label, linewidth=1.5)
    axes[1].plot(wavelengths, X_snv_deriv2[list(idx)].mean(axis=0), label=label, linewidth=1.5)
axes[0].set_title("N/L材別 平均スペクトル（raw）")
axes[0].set_ylabel("吸光度/信号")
format_spectral_axis_with_wavelength(axes[0], axis_unit)
axes[1].set_title("N/L材別 平均スペクトル（SNV + 2次微分）")
axes[1].set_ylabel("前処理後信号")
format_spectral_axis_with_wavelength(axes[1], axis_unit)
robust_ylim(axes[1], X_snv_deriv2, lower=0.01, upper=0.99, symmetric=True)
for ax in axes:
    ax.legend(title="N/L")
fig_paths["kl_mean_spectra"] = str(FIG_DIR / "kl_mean_spectra.png")
fig.savefig(fig_paths["kl_mean_spectra"], dpi=180, bbox_inches="tight")
plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
for label, idx in df.groupby("kl_label").groups.items():
    raw_group = X.loc[idx].to_numpy()
    proc_group = X_snv_deriv2[list(idx)]
    raw_mean = raw_group.mean(axis=0)
    raw_std = raw_group.std(axis=0, ddof=1)
    proc_mean = proc_group.mean(axis=0)
    proc_std = proc_group.std(axis=0, ddof=1)
    axes[0].plot(wavelengths, raw_mean, label=f"{label} mean", linewidth=1.4)
    axes[0].fill_between(wavelengths, raw_mean - raw_std, raw_mean + raw_std, alpha=0.16)
    axes[1].plot(wavelengths, proc_mean, label=f"{label} mean", linewidth=1.4)
    axes[1].fill_between(wavelengths, proc_mean - proc_std, proc_mean + proc_std, alpha=0.16)
axes[0].set_title("N/L材別 平均 ± 1SD（raw）")
axes[0].set_ylabel("吸光度/信号")
format_spectral_axis_with_wavelength(axes[0], axis_unit)
axes[1].set_title("N/L材別 平均 ± 1SD（SNV + 2次微分）")
axes[1].set_ylabel("前処理後信号")
format_spectral_axis_with_wavelength(axes[1], axis_unit)
robust_ylim(axes[1], np.concatenate([k_mean - k_std, k_mean + k_std, l_mean - l_std, l_mean + l_std]), lower=0.02, upper=0.98, symmetric=True)
for ax in axes:
    ax.legend(title="N/L")
fig_paths["kl_mean_sd"] = str(FIG_DIR / "kl_mean_sd_spectra.png")
fig.savefig(fig_paths["kl_mean_sd"], dpi=180, bbox_inches="tight")
plt.close(fig)

fig, ax = plt.subplots(figsize=(11, 4.8), constrained_layout=True)
k_mean = X_snv_deriv2[df["kl_label"].eq("N").to_numpy()].mean(axis=0)
l_mean = X_snv_deriv2[df["kl_label"].eq("L").to_numpy()].mean(axis=0)
ax.plot(wavelengths, l_mean - k_mean, color="#B07AA1", linewidth=1.3)
ax.axhline(0, color="black", linewidth=0.8)
ax.set_title("L材 - N材 平均差スペクトル（SNV + 2次微分）")
ax.set_ylabel("平均差")
format_spectral_axis_with_wavelength(ax, axis_unit)
robust_ylim(ax, l_mean - k_mean, lower=0.01, upper=0.99, symmetric=True)
fig_paths["kl_difference"] = str(FIG_DIR / "kl_difference_spectrum.png")
fig.savefig(fig_paths["kl_difference"], dpi=180, bbox_inches="tight")
plt.close(fig)

fig, axes = plt.subplots(2, 1, figsize=(11, 7), constrained_layout=True, sharex=True)
axes[0].plot(wavelengths, variance_df["standardized_l_minus_k"], color="#E15759", linewidth=1.1)
axes[0].axhline(0, color="black", linewidth=0.8)
axes[0].axhline(2, color="#777777", linewidth=0.8, linestyle="--")
axes[0].axhline(-2, color="#777777", linewidth=0.8, linestyle="--")
axes[0].set_title("L材-N材 標準化平均差（平均差 / pooled SD）")
axes[0].set_ylabel("標準化差")
format_spectral_axis_with_wavelength(axes[0], axis_unit)
axes[1].plot(wavelengths, variance_df["pooled_std_snv_deriv2"], color="#4C78A8", linewidth=1.1)
axes[1].set_title("N/L材内ばらつき（pooled SD）")
axes[1].set_ylabel("pooled SD")
format_spectral_axis_with_wavelength(axes[1], axis_unit)
robust_ylim(axes[1], variance_df["pooled_std_snv_deriv2"], lower=0.01, upper=0.97)
fig_paths["kl_standardized_diff"] = str(FIG_DIR / "kl_standardized_difference.png")
fig.savefig(fig_paths["kl_standardized_diff"], dpi=180, bbox_inches="tight")
plt.close(fig)

pca = PCA(n_components=5, random_state=RANDOM_STATE)
scores = pca.fit_transform(X_snv)
pca_df = pd.DataFrame({"PC1": scores[:, 0], "PC2": scores[:, 1], "species": df["species_display"], "kl": df["kl_label"]})
if MOISTURE_COL in df.columns:
    pca_df[MOISTURE_COL] = pd.to_numeric(df[MOISTURE_COL], errors="coerce").values
pca_metrics = {
    "species_silhouette_pc2": float(silhouette_score(scores[:, :2], df["species_display"].astype(str))),
    "species_silhouette_pc5": float(silhouette_score(scores[:, :5], df["species_display"].astype(str))),
    "kl_silhouette_pc2": float(silhouette_score(scores[:, :2], df["kl_label"].astype(str))),
    "kl_silhouette_pc5": float(silhouette_score(scores[:, :5], df["kl_label"].astype(str))),
    "pc1_explained_variance": float(pca.explained_variance_ratio_[0]),
    "pc2_explained_variance": float(pca.explained_variance_ratio_[1]),
    "pc5_cumulative_explained_variance": float(pca.explained_variance_ratio_[:5].sum()),
}
if MOISTURE_COL in df.columns:
    moisture = pd.to_numeric(df[MOISTURE_COL], errors="coerce")
    for i in range(scores.shape[1]):
        pca_metrics[f"pc{i + 1}_moisture_corr"] = float(pd.Series(scores[:, i]).corr(moisture))
pca_metrics_df = pd.DataFrame([pca_metrics])
pca_metrics_df.to_csv(OUT_DIR / "pca_separability_metrics.csv", index=False, encoding="utf-8-sig")
pca_loading_rows = []
for pc_idx in range(2):
    loading = pca.components_[pc_idx]
    order = np.argsort(np.abs(loading))[::-1][:15]
    pca_loading_rows.append(
        pd.DataFrame(
            {
                "pc": f"PC{pc_idx + 1}",
                "wavenumber_cm-1": wavelengths[order],
                "wavelength_nm": wavenumber_to_nm(wavelengths[order]),
                "loading": loading[order],
                "abs_loading": np.abs(loading[order]),
            }
        )
    )
pca_loading_df = pd.concat(pca_loading_rows, ignore_index=True)
pca_loading_df.to_csv(OUT_DIR / "pca_top_loadings.csv", index=False, encoding="utf-8-sig")
fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
sns.scatterplot(data=pca_df, x="PC1", y="PC2", hue="species", s=28, alpha=0.85, ax=axes[0])
axes[0].set_title("PCAスコア（樹種）")
axes[0].legend(fontsize=7, bbox_to_anchor=(1.02, 1), loc="upper left")
sns.scatterplot(data=pca_df, x="PC1", y="PC2", hue="kl", s=38, alpha=0.85, ax=axes[1])
axes[1].set_title("PCAスコア（N/L材）")
for ax in axes:
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)")
fig_paths["pca"] = str(FIG_DIR / "pca_species_kl.png")
fig.savefig(fig_paths["pca"], dpi=180, bbox_inches="tight")
plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)
pc_numbers = np.arange(1, len(pca.explained_variance_ratio_) + 1)
axes[0].bar(pc_numbers, pca.explained_variance_ratio_ * 100, color="#4C78A8")
axes[0].plot(pc_numbers, np.cumsum(pca.explained_variance_ratio_) * 100, marker="o", color="#E15759")
axes[0].set_title("PCA寄与率")
axes[0].set_xlabel("PC")
axes[0].set_ylabel("Explained variance (%)")
axes[1].plot(wavelengths, pca.components_[0], label="PC1 loading", linewidth=1.0)
axes[1].plot(wavelengths, pca.components_[1], label="PC2 loading", linewidth=1.0)
axes[1].set_title("PCAローディング（PC1/PC2）")
axes[1].set_ylabel("Loading")
format_spectral_axis_with_wavelength(axes[1], axis_unit)
axes[1].legend()
fig_paths["pca_scree_loadings"] = str(FIG_DIR / "pca_scree_loadings.png")
fig.savefig(fig_paths["pca_scree_loadings"], dpi=180, bbox_inches="tight")
plt.close(fig)

fig, ax = plt.subplots(figsize=(7, 5.5), constrained_layout=True)
sns.scatterplot(data=pca_df, x="PC1", y="PC2", hue="kl", s=36, alpha=0.45, ax=ax)
centroids = pca_df.groupby("kl")[["PC1", "PC2"]].mean().reset_index()
sns.scatterplot(data=centroids, x="PC1", y="PC2", hue="kl", s=260, marker="X", edgecolor="black", linewidth=1.2, ax=ax, legend=False)
for _, row in centroids.iterrows():
    ax.text(row["PC1"], row["PC2"], f"  {row['kl']} centroid", va="center", fontsize=10)
ax.set_title("PCA上のN/L材セントロイド")
ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)")
ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)")
fig_paths["pca_kl_centroids"] = str(FIG_DIR / "pca_kl_centroids.png")
fig.savefig(fig_paths["pca_kl_centroids"], dpi=180, bbox_inches="tight")
plt.close(fig)

if MOISTURE_COL in pca_df.columns:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    sc = axes[0].scatter(pca_df["PC1"], pca_df["PC2"], c=pca_df[MOISTURE_COL], cmap="viridis", s=28, alpha=0.85)
    axes[0].set_title("PCAスコア（含水率で着色）")
    axes[0].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)")
    axes[0].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)")
    fig.colorbar(sc, ax=axes[0], label=MOISTURE_COL)
    sns.regplot(data=pca_df, x="PC1", y=MOISTURE_COL, scatter_kws={"s": 24, "alpha": 0.65}, line_kws={"color": "#E15759"}, ax=axes[1])
    axes[1].set_title(f"PC1と{MOISTURE_COL}の関係")
    axes[1].set_xlabel("PC1 score")
    axes[1].set_ylabel(MOISTURE_COL)
    fig_paths["pca_moisture"] = str(FIG_DIR / "pca_moisture_relationship.png")
    fig.savefig(fig_paths["pca_moisture"], dpi=180, bbox_inches="tight")
    plt.close(fig)

fig, ax = plt.subplots(figsize=(11, 5.5), constrained_layout=True)
plot_scores = model_scores.sort_values(["task", "balanced_accuracy"], ascending=[True, False]).copy()
plot_scores["setting"] = plot_scores["preprocess"] + " / " + plot_scores["model"]
sns.barplot(data=plot_scores, y="setting", x="balanced_accuracy", hue="task", ax=ax)
ax.set_title("代表モデルによる3-fold CV分類性能")
ax.set_xlabel("Balanced accuracy")
ax.set_ylabel("")
fig_paths["cv_scores"] = str(FIG_DIR / "cv_scores.png")
fig.savefig(fig_paths["cv_scores"], dpi=180, bbox_inches="tight")
plt.close(fig)

for task_name in ["species", "kl"]:
    cm = pd.read_csv(OUT_DIR / f"{task_name}_confusion_matrix.csv", index_col=0)
    fig, ax = plt.subplots(figsize=(8 if task_name == "kl" else 11, 6 if task_name == "kl" else 9), constrained_layout=True)
    sns.heatmap(cm, annot=True, fmt=".0f", cmap="Blues", ax=ax)
    ax.set_title(f"混同行列（{task_name}）")
    ax.set_xlabel("予測")
    ax.set_ylabel("実測")
    fig_paths[f"{task_name}_cm"] = str(FIG_DIR / f"{task_name}_confusion_matrix.png")
    fig.savefig(fig_paths[f"{task_name}_cm"], dpi=180, bbox_inches="tight")
    plt.close(fig)

fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
top_kl = top_features[top_features["task"].eq("kl")].head(12).sort_values("f_value")
top_kl_labels = [
    f"{wn:.0f} cm$^{{-1}}$ / {wl:.0f} nm"
    for wn, wl in zip(top_kl["wavenumber_cm-1"], top_kl["wavelength_nm"])
]
sns.barplot(data=top_kl, y=top_kl_labels, x="f_value", ax=ax, color="#E15759")
ax.set_title("N/L分類に寄与しやすい波数候補（ANOVA F値）")
ax.set_xlabel("F値")
ax.set_ylabel("波数 / 波長")
fig_paths["top_kl_wavenumbers"] = str(FIG_DIR / "top_kl_wavenumbers.png")
fig.savefig(fig_paths["top_kl_wavenumbers"], dpi=180, bbox_inches="tight")
plt.close(fig)

summary = {
    "run_id": RUN_ID,
    "out_dir": str(OUT_DIR),
    "figure_paths": fig_paths,
    "font": font,
    "n_samples": int(len(df)),
    "n_spectral_features": int(len(spec_cols)),
    "axis_min": float(wavelengths.min()),
    "axis_max": float(wavelengths.max()),
    "axis_unit": axis_unit,
    "species_n_classes": int(df["species_display"].nunique()),
    "kl_counts": kl_counts.to_dict(orient="records"),
    "species_best": model_scores[model_scores.task.eq("species")].sort_values("balanced_accuracy", ascending=False).head(1).to_dict(orient="records")[0],
    "kl_best": model_scores[model_scores.task.eq("kl")].sort_values("balanced_accuracy", ascending=False).head(1).to_dict(orient="records")[0],
    "hard_species_pairs": pairwise_df.head(8).to_dict(orient="records"),
    "easy_species_pairs": pairwise_df.tail(8).to_dict(orient="records"),
    "low_recall_classes": class_recalls.groupby("task").head(6).to_dict(orient="records"),
    "top_kl_wavenumbers": top_features[top_features.task.eq("kl")].head(10).to_dict(orient="records"),
    "top_kl_standardized_differences": variance_df.sort_values("abs_standardized_l_minus_k", ascending=False).head(10).to_dict(orient="records"),
    "pca_metrics": pca_metrics,
    "pca_top_loadings": pca_loading_df.head(10).to_dict(orient="records"),
}
(OUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(summary, ensure_ascii=False, indent=2))
