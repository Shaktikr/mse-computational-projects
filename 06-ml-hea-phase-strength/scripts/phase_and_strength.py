"""Phase classification and yield-strength regression for HEAs / MEAs on the MPEA dataset.

    python scripts/phase_and_strength.py

Steps
  1. Featurise every composition with thermodynamic/electronic descriptors (+ processing).
  2. Phase classification (FCC / BCC / FCC+BCC / IM): empirical rules vs ML models,
     GroupKFold by composition so the same alloy never sits in train and test.
  3. Yield strength vs test temperature (regression), GroupKFold by composition.
  4. Virtual screening: Al_x CoCrFeNi phase evolution, and the two constituents of a
     dual-phase MEA composite, CoCrFeNi (fcc) + AlCoCrFe (bcc).
Outputs in results/.
"""

import json
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.base import clone  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, f1_score, mean_absolute_error, r2_score  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402
from sklearn.neighbors import KNeighborsClassifier  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.svm import SVC  # noqa: E402

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from heaml.composition import alloy_formula  # noqa: E402
from heaml.data import load, phase_table, strength_table  # noqa: E402
from heaml.descriptors import DESCRIPTORS, featurize  # noqa: E402
from heaml.rules import combined_rule  # noqa: E402

plt.rcParams.update({"figure.dpi": 150, "axes.grid": True, "grid.alpha": 0.3})
CLASSES = ["FCC", "BCC", "FCC+BCC", "IM"]
COLORS = {"FCC": "C0", "BCC": "C3", "FCC+BCC": "C4", "IM": "C2"}
PROC = ["CAST", "WROUGHT", "ANNEAL", "POWDER", "OTHER"]
ELEMENTS = ["Al", "Co", "Cr", "Fe", "Ni", "Ti", "V", "Nb", "Mo", "Ta", "W", "Zr", "Hf", "Cu", "Mn"]


def processing_onehot(s: pd.Series) -> pd.DataFrame:
    return pd.DataFrame({f"proc_{p}": (s == p).astype(float) for p in PROC})


def grouped_oof(model, X, y, groups, n_splits=5, proba=False):
    oof = np.zeros((len(y), len(CLASSES))) if proba else np.empty(len(y), dtype=object if y.dtype == object else float)
    for tr, te in GroupKFold(n_splits).split(X, y, groups):
        m = clone(model).fit(X.iloc[tr], y[tr])
        oof[te] = m.predict_proba(X.iloc[te]) if proba else m.predict(X.iloc[te])
    return oof


def main():
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    raw = load()
    report = {"dataset": "MPEA dataset, Borg et al., Sci. Data 7, 430 (2020)", "rows_total": len(raw)}

    # ================================================================ 1-2. phase classification
    ph = phase_table(raw)
    F = featurize(ph["formula"], ELEMENTS)
    X = pd.concat([F, processing_onehot(ph["processing"])], axis=1)
    y = ph["phase"].to_numpy()
    groups = ph["key"].to_numpy()
    report["phase_rows"] = len(ph)
    report["phase_counts"] = ph["phase"].value_counts().to_dict()
    feat_desc = DESCRIPTORS + [f"proc_{p}" for p in PROC]
    feat_all = list(X.columns)

    results = []
    rule = combined_rule(F)
    results.append({"model": "Empirical rules (Yang-Zhang Ω-δ + Guo VEC)", "features": "descriptors",
                    "accuracy": accuracy_score(y, rule), "macro_F1": f1_score(y, rule, average="macro")})
    models = {
        "Logistic regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, C=1.0)),
        "k-nearest neighbours": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=7, weights="distance")),
        "SVM (RBF)": make_pipeline(StandardScaler(), SVC(C=10, gamma="scale", probability=True, random_state=0)),
        "Random forest": RandomForestClassifier(n_estimators=500, min_samples_leaf=1, class_weight="balanced", random_state=0, n_jobs=-1),
        "Gradient boosting": HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05, max_leaf_nodes=15,
                                                            class_weight="balanced", random_state=0),
    }
    preds = {}
    for fs_name, cols in [("descriptors", feat_desc), ("descriptors + composition", feat_all)]:
        for name, model in models.items():
            p = grouped_oof(model, X[cols], y, groups)
            preds[(name, fs_name)] = p
            results.append({"model": name, "features": fs_name, "accuracy": accuracy_score(y, p),
                            "macro_F1": f1_score(y, p, average="macro")})
    res = pd.DataFrame(results).sort_values("macro_F1", ascending=False)
    res.to_csv(out / "phase_classification_metrics.csv", index=False, float_format="%.3f")
    report["phase_classification"] = res.to_dict("records")
    print(res.to_string(index=False))

    best = res.iloc[0]
    best_cols = feat_all if best["features"] == "descriptors + composition" else feat_desc
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    ConfusionMatrixDisplay.from_predictions(y, rule, labels=CLASSES, ax=ax[0], colorbar=False, normalize="true",
                                            values_format=".2f", cmap="Blues")
    ax[0].set_title("Empirical rules", fontsize=9)
    ConfusionMatrixDisplay.from_predictions(y, preds[(best["model"], best["features"])], labels=CLASSES, ax=ax[1],
                                            colorbar=False, normalize="true", values_format=".2f", cmap="Greens")
    ax[1].set_title(f"{best['model']} ({best['features']}), GroupKFold", fontsize=9)
    for a in ax:
        a.grid(False)
    fig.tight_layout()
    fig.savefig(out / "fig_confusion_matrices.png")
    plt.close(fig)

    # classic descriptor maps
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    for cl in CLASSES:
        m = y == cl
        ax[0].scatter(F["delta"][m], F["dHmix"][m], s=10, alpha=0.6, c=COLORS[cl], label=cl)
        ax[1].hist(F["VEC"][m], bins=np.arange(3, 11, 0.25), alpha=0.55, color=COLORS[cl], label=cl)
    ax[0].axvline(6.6, ls="--", c="k", lw=0.8)
    ax[0].set(xlabel="atomic size mismatch δ (%)", ylabel="ΔH$_{mix}$ (kJ/mol)", title="δ – ΔH$_{mix}$ map")
    ax[1].axvline(6.87, ls="--", c="C3", lw=0.8)
    ax[1].axvline(8.0, ls="--", c="C0", lw=0.8)
    ax[1].set(xlabel="valence electron concentration (VEC)", ylabel="count", title="VEC criterion (Guo 2011)")
    for a in ax:
        a.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "fig_descriptor_maps.png")
    plt.close(fig)

    # feature importance (permutation) of a random forest on the full table
    rf = clone(models["Random forest"]).fit(X[best_cols], y)
    from sklearn.inspection import permutation_importance
    pi = permutation_importance(rf, X[best_cols], y, n_repeats=5, random_state=0, scoring="f1_macro")
    imp = pd.Series(pi.importances_mean, index=best_cols).sort_values(ascending=True).tail(15)
    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    imp.plot.barh(ax=ax, color="C2")
    ax.set(xlabel="drop in macro-F1 when shuffled", title="Permutation importance (random forest)")
    fig.tight_layout()
    fig.savefig(out / "fig_phase_feature_importance.png")
    plt.close(fig)
    report["phase_feature_importance"] = imp.sort_values(ascending=False).to_dict()

    # ================================================================ 4. virtual screening
    clf = clone(models["Gradient boosting"]).fit(X[best_cols], y)

    def predict_phase(formulas, proc="CAST"):
        Fx = featurize(formulas, ELEMENTS)
        Xx = pd.concat([Fx, processing_onehot(pd.Series([proc] * len(formulas)))], axis=1)
        return pd.DataFrame(clf.predict_proba(Xx[best_cols]), columns=clf.classes_), Fx

    xs = np.round(np.arange(0.0, 2.01, 0.1), 2)
    alx = [alloy_formula(Al=x, Co=1, Cr=1, Fe=1, Ni=1) for x in xs]
    P, Fx = predict_phase(alx)
    fig, ax = plt.subplots(figsize=(6, 4))
    for cl in CLASSES:
        if cl in P:
            ax.plot(xs, P[cl], "o-", ms=3, color=COLORS[cl], label=cl)
    ax.axvspan(0.5, 0.9, color="C4", alpha=0.08)
    ax.text(0.7, 0.95, "duplex region\nreported in literature", ha="center", va="top", fontsize=7)
    ax.set(xlabel="x in Al$_x$CoCrFeNi", ylabel="predicted probability (as-cast)", ylim=(0, 1.02),
           title="Virtual screening: Al$_x$CoCrFeNi")
    ax2 = ax.twinx()
    ax2.plot(xs, Fx["VEC"], "k:", lw=1)
    ax2.set_ylabel("VEC (dotted)")
    ax2.grid(False)
    ax.legend(fontsize=8, loc="center right")
    fig.tight_layout()
    fig.savefig(out / "fig_screening_AlxCoCrFeNi.png")
    plt.close(fig)
    report["AlxCoCrFeNi_predicted_phase"] = {f"x={x:g}": P.idxmax(axis=1)[i] for i, x in enumerate(xs)}

    comp_alloys = ["Co1 Cr1 Fe1 Ni1", "Al1 Co1 Cr1 Fe1", "Co1 Cr1 Ni1", "Al0.3 Co1 Cr1 Fe1 Ni1"]
    Pc, Fc = predict_phase(comp_alloys, proc="POWDER")
    tab = pd.concat([pd.Series(comp_alloys, name="alloy"), Fc[["VEC", "delta", "dHmix", "Omega"]], Pc.round(3)], axis=1)
    tab.to_csv(out / "screening_mea_composite_constituents.csv", index=False, float_format="%.3f")
    report["mea_composite_constituents"] = tab.to_dict("records")

    # ================================================================ 3. strength vs temperature
    st = strength_table(raw)
    Fs = featurize(st["formula"], ELEMENTS)
    Xs = pd.concat([Fs, processing_onehot(st["processing"]),
                    pd.DataFrame({"test_T_C": st["test_T_C"].to_numpy(),
                                  "compression": (st["test_type"] == "C").astype(float).to_numpy()})], axis=1)
    ys = np.log10(st["YS_MPa"].to_numpy())
    gs = st["key"].to_numpy()
    reg_rows, reg_pred = [], {}
    for name, model in {"Random forest": RandomForestRegressor(n_estimators=500, min_samples_leaf=2, random_state=0, n_jobs=-1),
                        "Gradient boosting": HistGradientBoostingRegressor(max_iter=600, learning_rate=0.05, max_leaf_nodes=15,
                                                                           l2_regularization=1.0, random_state=0)}.items():
        p = grouped_oof(model, Xs, ys, gs)
        reg_pred[name] = p
        reg_rows.append({"model": name, "R2_log10YS": r2_score(ys, p), "MAE_MPa": mean_absolute_error(10**ys, 10**p),
                         "median_abs_pct_error": float(np.median(np.abs(10**p / 10**ys - 1)) * 100)})
    reg = pd.DataFrame(reg_rows)
    reg.to_csv(out / "strength_regression_metrics.csv", index=False, float_format="%.3f")
    report["strength_regression"] = reg.to_dict("records")
    report["strength_rows"] = len(st)
    print(reg.to_string(index=False))

    bestr = reg.sort_values("R2_log10YS").iloc[-1]["model"]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    p = reg_pred[bestr]
    sc = ax[0].scatter(10**ys, 10**p, c=st["test_T_C"], s=10, cmap="plasma", alpha=0.7)
    lim = [30, 4000]
    ax[0].plot(lim, lim, "k-", lw=0.8)
    ax[0].set(xscale="log", yscale="log", xlim=lim, ylim=lim, xlabel="measured YS (MPa)", ylabel="predicted YS (MPa)",
              title=f"{bestr}, GroupKFold (R² = {r2_score(ys, p):.2f} in log YS)")
    plt.colorbar(sc, ax=ax[0], label="test temperature (°C)")

    final = clone(HistGradientBoostingRegressor(max_iter=600, learning_rate=0.05, max_leaf_nodes=15,
                                                l2_regularization=1.0, random_state=0)).fit(Xs, ys)
    Ts = np.linspace(25, 1600, 80)
    for i, (f, proc) in enumerate([("Mo1 Nb1 Ta1 W1", "CAST"), ("Hf1 Nb1 Ta1 Ti1 Zr1", "CAST"), ("Co1 Cr1 Fe1 Ni1", "CAST"),
                                    ("Al1 Co1 Cr1 Fe1 Ni1", "CAST")]):
        Ff = featurize([f] * len(Ts), ELEMENTS)
        Xf = pd.concat([Ff, processing_onehot(pd.Series([proc] * len(Ts))),
                        pd.DataFrame({"test_T_C": Ts, "compression": np.ones_like(Ts)})], axis=1)
        ax[1].plot(Ts, 10 ** final.predict(Xf[Xs.columns]), "-", color=f"C{i}", label=f.replace("1", ""))
        dd = st[(st["key"] == raw.loc[raw["formula"] == f, "key"].iloc[0]) & (st["test_type"] == "C")] if (raw["formula"] == f).any() else st.iloc[:0]
        ax[1].plot(dd["test_T_C"], dd["YS_MPa"], "o", color=f"C{i}", ms=4)
    ax[1].set(xlabel="test temperature (°C)", ylabel="yield strength (MPa)", yscale="log",
              title="High-temperature strength (lines: model, dots: data)")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "fig_strength.png")
    plt.close(fig)

    (out / "report.json").write_text(json.dumps(report, indent=2, default=float))
    print(f"results in {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
