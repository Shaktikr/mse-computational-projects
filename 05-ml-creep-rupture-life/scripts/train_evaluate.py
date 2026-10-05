"""Train, validate and interpret creep-rupture-life models; then do inverse design.

    python scripts/train_evaluate.py                       # synthetic superalloy data
    python scripts/train_evaluate.py --data data/my.csv    # your own data (same columns)

Outputs (results/):
    metrics_grouped_cv.csv     every model, 5-fold GroupKFold by alloy
    leakage_demo.csv           random KFold vs GroupKFold for the same models
    metrics_extrapolation.csv  train T <= 950 C, test T > 950 C
    fig_parity.png, fig_extrapolation.png, fig_uncertainty.png, fig_shap.png
    design_candidates.csv      top compositions from GP-UCB search (+ true life if synthetic)
    report.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.base import clone  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from creepml.data import GROUP, TARGET, generate_superalloy_dataset, life_hours_true, load_csv  # noqa: E402
from creepml.design import ucb_search  # noqa: E402
from creepml.features import add_physics_features, feature_columns  # noqa: E402
from creepml.models import conformal_interval, cross_validate, extrapolation_split, metrics, model_zoo  # noqa: E402

plt.rcParams.update({"figure.dpi": 150, "axes.grid": True, "grid.alpha": 0.3})


def parity(ax, y, yhat, title, color="C0"):
    lo, hi = min(y.min(), yhat.min()) - 0.2, max(y.max(), yhat.max()) + 0.2
    ax.plot([lo, hi], [lo, hi], "k-", lw=0.8)
    ax.fill_between([lo, hi], [lo - np.log10(2), hi - np.log10(2)], [lo + np.log10(2), hi + np.log10(2)],
                    color="0.85", label="factor of 2")
    ax.scatter(y, yhat, s=6, alpha=0.6, color=color)
    m = metrics(y, yhat)
    ax.set(xlim=(lo, hi), ylim=(lo, hi), xlabel="measured log$_{10}$ t$_r$ (h)", ylabel="predicted log$_{10}$ t$_r$ (h)",
           title=f"{title}\nR² = {m['R2']:.3f}, RMSE = {m['RMSE_log10']:.3f}")
    ax.title.set_fontsize(8)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=None)
    ap.add_argument("--design-T", type=float, default=1000.0)
    ap.add_argument("--design-stress", type=float, default=200.0)
    args = ap.parse_args()
    out = ROOT / "results"
    out.mkdir(exist_ok=True)

    synthetic = args.data is None
    if synthetic:
        df = generate_superalloy_dataset()
        df.to_csv(ROOT / "data" / "synthetic_superalloy_creep.csv", index=False)
    else:
        df = load_csv(args.data)
    df = add_physics_features(df)
    y = df[TARGET].to_numpy()
    groups = df[GROUP].to_numpy()
    cols = feature_columns(physics=True)
    zoo = model_zoo(cols)
    report = {"synthetic_data": synthetic, "n_tests": len(df), "n_alloys": int(df[GROUP].nunique()), "features": cols}
    print(f"{len(df)} tests, {df[GROUP].nunique()} alloys, {len(cols)} features")

    # ------------------------------------------------------------ grouped CV, all models
    rows, oof = [], {}
    for name, model in zoo.items():
        t0 = time.time()
        oof[name] = cross_validate(model, df, y, groups)
        rows.append({"model": name, **metrics(y, oof[name]), "time_s": time.time() - t0})
        print(f"  {name:34s} R2={rows[-1]['R2']:.3f} RMSE={rows[-1]['RMSE_log10']:.3f} ({rows[-1]['time_s']:.0f}s)")
    cv = pd.DataFrame(rows).sort_values("RMSE_log10")
    cv.to_csv(out / "metrics_grouped_cv.csv", index=False, float_format="%.4f")
    report["grouped_cv"] = cv.to_dict("records")

    ncol = 4
    nrow = int(np.ceil(len(zoo) / ncol))
    fig, axs = plt.subplots(nrow, ncol, figsize=(15, 3.8 * nrow))
    for ax, name in zip(axs.flat, zoo):
        parity(ax, y, oof[name], name)
    for ax in list(axs.flat)[len(zoo):]:
        ax.axis("off")
    fig.suptitle("Out-of-fold predictions, 5-fold GroupKFold (alloys never shared between train and test)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "fig_parity.png")
    plt.close(fig)

    # ------------------------------------------------------------ data-leakage demonstration
    leak = []
    for name in ["Random forest", "Gradient boosting"]:
        r_rand = metrics(y, cross_validate(zoo[name], df, y, groups=None))
        r_grp = metrics(y, oof[name])
        leak.append({"model": name, "RMSE_random_KFold": r_rand["RMSE_log10"], "RMSE_GroupKFold": r_grp["RMSE_log10"],
                     "R2_random_KFold": r_rand["R2"], "R2_GroupKFold": r_grp["R2"]})
    pd.DataFrame(leak).to_csv(out / "leakage_demo.csv", index=False, float_format="%.4f")
    report["leakage_demo"] = leak

    # ------------------------------------------------------------ extrapolation in temperature
    tr, te = extrapolation_split(df, 950.0)
    ext_rows, ext_pred = [], {}
    for name in ["Larson-Miller (no composition)", "Random forest", "Gradient boosting", "Gaussian process",
                 "Hybrid: LMP + gradient boosting", "Gradient boosting on LMP target"]:
        m = clone(zoo[name]).fit(df.iloc[tr], y[tr])
        ext_pred[name] = m.predict(df.iloc[te])
        ext_rows.append({"model": name, **metrics(y[te], ext_pred[name])})
    ext = pd.DataFrame(ext_rows).sort_values("RMSE_log10")
    ext.to_csv(out / "metrics_extrapolation.csv", index=False, float_format="%.4f")
    report["extrapolation_T_gt_950C"] = ext.to_dict("records")
    fig, axs = plt.subplots(1, len(ext_pred), figsize=(4 * len(ext_pred), 4))
    for ax, (name, p) in zip(axs, ext_pred.items()):
        parity(ax, y[te], p, f"{name}\ntrained ≤ 950 °C → tested > 950 °C", color="C3")
    fig.tight_layout()
    fig.savefig(out / "fig_extrapolation.png")
    plt.close(fig)

    # ------------------------------------------------------------ uncertainty: GP std + conformal
    gp = clone(zoo["Gaussian process"]).fit(df.iloc[tr], y[tr])
    inner = gp[-1]
    mu, sd = inner[-1].predict(inner[0].transform(df.iloc[te][cols].to_numpy(float)), return_std=True)
    z = np.abs(y[te] - mu) / sd
    levels = np.linspace(0.05, 0.99, 20)
    from scipy.stats import norm
    observed = [np.mean(z <= norm.ppf(0.5 + p / 2)) for p in levels]
    # split-conformal intervals for the physics-informed GBM (LMP target):
    # (a) exchangeable case - new alloys drawn from the same distribution as the training alloys
    # (b) distribution shift - extrapolation to T > 950 C
    cmodel = zoo["Gradient boosting on LMP target"]
    rng = np.random.default_rng(0)
    test_alloys = rng.choice(np.unique(groups), size=len(np.unique(groups)) // 5, replace=False)
    te_a = np.isin(groups, test_alloys)
    _, lo_a, hi_a, q_a = conformal_interval(cmodel, df[~te_a], y[~te_a], df[te_a], 0.1, groups[~te_a])
    cover_a = float(np.mean((y[te_a] >= lo_a) & (y[te_a] <= hi_a)))
    pred_c, lo_c, hi_c, q = conformal_interval(cmodel, df.iloc[tr], y[tr], df.iloc[te], 0.1, groups[tr])
    cover = float(np.mean((y[te] >= lo_c) & (y[te] <= hi_c)))
    report["uncertainty"] = {"gp_coverage_90_extrapolation": float(np.mean(z <= norm.ppf(0.95))),
                             "conformal_model": "Gradient boosting on LMP target",
                             "conformal_90_halfwidth_log10": float(q_a),
                             "conformal_90_coverage_new_alloys": cover_a,
                             "conformal_90_coverage_extrapolation_T_gt_950C": cover}
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(levels, observed, "o-", label="Gaussian process")
    ax[0].plot([0, 1], [0, 1], "k--", lw=0.8, label="perfect calibration")
    ax[0].set(xlabel="nominal coverage", ylabel="observed coverage", title="GP calibration (extrapolation set)")
    ax[0].legend()
    order = np.argsort(y[te])
    ax[1].fill_between(np.arange(len(te)), lo_c[order], hi_c[order], color="C1", alpha=0.3, label="90% conformal interval")
    ax[1].plot(np.arange(len(te)), pred_c[order], "C1-", lw=1, label="GBM on LMP target")
    ax[1].plot(np.arange(len(te)), y[te][order], "k.", ms=3, label="measured")
    ax[1].set(xlabel="test points (sorted)", ylabel="log$_{10}$ t$_r$ (h)", title=f"90% conformal, T > 950 °C: coverage {cover:.2f}\n(new alloys, same T range: {cover_a:.2f})")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "fig_uncertainty.png")
    plt.close(fig)

    # ------------------------------------------------------------ interpretability (SHAP)
    gbm = clone(zoo["Gradient boosting"]).fit(df, y)
    try:
        import shap
        Xs = df[cols].sample(min(600, len(df)), random_state=0)
        expl = shap.TreeExplainer(gbm[-1])
        sv = expl.shap_values(Xs.to_numpy(float))
        imp = pd.Series(np.abs(sv).mean(0), index=cols).sort_values(ascending=False)
        plt.figure()
        shap.summary_plot(sv, Xs, show=False, max_display=15)
        plt.title("SHAP: what drives predicted rupture life", fontsize=10)
        plt.tight_layout()
        plt.savefig(out / "fig_shap.png", dpi=150)
        plt.close("all")
    except Exception as exc:  # pragma: no cover - fall back to permutation importance
        from sklearn.inspection import permutation_importance
        print("SHAP failed, using permutation importance:", exc)
        pi = permutation_importance(gbm, df, y, n_repeats=5, random_state=0)
        imp = pd.Series(pi.importances_mean, index=df.columns).loc[cols].sort_values(ascending=False)
    report["feature_importance_top10"] = imp.head(10).to_dict()

    # ------------------------------------------------------------ inverse design
    gp_all = clone(zoo["Gaussian process"]).fit(df, y)
    cand = ucb_search(gp_all, cols, df, args.design_T, args.design_stress, n_candidates=20000, kappa=0.5, top=10)
    near = df[(df["T_C"] == args.design_T)]
    if synthetic:
        cand["true_log10_life"] = np.log10(life_hours_true(cand, cand["T_C"], cand["stress_MPa"]))
        best_train = df.drop_duplicates(GROUP).copy()
        best_train_life = np.log10(life_hours_true(best_train, args.design_T, args.design_stress)).max()
        report["design"] = {"condition": f"{args.design_T:.0f} C / {args.design_stress:.0f} MPa",
                            "best_training_alloy_true_log10_life": float(best_train_life),
                            "designed_true_log10_life_top3": cand["true_log10_life"].head(3).tolist(),
                            "designed_pred_log10_life_top3": cand["pred_log10_life"].head(3).tolist()}
    cand.to_csv(out / "design_candidates.csv", index=False, float_format="%.4g")
    del near

    (out / "report.json").write_text(json.dumps(report, indent=2, default=float))
    print(json.dumps({k: report[k] for k in ("leakage_demo", "uncertainty") if k in report}, indent=1, default=float))
    if "design" in report:
        print("design:", report["design"])


if __name__ == "__main__":
    main()
