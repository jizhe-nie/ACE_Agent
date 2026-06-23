"""
W6① 生存验证 (Survival validation) — 用临床终点回答 W5 没答的"分几型"。

W5 结论：结构在所有 k 都显著(null 校准)，但 null 显著性选不出粒度。
W6①：用**总生存(OS)** 做外部裁判——对每个 k 的亚型做 **多组 log-rank 检验**，
**能把病人按预后分开(log-rank p 最小)的粒度，才是临床有意义的粒度**。并与 PAM50 自身的区分力对比。

实现：lifelines 未装，这里用 numpy + scipy.stats.chi2 自实现 KM 曲线 + 多组 log-rank（零依赖、可复现）。
说明：这是单变量 log-rank；正式论文还应做 Cox 调整年龄/分期等混杂(见日志 W6 待办)。

用法：
  python subtype/05_survival_validation.py \
      --expr data/brca/HiSeqV2.gz --pheno data/brca/BRCA_clinicalMatrix --subtype-col PAM50Call_RNAseq
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
import pandas as pd
from scipy.stats import chi2

from _data import exclude_normal_tissue, filter_labeled, load_real, preprocess
from _stability import ward_partition

OUT_DIR = Path(__file__).resolve().parent / "outputs"


def build_os(pheno: pd.DataFrame):
    """从临床表构造总生存：event=1 if DECEASED；time=死亡日 or 末次随访日(天→年)。"""
    vs = pheno["vital_status"].astype(str).str.upper()
    event = (vs == "DECEASED").astype(int).to_numpy()
    d2d = pd.to_numeric(pheno.get("days_to_death"), errors="coerce")
    d2f = pd.to_numeric(pheno.get("days_to_last_followup"), errors="coerce")
    time = np.where(event == 1, d2d, d2f).astype(float)
    valid = np.isfinite(time) & (time > 0) & np.isin(vs, ["LIVING", "DECEASED"])
    return time / 365.25, event, valid  # 年


def logrank(time, event, group):
    """多组 log-rank：返回 (chi2 统计量, 自由度, p)。"""
    groups = np.unique(group)
    G = len(groups)
    if G < 2:
        return 0.0, 0, 1.0
    et = np.unique(time[event == 1])
    O = np.zeros(G); E = np.zeros(G); V = np.zeros((G, G))
    for t in et:
        at_risk = time >= t
        n = int(at_risk.sum())
        d = int(((time == t) & (event == 1)).sum())
        if n < 2 or d == 0:
            continue
        ng = np.array([int((at_risk & (group == g)).sum()) for g in groups], float)
        dg = np.array([int(((time == t) & (event == 1) & (group == g)).sum()) for g in groups], float)
        O += dg
        E += d * ng / n
        c = d * (n - d) / (n - 1)
        for i in range(G):
            V[i, i] += c * (ng[i] / n) * (1 - ng[i] / n)
            for j in range(i + 1, G):
                V[i, j] -= c * (ng[i] / n) * (ng[j] / n)
                V[j, i] = V[i, j]
    OE = (O - E)[:-1]
    Vr = V[:-1, :-1]
    try:
        stat = float(OE @ np.linalg.solve(Vr, OE))
    except np.linalg.LinAlgError:
        stat = float(OE @ np.linalg.pinv(Vr) @ OE)
    return stat, G - 1, float(chi2.sf(stat, G - 1))


def km(time, event):
    """Kaplan-Meier 估计：返回阶梯 (t, S)。"""
    ts, ss, S = [0.0], [1.0], 1.0
    for ut in np.unique(time):
        at_risk = int((time >= ut).sum())
        d = int(((time == ut) & (event == 1)).sum())
        if at_risk > 0 and d > 0:
            S *= (1 - d / at_risk)
        ts.append(float(ut)); ss.append(S)
    return ts, ss


def main():
    ap = argparse.ArgumentParser(description="W6① 生存验证")
    ap.add_argument("--expr", required=True)
    ap.add_argument("--pheno", required=True)
    ap.add_argument("--subtype-col", default="PAM50Call_RNAseq")
    ap.add_argument("--kmin", type=int, default=2)
    ap.add_argument("--kmax", type=int, default=6)
    args = ap.parse_args()

    expr, pheno = load_real(args.expr, args.pheno)
    expr, pheno, _ = exclude_normal_tissue(expr, pheno)
    expr, pheno, _ = filter_labeled(expr, pheno, args.subtype_col)
    time, event, valid = build_os(pheno)
    expr, pheno = expr.loc[pheno.index[valid]], pheno.loc[pheno.index[valid]]
    time, event = time[valid], event[valid]
    _, Xp, _ = preprocess(expr)
    print(f"[队列] {Xp.shape[0]} 例有生存数据 | 事件(死亡) {int(event.sum())} | 中位随访 {np.median(time):.1f} 年\n")

    # 各 k 的亚型生存区分力
    print(f"{'k':<5}{'log-rank χ²':<14}{'df':<5}{'p':<12}{'-log10(p)':<12}")
    print("-" * 48)
    results = {}
    for k in range(args.kmin, args.kmax + 1):
        lab = ward_partition(Xp, k)
        stat, df, p = logrank(time, event, lab)
        results[k] = p
        print(f"{k:<5}{stat:<14.1f}{df:<5}{p:<12.2e}{-np.log10(max(p,1e-300)):<12.2f}")

    # PAM50 自身作对照
    pam = pd.Categorical(pheno[args.subtype_col]).codes
    s_p, df_p, p_p = logrank(time, event, pam)
    print(f"{'PAM50':<5}{s_p:<14.1f}{df_p:<5}{p_p:<12.2e}{-np.log10(max(p_p,1e-300)):<12.2f}  (临床金标准对照)")

    kbest = min(results, key=results.get)
    print(f"\n[临床粒度] log-rank p 最小 → k = {kbest} (p={results[kbest]:.2e})")
    print("  解读：这是用'病人预后'选出的亚型数——能把生存曲线分得最开的粒度。")
    print("  与 W5 的'稳定性偏好 k=2'对照：稳定 ≠ 临床有用；外部终点才定夺粒度。")

    _plot_km(Xp, time, event, kbest, pheno, args.subtype_col)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"k": list(results), "logrank_p": list(results.values())}).to_csv(
        OUT_DIR / "survival_logrank.csv", index=False)
    print(f"\n[输出] log-rank 表 + KM 图已存到 {OUT_DIR}")


def _plot_km(Xp, time, event, kbest, pheno, subtype_col):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    # 左：我们 k*=kbest 的亚型
    lab = ward_partition(Xp, kbest)
    for c in np.unique(lab):
        m = lab == c
        ts, ss = km(time[m], event[m])
        axes[0].step(ts, ss, where="post", label=f"Subtype {c} (n={int(m.sum())})")
    _, _, p = logrank(time, event, lab)
    axes[0].set_title(f"Our subtypes (k={kbest}), log-rank p={p:.1e}")
    # 右：PAM50
    pam = pheno[subtype_col].to_numpy()
    for c in pd.unique(pam):
        m = pam == c
        ts, ss = km(time[m], event[m])
        axes[1].step(ts, ss, where="post", label=f"{c} (n={int(m.sum())})")
    pcodes = pd.Categorical(pam).codes
    _, _, pp = logrank(time, event, pcodes)
    axes[1].set_title(f"PAM50 (clinical), log-rank p={pp:.1e}")
    for ax in axes:
        ax.set_xlabel("years"); ax.set_ylabel("overall survival"); ax.set_ylim(0, 1.02); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(OUT_DIR / "survival_km.png", dpi=120); plt.close(fig)


if __name__ == "__main__":
    main()
