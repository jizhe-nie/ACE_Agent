"""
subtype/_survival.py — 生存分析共享 helpers（W6/W7 复用）。
KM 曲线 + 多组 log-rank（numpy+scipy 自实现，无 lifelines 依赖）。
（注：05_ 有早期内联副本，逻辑一致，后续小重构统一。）
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import chi2


def build_os(pheno: pd.DataFrame):
    """从 TCGA 临床表构造总生存：event=1 if DECEASED；time(年)=死亡日 or 末次随访日。
    返回 (time_years, event, valid_mask)。"""
    vs = pheno["vital_status"].astype(str).str.upper()
    event = (vs == "DECEASED").astype(int).to_numpy()
    d2d = pd.to_numeric(pheno.get("days_to_death"), errors="coerce")
    d2f = pd.to_numeric(pheno.get("days_to_last_followup"), errors="coerce")
    time = np.where(event == 1, d2d, d2f).astype(float)
    valid = np.isfinite(time) & (time > 0) & np.isin(vs, ["LIVING", "DECEASED"])
    return time / 365.25, event, valid


def logrank(time, event, group):
    """多组 log-rank：返回 (chi2, df, p)。删失安全。"""
    time = np.asarray(time, float); event = np.asarray(event, int)
    group = np.asarray(group)
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
    OE = (O - E)[:-1]; Vr = V[:-1, :-1]
    try:
        stat = float(OE @ np.linalg.solve(Vr, OE))
    except np.linalg.LinAlgError:
        stat = float(OE @ np.linalg.pinv(Vr) @ OE)
    return stat, G - 1, float(chi2.sf(stat, G - 1))


def km(time, event):
    """Kaplan-Meier 阶梯估计：返回 (t_list, S_list)。"""
    time = np.asarray(time, float); event = np.asarray(event, int)
    ts, ss, S = [0.0], [1.0], 1.0
    for ut in np.unique(time):
        at_risk = int((time >= ut).sum())
        d = int(((time == ut) & (event == 1)).sum())
        if at_risk > 0 and d > 0:
            S *= (1 - d / at_risk)
        ts.append(float(ut)); ss.append(S)
    return ts, ss
