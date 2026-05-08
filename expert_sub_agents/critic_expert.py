"""
expert_sub_agents/critic_expert.py
==================================
Critic Expert: Independent post-hoc auditor for clustering quality.

Phase 1 redesign (2026-04-29):
- Repositioned from parallel voter to post-hoc independent auditor.
- Runs AFTER the winner is selected; audits its trustworthiness.
- Outputs structured ``audit_report`` — does NOT produce a competitive score.
- Examines: Hopkins, bootstrap stability, CVI consistency, overfitting risk.
"""

from __future__ import annotations

import json
from typing import Any

from ACE_Agent.agent_core.schemas import AlgorithmRunResult, DatasetBundle
from ACE_Agent.expert_sub_agents.base import BaseExpert, _strip_code_fences
from ACE_Agent.tools.llm_client import LLMSettings, UniversalLLMClient


class CriticExpert(BaseExpert):
    """Post-hoc clustering quality auditor.

    Runs independently AFTER the winner is selected.  Examines the
    dataset's cluster tendency (Hopkins), the winner's stability
    (bootstrap resampling), and CVI consensus to produce a structured
    ``audit_report`` that endorses, qualifies, or warns about the result.
    """

    def __init__(self) -> None:
        super().__init__("critic", "审计专家")
        self._audit_target: dict[str, Any] | None = None

    # ------------------------------------------------------------------
    # Public audit entry point
    # ------------------------------------------------------------------

    def execute_audit(
        self,
        winner_result: AlgorithmRunResult,
        dataset: DatasetBundle,
        settings: LLMSettings,
    ) -> dict[str, Any] | None:
        """Run a post-hoc audit against the selected winner.

        Returns an ``audit_report`` dict on success, or ``None`` if the
        audit could not be completed.
        """
        winner_labels = getattr(winner_result, "labels", None)
        n_labels = len(set(winner_labels)) if winner_labels is not None and hasattr(winner_labels, '__iter__') else 0

        # ---- Sampling strategy for large datasets -------------------------
        n_total = dataset.X.shape[0]
        audit_n = n_total if n_total <= 2000 else min(2000, max(1000, int(n_total * 0.15)))
        audit_sampled = audit_n < n_total

        # ---- Pre-computed metrics from winner (avoid recomputation) --------
        precomputed = {}
        wm = winner_result.metrics if isinstance(winner_result.metrics, dict) else {}
        for k in ("silhouette", "calinski_harabasz", "davies_bouldin", "ari", "nmi", "dbcv_score"):
            if k in wm:
                precomputed[k] = wm[k]

        self._audit_target = {
            "algorithm_name": winner_result.algorithm_name,
            "expert_label": winner_result.expert_label,
            "metrics": wm,
            "n_labels": n_labels,
            # Phase 5.3: sampling strategy
            "audit_sample_size": audit_n,
            "audit_sampled": audit_sampled,
            "n_total": n_total,
            # Phase 5.3: pre-computed metrics to reuse
            "precomputed_metrics": precomputed,
            # Phase 5.3: fast audit mode
            "fast_audit": bool(getattr(settings, "fast_audit", False)),
        }
        try:
            results = self.execute_with_self_correction(dataset, "", settings)
            if results:
                for r in results:
                    m = r.metrics if isinstance(r.metrics, dict) else {}
                    # Check metrics.audit_report (correct nesting per prompt template)
                    audit = m.get("audit_report")
                    if isinstance(audit, dict):
                        return audit
                    # Fallback: check if metrics IS the audit_report (LLM mis-nested)
                    if "endorsement" in m and "confidence_level" in m:
                        return m
                # Results exist but no audit_report found → last_logs has the error
                self.last_logs.append(
                    f"[{self.label}] 审计代码执行成功但未产出 audit_report 字段，"
                    f"实际返回的 metrics keys: {list(results[0].metrics.keys()) if results else '?'}"
                )
            else:
                self.last_logs.append(
                    f"[{self.label}] 审计执行未产���任何有效结果 "
                    f"(全部 {self.MAX_RETRIES} 次重试均失败)"
                )
            return None
        finally:
            self._audit_target = None

    # ------------------------------------------------------------------
    # Code generation
    # ------------------------------------------------------------------

    def _generate_code(
        self,
        client: UniversalLLMClient,
        dataset: DatasetBundle,
        prompt: str,
        constraints=None,
    ) -> str:
        winner = self._audit_target or {}
        winner_json = json.dumps(winner, ensure_ascii=False, indent=2)

        system_prompt = (
            "你是一个聚类审计专家。你的任务是审查一个已获胜的聚类结果的可信度。\n"
            "生成沙箱中执行的 Python 审计代码，严格遵守以下规则。\n\n"
            "## 预注入变量（沙箱已注入，直接用变量名，严禁 import 这些）\n"
            "- CTX_DATA: .X, .y, .n_samples, .n_features, .expected_clusters, .has_labels\n"
            "- numpy as np, StandardScaler, KMeans, GaussianMixture, PCA\n"
            "- silhouette_score, calinski_harabasz_score, davies_bouldin_score,\n"
            "  adjusted_rand_score, normalized_mutual_info_score\n"
            "- DBSCAN, OPTICS, SpectralClustering, AgglomerativeClustering, HDBSCAN\n"
            "- NearestNeighbors, kneighbors_graph, radius_neighbors_graph\n"
            "- StratifiedShuffleSplit, DecisionTreeClassifier\n"
            "- csgraph (scipy.sparse.csgraph), sparse (scipy.sparse)\n"
            "- dbcv_score(X, labels) -> float  # 密度聚类验证指标\n\n"
            "## 可 import 的模块（不在预注入列表中的模块使用 import）\n"
            "所有标准 sklearn/scipy 模块均可通过标准 import 导入。\n"
            "matplotlib 使用 'import matplotlib; matplotlib.use(\"Agg\"); import matplotlib.pyplot as plt'。\n\n"
            "## 获胜者上下文（代码顶部已预先定义 WINNER 变量，直接使用）\n"
            f"# WINNER = {winner_json}\n\n"
            "WINNER 字段说明:\n"
            "- audit_sample_size: 本次审计应使用的最大样本数（若 n_total > 2000）\n"
            "- audit_sampled: True 表示需要采样\n"
            "- precomputed_metrics: 专家已计算的指标，直接复用，禁止重复计算\n"
            "- fast_audit: True 表示快速审计模式（跳过 Bootstrap 和 Graph 任务）\n\n"
            "## 0. 审计前采样策略（Phase 5.3，必读）\n"
            "**规则：禁止在超过 2000 样本上执行全量审计！**\n"
            "- 若 WINNER['audit_sampled'] == True:\n"
            "  * 从 CTX_DATA.X 中使用分层采样提取 n=audit_sample_size 的子集。\n"
            "  * 若有 y: 用 StratifiedShuffleSplit 确保每类比例一致。\n"
            "  * 若 y 为 None: 用 KMeans 粗聚类(5簇)做伪分层采样。\n"
            "  * 同时子采样 WINNER labels（用相同索引）得到子集 labels。\n"
            "  * 所有后续审计任务（Hopkins/Bootstrap/CVI/DBCV/Graph）均在子集上运行。\n"
            "  * 变量命名: _X_audit, _y_audit, _labels_audit（替代原始 X/y/labels）。\n"
            "- 若 audit_sampled == False: 直接使用原始数据，无需采样。\n\n"
            "## 审计任务（全部必须完成，但 fast_audit 模式下跳过标注 [FAST_SKIP] 的任务）\n\n"
            "### 1. Hopkins Statistic（聚类趋势检验）\n"
            "- 用 _X_audit 的子集（最多 500 点）计算 Hopkins。\n"
            "- 实现标准 Hopkins: H > 0.7 强趋势, H ≈ 0.5 随机, H < 0.3 无趋势\n\n"
            "### 2. CVI 多k扫描\n"
            "- 用 _X_audit 计算，k=2..min(15, sqrt(n_audit))\n"
            "- 若 WINNER['precomputed_metrics'] 中有 silhouette/ch/dbi，可作为单点参考但扫描仍需进行。\n"
            "- 多指标投票得 k_consensus\n"
            "- 从 WINNER metrics 推断获胜者实际使用的 k，检查与 k_consensus 是否一致\n\n"
            "### 3. DBCV 指标计算 [FAST_SKIP]\n"
            "- 若 WINNER['precomputed_metrics'] 中已有 dbcv_score，直接复用，跳过计算。\n"
            "- 否则调用 dbcv_score(_X_audit, _labels_audit)（沙箱已预注入）。\n"
            "- dbcv_score 函数签名为 dbcv_score(X, labels) -> float，返回值范围 [-1, 1]。\n"
            "- 当 Hopkins > 0.6 或 winner 算法为基于密度的算法(HDBSCAN/DBSCAN/OPTICS)时:\n"
            "  DBCV 必须作为首要评分指标，Silhouette 仅供参考\n"
            "- 当 Hopkins ≤ 0.5 且数据接近高斯时: DBCV 作为辅助指标\n"
            "- 如果 DBCV < 0 (簇间分离度弱于簇内离散度)，视为严重警告信号\n\n"
            "### 4. Bootstrap 稳定性 — 动态早停版（Phase 5.3）[FAST_SKIP]\n"
            "- 用与获胜者相同的算法+k，对 _labels_audit 做子采样验证。\n"
            "- **自适应轮数**（非固定值）：\n"
            "  * Phase 1 (5轮): 执行 5 次 80% 子采样，计算 ARI 序列。\n"
            "  * 若 std(ARI[0:5]) < 0.02: 提前终止，stability_score = median(ARI[0:5])。\n"
            "  * Phase 2 (最多 10 轮追加): 逐轮执行，每轮后更新 std。\n"
            "    当 std(ARI) < 0.03 或 total_rounds >= 15 时终止。\n"
            "  * stability_score = median(all ARIs)。\n"
            "- 注意: 使用 _X_audit（已采样子集），每次 80% 子采样从 _X_audit 中取。\n"
            "- 最大总轮数 15，最小总轮数 5。\n\n"
            "### 5. 过拟合风险评估\n"
            "- Silhouette > 0.8 但 Hopkins < 0.5 → overfitting_risk='high'\n"
            "- DBCV < 0 且 Silhouette > 0.5 → 疑似'球形偏差'过拟合（Silhouette 被欧氏距离愚弄）\n"
            "- stability_score < 0.6 → overfitting_risk ≥ 'medium'\n"
            "- 多指标最优k分歧大 → 数据信号弱\n\n"
            "### 6. 审计裁决 (endorsement) + 行动指令 (action, Critic 2.0)\n"
            "- 'endorsed' → action='CLEAR': stability_score ≥ 0.75 + Hopkins ≥ 0.6 + k 一致\n"
            "- 'qualified' → action='WARN': stability_score ≥ 0.5，存在轻微不一致\n"
            "- 'qualified_with_warning' → action='RETRY': 存在明显问题（不稳定 / 过拟合 / k 矛盾）\n"
            "- 特别规则: DBCV < 0 → endorsement 不得为 'endorsed'（密度分离不足，结果不可信）\n"
            "- 特别规则: boundary_quality_score < 0.4 且 geodesic_distortion > 0.3 → endorsement 不得为 'endorsed'\n"
            "- 特别规则: topology_split_ratio > 0.5 → endorsement 不得为 'endorsed', action='RETRY'\n"
            "- 特别规则: 存在簇被切成 >= 3 个连通分量 → endorsement 降级为 'qualified_with_warning'\n"
            "- 当 action='RETRY' 时，必须填写 retry_constraints：\n"
            "  - force_k: 根据 CVI 多指标投票得出的 k_consensus（若与 winner k 不一致）\n"
            "  - blocked_algorithms: 列出表现差的算法（如 stability<0.3 / DBCV<0 的算法）\n"
            "  - force_preprocessing: 若 Hopkins>0.6 且 DBCV<0 建议 'umap' 流形嵌入\n\n"
            "### 7. Graph-Based 指标（Phase 3, Topology-Aware）[FAST_SKIP]\n"
            "- 若 fast_audit 模式: 跳过，所有 graph 字段填 0 或默认值。\n"
            "- 在 _X_audit（采样子集）上构建 kNN 图（非全量数据！）:\n"
            "  adjacency = kneighbors_graph(_X_audit, min(15, n_audit-1), mode='distance')\n"
            "- 用 csgraph.shortest_path 计算 geodesic distances\n"
            "- 计算 geodesic_distortion: median(|d_geo - d_euc| / max(d_euc, 1e-8))\n"
            "- 检测 wall-crossing 点对: 欧氏距离 < 0.1 但 geodesic > 5 跳\n"
            "  若 wall_crossing_ratio > 0.1 → endorsement 不得为 'endorsed'\n"
            "- 计算 graph_modularity / graph_conductance / neighborhood_preservation\n"
            "- 当 graph_conductance > 0.7: action = 'RETRY', force_preprocessing = 'umap'\n\n"
            "### 8. Boundary Quality Audit（Phase 3.2，图连通数据）[FAST_SKIP]\n"
            "- 若 fast_audit 模式: 跳过，boundary 字段填默认值。\n"
            "- 在 _X_audit 的 kNN 图上计算 boundary_quality_score (0-1):\n"
            "  * inter_community_edge_flow / bottleneck_alignment / axis_aligned_penalty\n"
            "  * threshold_explainable: DecisionTreeClassifier(max_depth=1) 检查\n"
            "- 当 boundary_quality_score < 0.4 且 geodesic_distortion > 0.3:\n"
            "  action = 'RETRY', endorsement 不得为 'endorsed'\n\n"
            "### 9. Bootstrapping 稳定性门禁（Phase 5.1，低 ARI 增强审计）[FAST_SKIP]\n"
            "- 复用任务 4 的动态早停结果。任务 4 已产出 stability_score 序列。\n"
            "- 计算 stability_iqr = Q3(ARI) - Q1(ARI)，衡量稳定性分散度。\n"
            "- **诚实失败门禁**（复用任务 4 的 stability_score）：\n"
            "  * 若 stability_score < 0.4: honest_failure = True。\n"
            "  * 若 stability_iqr > 0.3: honest_failure = True（高度不稳定）。\n"
            "  * 若 honest_failure: recommendation 应为诚实失败建议：\n"
            "    '该数据结构复杂（极窄薄流形/并行曲线），现有模型召回率低。'\n"
            "- 新增审计字段：honest_failure, stability_iqr, bootstrap_aris\n\n"
            "### 10. 拓扑连通性审计（Topological Connectivity）\n"
            "- 在 _X_audit 上构建 k-NN 图（k = min(10, n_audit-1), mode='connectivity'）。\n"
            "- 对每个簇（labels in _labels_audit, 排除噪声 label=-1）：\n"
            "  * 提取该簇所有点的 induced subgraph。\n"
            "  * 用 csgraph.connected_components 检查连通分量数。\n"
            "  * 若某簇有 > 1 个连通分量 → 该簇在拓扑上被「切断」, 记录为 split_cluster。\n"
            "- 计算 topology_split_ratio = split_clusters / total_clusters。\n"
            "- **裁决规则**：\n"
            "  * topology_split_ratio > 0.3 且 endorsement='endorsed' → 降级为 'qualified'\n"
            "  * topology_split_ratio > 0.5 → endorsement 不得为 'endorsed', action='RETRY'\n"
            "  * 任意簇被切成 >= 3 个连通分量 → 视为严重拓扑失败，追加 Strong Warning\n"
            "- 新增审计字段：topology_connectivity_pass (bool), topology_split_ratio (float),\n"
            "  topology_split_clusters (list[int] 被切断的簇标签)\n\n"
            "## 输出格式（严格遵守）\n"
            "```python\n"
            "artifacts['Critic_Audit'] = {\n"
            "    'labels': [],\n"
            "    'metrics': {\n"
            "        'score': 0.0,  # 审计不参与排名，固定为 0\n"
            "        'score_source': 'audit',\n"
            "        'audit_report': {\n"
            "            'confidence_level': float,       # 0-1 综合置信度\n"
            "        'overfitting_risk': str,         # 'low' / 'medium' / 'high'\n"
            "        'stability_score': float,        # bootstrap 稳定性 0-1\n"
            "        'hopkins': float,                # Hopkins 统计量\n"
            "        'dbcv_score': float,             # Phase 2.4 DBCV 指标 [-1, 1]\n"
            "        'winner_k_consistency': bool,    # 获胜者 k 与 CVI 共识一致?\n"
            "        'geodesic_distortion': float,    # Phase 3 欧氏-图测地距离失真率\n"
            "        'graph_modularity': float,       # Phase 3 簇内边 / 总边 (0-1)\n"
            "        'graph_conductance': float,      # Phase 3 跨簇边 / 簇内总边 (0-1)\n"
            "        'wall_crossing_ratio': float,    # Phase 3 欧氏近但图远的点对比例\n"
            "        'neighborhood_preservation': float, # Phase 3 kNN 边同簇保留比例\n"
            "        'boundary_quality_score': float, # Phase 3.2 边界质量 0-1\n"
            "        'boundary_axis_aligned': bool,    # Phase 3.2 边界是否轴对齐\n"
            "        'boundary_assessment': str,       # Phase 3.2 'good'/'fair'/'poor'\n"
            "        'honest_failure': bool,          # Phase 5.1: 诚实失败标志\n"
            "        'stability_iqr': float,          # Phase 5.1: 15 次 bootstrap ARI 的 IQR\n"
            "        'bootstrap_aris': [float, ...],  # Phase 5.1: 15 次子采样 ARI 全量\n"
            "        'topology_connectivity_pass': bool,  # Phase 5.3: 所有簇在kNN图中连续?\n"
            "        'topology_split_ratio': float,    # Phase 5.3: 被切断簇的比例\n"
            "        'topology_split_clusters': [int, ...],  # Phase 5.3: 被切断的簇标签\n"
            "        'endorsement': str,              # 'endorsed' / 'qualified' / 'qualified_with_warning'\n"
            "        'action': str,                   # Critic 2.0: 'CLEAR' / 'WARN' / 'RETRY'\n"
            "        'retry_constraints': {           # 仅 action==RETRY 时有效\n"
            "            'force_k': int | None,\n"
            "            'blocked_algorithms': [str, ...],\n"
            "            'force_preprocessing': str | None,  # 'standardize'/'normalize'/'pca'\n"
            "        },\n"
            "        'findings': [str, ...],          # 3-5 条关键审计发现\n"
            "        'recommendation': str,           # 给用户的建议\n"
            "    },\n"
            "    },\n"
            "}\n"
            "```\n\n"
            "只返回 Python 代码，不要解释或 Markdown 包裹。"
            "代码顶部不需要再定义 WINNER —— 沙箱已预先注入该变量。"
        )

        user_msg = (
            f"审计数据集 '{dataset.display_name}'"
            f"（{dataset.X.shape[0]} 样本, {dataset.X.shape[1]} 特征"
        )
        if dataset.y is not None:
            user_msg += f", {len(set(dataset.y.ravel()))} 类标签可用"
        user_msg += (
            f"）。\n获胜算法: {winner.get('algorithm_name', 'unknown')}"
            f"（{winner.get('expert_label', 'unknown')}），"
            f"得分: {winner.get('metrics', {}).get('score', 'N/A')}。"
            "\n请对此获胜结果进行独立审计。"
        )
        raw = client.chat_completion(
            [{"role": "user", "content": user_msg}],
            system_prompt,
        ).strip()

        # Strip code fences from LLM output BEFORE injecting WINNER, so the
        # fence markers don't end up in the middle of the final code.
        clean = _strip_code_fences(raw)
        winner_line = f"WINNER = {winner_json}\n"
        return winner_line + clean

    def _fix_code(
        self,
        client: UniversalLLMClient,
        old_code: str,
        error: str,
        *,
        attempt: int = 1,
    ) -> str:
        """Override to re-inject WINNER after every self-healing retry.

        Strips any existing WINNER line from old_code before sending to the LLM,
        so the LLM sees clean code without a duplicate variable definition.
        """
        # Strip previously prepended WINNER line from old_code so the LLM
        # doesn't see a duplicate variable and inadvertently carry it forward.
        import re as _re

        _clean_old = _re.sub(r"^WINNER\s*=\s*\{[^}]*\}\s*\n", "", old_code, count=1)
        repaired = super()._fix_code(client, _clean_old, error, attempt=attempt)
        winner = self._audit_target or {}
        winner_json = json.dumps(winner, ensure_ascii=False, indent=2)
        return f"WINNER = {winner_json}\n" + _strip_code_fences(repaired)
