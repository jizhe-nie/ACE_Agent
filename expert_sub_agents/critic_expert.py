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
        self._audit_target = {
            "algorithm_name": winner_result.algorithm_name,
            "expert_label": winner_result.expert_label,
            "metrics": winner_result.metrics,
            "n_labels": len(set(winner_labels)) if winner_labels is not None and hasattr(winner_labels, '__iter__') else 0,
        }
        try:
            results = self.execute_with_self_correction(dataset, "", settings)
            if results:
                for r in results:
                    audit = r.metrics.get("audit_report") if isinstance(r.metrics, dict) else None
                    if audit and isinstance(audit, dict):
                        return audit
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
            "## 上下文变量（沙箱已预注入，直接使用，无需 import）\n"
            "- CTX_DATA: .X, .y, .n_samples, .n_features, .expected_clusters, .has_labels\n"
            "- StandardScaler, KMeans, silhouette_score, davies_bouldin_score,\n"
            "  calinski_harabasz_score, adjusted_rand_score, numpy as np\n\n"
            "## 获胜者上下文（代码顶部已预先定义 WINNER 变量，直接使用）\n"
            f"# WINNER = {winner_json}\n\n"
            "## 审计任务（全部必须完成）\n\n"
            "### 1. Hopkins Statistic（聚类趋势检验）\n"
            "- 实现标准 Hopkins: H > 0.7 强趋势, H ≈ 0.5 随机, H < 0.3 无趋势\n\n"
            "### 2. CVI 多k扫描\n"
            "- k=2..min(15, sqrt(n)) 扫描 Silhouette / DBI / CHI\n"
            "- 多指标投票得 k_consensus\n"
            "- 从 WINNER metrics 推断获胜者实际使用的 k，检查与 k_consensus 是否一致\n\n"
            "### 3. Bootstrap 稳定性（针对获胜算法 + k）\n"
            "- 用与获胜者相同的算法+k，做 10 次 80% 子采样\n"
            "- 每次聚类后与全量结果对比 ARI\n"
            "- stability_score = 10 次 ARI 中位数\n\n"
            "### 4. 过拟合风险评估\n"
            "- Silhouette > 0.8 但 Hopkins < 0.5 → overfitting_risk='high'\n"
            "- stability_score < 0.6 → overfitting_risk ≥ 'medium'\n"
            "- 多指标最优k分歧大 → 数据信号弱\n\n"
            "### 5. 审计裁决 (endorsement) + 行动指令 (action, Critic 2.0)\n"
            "- 'endorsed' → action='CLEAR': stability_score ≥ 0.75 + Hopkins ≥ 0.6 + k 一致\n"
            "- 'qualified' → action='WARN': stability_score ≥ 0.5，存在轻微不一致\n"
            "- 'qualified_with_warning' → action='RETRY': 存在明显问题（不稳定 / 过拟合 / k 矛盾）\n"
            "- 当 action='RETRY' 时，必须填写 retry_constraints：\n"
            "  - force_k: 根据 CVI 多指标投票得出的 k_consensus（若与 winner k 不一致）\n"
            "  - blocked_algorithms: 列出表现差的算法（如 stability<0.3 的算法）\n"
            "  - force_preprocessing: 若 Hopkins<0.5 建议 'standardize' 或 'pca'\n\n"
            "## 输出格式（严格遵守）\n"
            "```python\n"
            "artifacts['Critic_Audit'] = {\n"
            "    'labels': [],\n"
            "    'metrics': {\n"
            "        'score': 0.0,  # 审计不参与排名，固定为 0\n"
            "        'score_source': 'audit',\n"
            "    },\n"
            "    'audit_report': {\n"
            "        'confidence_level': float,       # 0-1 综合置信度\n"
            "        'overfitting_risk': str,         # 'low' / 'medium' / 'high'\n"
            "        'stability_score': float,        # bootstrap 稳定性 0-1\n"
            "        'hopkins': float,                # Hopkins 统计量\n"
            "        'winner_k_consistency': bool,    # 获胜者 k 与 CVI 共识一致?\n"
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
        """Override to re-inject WINNER after every self-healing retry."""
        repaired = super()._fix_code(client, old_code, error, attempt=attempt)
        winner = self._audit_target or {}
        winner_json = json.dumps(winner, ensure_ascii=False, indent=2)
        return f"WINNER = {winner_json}\n" + repaired
