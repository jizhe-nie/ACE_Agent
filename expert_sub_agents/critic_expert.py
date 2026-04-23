"""
expert_sub_agents/critic_expert.py
==================================
Critic Expert: The independent auditor for clustering quality.
Calculates cluster tendency (Hopkins) and internal/external validation scores
beyond just silhouette.

Phase 1 Milestone.
"""
from __future__ import annotations

from ACE_Agent.agent_core.schemas import DatasetBundle
from ACE_Agent.expert_sub_agents.base import BaseExpert
from ACE_Agent.tools.llm_client import UniversalLLMClient


class CriticExpert(BaseExpert):
    """专家：聚类质量的独立审计方。"""

    def __init__(self) -> None:
        super().__init__("critic", "评价专家")

    def _generate_code(
        self,
        client: UniversalLLMClient,
        dataset: DatasetBundle,
        prompt: str,
    ) -> str:
        system_prompt = (
            "你是一个高级 Python 统计学与聚类审计专家。\n"
            "## 核心指令：严谨的导入与稳健实现\n"
            "1. **必须显式包含导入**：\n"
            "   ```python\n"
            "   import warnings\n"
            "   import numpy as np\n"
            "   from sklearn.neighbors import NearestNeighbors\n"
            "   from sklearn.metrics import silhouette_score\n"
            "   warnings.filterwarnings('ignore')\n"
            "   ```\n"
            "2. **Hopkins Statistic 标准实现**：\n"
            "   请确保 `hopkins_statistic(X)` 逻辑中正确解包 `distances, indices = nbrs.kneighbors(...)`。\n"
            "## 任务要求：\n"
            "1. 生成代码计算 Hopkins Statistic。若 H > 0.7 视为有强聚类趋势。\n"
            "2. 建议的簇数量评估。\n"
            "3. 结果写入 `artifacts['Critic_Audit']`。\n"
            "只返回 Python 代码，不要有任何解释。"
        )
        user_input = f"对该数据集进行聚类趋势审计。确保包含所有必要的 import 语句。"
        return client.chat_completion(
            [{"role": "user", "content": user_input}],
            system_prompt,
        ).strip()
