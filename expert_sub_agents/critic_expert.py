"""
expert_sub_agents/critic_expert.py
==================================
Critic Expert: The independent auditor for clustering quality.

Phase 1 milestone — upgraded from thin Hopkins-only wrapper to a comprehensive
audit expert that generates code for:
  - Hopkins statistic (cluster tendency)
  - Bootstrap resampling stability (KMeans ARI across resamples)
  - CVI suite: Davies-Bouldin, Calinski-Harabasz, Silhouette
  - Optimal-k scanning (k = 2 .. sqrt(n))
  - Overall audit conclusion with recommendations

The expert remains LLM-dependent (REQUIRES_LLM = True) because code generation
relies on the LLM's statistical reasoning to tailor the audit to the dataset.
"""
from __future__ import annotations

from ACE_Agent.agent_core.schemas import DatasetBundle
from ACE_Agent.expert_sub_agents.base import BaseExpert
from ACE_Agent.tools.llm_client import UniversalLLMClient


class CriticExpert(BaseExpert):
    """专家：聚类质量的独立审计方。

    在专家并行执行阶段运行，对数据集本身进行审计（不依赖其他专家的结果）。
    输出包含聚类趋势、稳定性、多指标CVI和最优k建议的综合报告。
    """

    def __init__(self) -> None:
        super().__init__("critic", "评价专家")

    def _generate_code(
        self,
        client: UniversalLLMClient,
        dataset: DatasetBundle,
        prompt: str,
    ) -> str:
        system_prompt = (
            "你是一个高级 Python 统计学与聚类审计专家。请生成完整可运行的审计代码。\n\n"
            "## 必须包含的导入（全部显式写出）\n"
            "```python\n"
            "import warnings, numpy as np\n"
            "from sklearn.preprocessing import StandardScaler\n"
            "from sklearn.neighbors import NearestNeighbors\n"
            "from sklearn.cluster import KMeans\n"
            "from sklearn.metrics import (silhouette_score, davies_bouldin_score,\n"
            "    calinski_harabasz_score, adjusted_rand_score)\n"
            "from sklearn.model_selection import ShuffleSplit\n"
            "warnings.filterwarnings('ignore')\n"
            "```\n\n"
            "## 审计任务（必须全部完成并写入 artifacts）\n\n"
            "### 1. Hopkins Statistic（聚类趋势检验）\n"
            "- 实现标准 Hopkins 统计量\n"
            "- H > 0.7 → 强聚类趋势; H ≈ 0.5 → 随机; H < 0.3 → 均匀分布\n\n"
            "### 2. CVI 指标多k扫描\n"
            "- 对 k = 2 到 min(15, sqrt(n_samples)) 扫描: Silhouette, DBI, CHI\n"
            "- 找出每个指标的最优 k\n"
            "- 记录所有 k 的指标值到列表\n\n"
            "### 3. Bootstrap 稳定性分析\n"
            "- 对数据做 10 次 80% 子采样\n"
            "- 每次对 k=2..8 运行 KMeans\n"
            "- 计算相邻 k 之间的 ARI 转移矩阵\n"
            "- 输出平均稳定性分数 (float, 0-1 越高越稳定)\n\n"
            "### 4. 综合审计结论\n"
            "- 根据 Hopkins、最优k 重合度、稳定性、是否有标签进行评级\n"
            "- 评级: 'excellent' / 'good' / 'fair' / 'poor'\n"
            "- 给出建议：首选算法类别（centroid / topology / hybrid）\n\n"
            "## 输出格式（必须严格遵守）\n"
            "```python\n"
            "artifacts['Critic_Audit'] = {\n"
            "    'labels': [],  # 审计不产生聚类标签，留空\n"
            "    'metrics': {\n"
            "        'score': float,           # 综合评分 0-1\n"
            "        'score_source': 'critic_audit',\n"
            "        'hopkins': float,        # Hopkins 统计量\n"
            "        'cluster_tendency': str, # 'strong'/'moderate'/'weak'\n"
            "        'stability_score': float, # bootstrap ARI 稳定性\n"
            "        'best_k_silhouette': int,\n"
            "        'best_k_dbi': int,\n"
            "        'best_k_chi': int,\n"
            "        'k_consensus': int,      # 多指标投票的最优 k\n"
            "        'audit_grade': str,      # excellent/good/fair/poor\n"
            "        'recommendation': str,   # centroid/topology/hybrid\n"
            "    },\n"
            "    'plot_path': '',  # 审计不生成图\n"
            "}\n"
            "```\n\n"
            "只返回 Python 代码，不要有任何解释或 Markdown 包裹。"
        )
        user_msg = (
            f"对数据集 '{dataset.display_name}'（{dataset.X.shape[0]} 样本, "
            f"{dataset.X.shape[1]} 特征"
        )
        if dataset.y is not None:
            user_msg += f", {len(set(dataset.y.ravel()))} 类标签可用"
        user_msg += "）进行完整聚类审计。"
        return client.chat_completion(
            [{"role": "user", "content": user_msg}],
            system_prompt,
        ).strip()
