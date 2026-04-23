from __future__ import annotations
from ACE_Agent.expert_sub_agents.base import BaseExpert, _strip_code_fences
from ACE_Agent.agent_core.schemas import DatasetBundle
from ACE_Agent.tools.llm_client import UniversalLLMClient

class TopologyExpert(BaseExpert):
    """拓扑专家：负责 DBSCAN 等基于密度的算法"""
    
    def __init__(self):
        super().__init__(key="topology", label="拓扑专家")

    def _generate_code(
        self,
        client: UniversalLLMClient,
        dataset: DatasetBundle,
        prompt: str,
    ) -> str:
        system_prompt = (
            "你是一个高级 Python 数据科学专家（拓扑与密度算法分支）。\n"
            "## 核心指令：严谨的导入与自适应 (Strict Imports & Auto-Tuning)\n"
            "1. **必须显式包含所有导入**（严禁省略！）：\n"
            "   ```python\n"
            "   import warnings\n"
            "   import numpy as np\n"
            "   from sklearn.preprocessing import StandardScaler\n"
            "   from sklearn.neighbors import NearestNeighbors\n"
            "   from sklearn.cluster import DBSCAN, SpectralClustering, AgglomerativeClustering\n"
            "   try: from sklearn.cluster import HDBSCAN\n"
            "   except ImportError: from hdbscan import HDBSCAN\n"
            "   from sklearn.metrics import adjusted_rand_score, silhouette_score\n"
            "   warnings.filterwarnings('ignore') # 抑制警告\n"
            "   ```\n"
            "2. **预处理与调参**：\n"
            "   - 必须先用 `StandardScaler` 处理 X。\n"
            "   - DBSCAN 必须使用 `NearestNeighbors` 计算分位数来估算 eps，严禁盲目硬编码。\n"
            "   - 若结果为单簇或全噪声，必须自动尝试 HDBSCAN。\n"
            "3. **结构约束**：代码在顶层直接运行，结果写入 artifacts[algo_name]。\n"
            "只返回 Python 代码，不要有任何解释。"
        )
        user_input = f"为样本量 {dataset.X.shape[0]} 的数据集生成具备高稳健性的拓扑聚类代码。确保包含所有必要的 import 并实现算法回退逻辑。"
        raw = client.chat_completion(
            [{"role": "user", "content": user_input}],
            system_prompt,
        )
        return _strip_code_fences(raw or "")
