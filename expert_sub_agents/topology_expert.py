from __future__ import annotations
from ACE_Agent.expert_sub_agents.base import BaseExpert
from ACE_Agent.agent_core.schemas import DatasetBundle
from ACE_Agent.tools.llm_client import UniversalLLMClient

class TopologyExpert(BaseExpert):
    """拓扑专家：负责 DBSCAN 等基于密度的算法"""
    
    def __init__(self):
        super().__init__(key="topology", label="拓扑专家")

    def _generate_code(self, client: UniversalLLMClient, dataset: DatasetBundle, prompt: str) -> str:
        system_prompt = (
            "你是一个 Python 聚类专家（拓扑与密度算法分支）。\n"
            "你的职责范围：DBSCAN, SpectralClustering (谱聚类), AgglomerativeClustering (层次聚类), OPTICS。\n"
            "任务要求：\n"
            f"1. 分析用户指令：'{prompt}'\n"
            "2. 如果用户指定了你职责范围内的算法（特别是'谱聚类'），必须实现它。如果是谱聚类，请确保设置合适的 n_clusters。\n"
            "3. 如果未指定或指定了其他领域算法，默认实现 DBSCAN。\n"
            "4. 输入变量：直接使用内存中的 X (numpy.ndarray)。\n"
            "5. 输出要求：必须在 artifacts 字典中存储结果。格式：\n"
            "artifacts['SpectralClustering'] = {'labels': ..., 'metrics': {'score': 轮廓系数}, 'plot_path': 'spectral.png'} (以此类推)\n"
            "注意：只返回 Python 代码，不要有任何解释。"
        )
        user_input = f"数据集画像：{dataset.description}，样本量：{dataset.X.shape[0]}。"
        return client.chat_completion([{"role": "user", "content": user_input}], system_prompt).strip("```python").strip("```")
