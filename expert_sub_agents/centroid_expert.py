from __future__ import annotations
from ACE_Agent.expert_sub_agents.base import BaseExpert
from ACE_Agent.agent_core.schemas import DatasetBundle
from ACE_Agent.tools.llm_client import UniversalLLMClient

class CentroidExpert(BaseExpert):
    """质心专家：负责 KMeans 等基于距离的算法"""
    
    def __init__(self):
        super().__init__(key="centroid", label="质心专家")

    def _generate_code(self, client: UniversalLLMClient, dataset: DatasetBundle, prompt: str) -> str:
        system_prompt = (
            "你是一个 Python 聚类专家。生成代码运行 KMeans 算法。\n"
            "输入变量: X (numpy.ndarray), y (可选 labels)。\n"
            "必须使用以下格式返回 artifacts：\n"
            "artifacts['KMeans'] = {'labels': ..., 'metrics': {'score': 轮廓系数, ...}, 'plot_path': 'kmeans.png'}\n"
            "只返回 Python 代码，不要有任何解释。"
        )
        user_input = f"数据集画像：{dataset.description}，样本量：{dataset.X.shape[0]}"
        return client.chat_completion([{"role": "user", "content": user_input}], system_prompt).strip("```python").strip("```")
