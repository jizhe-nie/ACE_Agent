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
            "你是一个 Python 聚类专家（质心算法分支）。\n"
            "你的职责范围：KMeans, MiniBatchKMeans, GaussianMixture (GMM)。\n"
            "任务要求：\n"
            f"1. 分析用户指令：'{prompt}'\n"
            "2. 如果用户指定了你职责范围内的算法，必须优先实现它。\n"
            "3. 如果用户指定了其他领域的算法（如 DBSCAN, 谱聚类等），你仍应生成一个 KMeans 作为基准对比，但在 artifacts 的结果中保留 KMeans 键值。\n"
            "4. 输入变量：直接使用内存中的 X (numpy.ndarray)。\n"
            "5. 输出要求：必须在 artifacts 字典中存储结果。格式：\n"
            "artifacts['KMeans'] = {'labels': ..., 'metrics': {'score': 轮廓系数}, 'plot_path': 'kmeans.png'}\n"
            "注意：只返回 Python 代码，不要有任何解释。"
        )
        user_input = f"数据集画像：{dataset.description}，样本量：{dataset.X.shape[0]}。"
        return client.chat_completion([{"role": "user", "content": user_input}], system_prompt).strip("```python").strip("```")
