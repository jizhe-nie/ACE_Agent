"""
expert_sub_agents/deep_representation.py
=======================================
Deep Representation Expert: Specialized in deep clustering methods
(AutoEncoders, DEC, IDEC) using PyTorch.

Integrated with the self-healing Think-Act-Fix loop (Phase 3 Prep).
"""

from __future__ import annotations

from ACE_Agent.agent_core.schemas import DatasetBundle
from ACE_Agent.expert_sub_agents.base import BaseExpert
from ACE_Agent.tools.llm_client import UniversalLLMClient


class DeepRepresentationExpert(BaseExpert):
    """专家：专注于利用神经网络（PyTorch）进行深度聚类。"""

    def __init__(self) -> None:
        super().__init__("deep", "深度表征专家")

    def _generate_code(
        self,
        client: UniversalLLMClient,
        dataset: DatasetBundle,
        prompt: str,
        constraints=None,
    ) -> str:
        constraint_prompt = self._inject_constraints_prompt(constraints)
        system_prompt = constraint_prompt + (
            "你是一个 Python 数据科学专家（深度聚类分支）。\n"
            "你的职责：利用神经网络（PyTorch）提取非线性特征并执行聚类。\n"
            "任务要求：\n"
            f"1. 分析用户意图：'{prompt}'\n"
            f"2. 数据画像：{dataset.description}，样本量：{dataset.X.shape[0]}，特征数：{dataset.X.shape[1]}。\n"
            "3. 核心逻辑：\n"
            "   - 设计一个简单的 AutoEncoder (AE) 或 DEC 结构的 PyTorch 模型。\n"
            "   - 实现训练循环（注意：epoch 数不宜过大，建议 20-50 轮以保证沙箱响应）。\n"
            "   - 在潜空间（Latent Space）执行 KMeans 聚类。\n"
            "   - 必须包含设备检测：优先使用 cuda，若不可用则回退到 cpu。\n"
            "4. 输入变量：直接使用内存中的 X (numpy.ndarray)。\n"
            "5. 输出要求：结果必须写入 artifacts[algo_name]，包含 labels, metrics, plot_path。\n"
            "只返回 Python 代码，不要有任何解释。"
        )
        user_input = "为该数据集生成 PyTorch 深度聚类管线。"
        return client.chat_completion(
            [{"role": "user", "content": user_input}],
            system_prompt,
        ).strip()
