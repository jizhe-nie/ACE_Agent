from __future__ import annotations

from ACE_Agent.agent_core.schemas import DatasetBundle
from ACE_Agent.expert_sub_agents.base import BaseExpert, _strip_code_fences
from ACE_Agent.tools.llm_client import UniversalLLMClient


class CentroidExpert(BaseExpert):
    """质心专家：负责 KMeans 等基于距离的算法"""

    def __init__(self):
        super().__init__(key="centroid", label="质心专家")

    def _generate_code(self, client: UniversalLLMClient, dataset: DatasetBundle, prompt: str, constraints=None) -> str:
        constraint_prompt = self._inject_constraints_prompt(constraints)
        system_prompt = constraint_prompt + (
            "你是一个 Python 聚类专家（质心算法分支）。\n"
            "⚠️ 代码结构强约束（违反即视为失败）：\n"
            '- 代码必须在**顶层直接执行**，禁止用 `if __name__ == "__main__":` 包裹主逻辑。\n'
            "- 禁止把主逻辑写进 `def main():` 或 `def run():` 然后忘记调用。\n"
            "- `X` 和 `artifacts` 是沙箱已注入的全局变量，直接使用，不要当参数传递。\n"
            '- 完成后必须通过 `artifacts["算法名"] = {"labels": ..., "metrics": {"score": float}, "plot_path": "..."}` 写入结果。\n'
            "\n"
            "你的职责范围：KMeans, MiniBatchKMeans, GaussianMixture (GMM)。\n"
            "⚠️ 沙箱预注入说明：KMeans, GaussianMixture, StandardScaler, silhouette_score 等均已预注入，"
            "可直接使用无需 import。若必须显式 import：\n"
            "- KMeans, MiniBatchKMeans 来自 sklearn.cluster\n"
            "- GaussianMixture 来自 **sklearn.mixture**（非 sklearn.cluster！）\n"
            "任务要求：\n"
            f"1. 分析用户指令：'{prompt}'\n"
            "2. 如果用户指定了你职责范围内的算法，必须优先实现它。\n"
            "3. 如果用户指定了其他领域的算法（如 DBSCAN, 谱聚类等），你仍应生成一个 KMeans 作为基准对比，但在 artifacts 的结果中保留 KMeans 键值。\n"
            "4. 输入变量：直接使用内存中的 X (numpy.ndarray)。沙箱同时注入 y（可能为 None）。\n"
            "5. 输出要求（metrics 字段约定）：\n"
            "   - 若数据带真实标签 y（沙箱已注入；可能为 None），**必须**用 sklearn.metrics.adjusted_rand_score(y, labels) 计算 ARI，\n"
            '     存为 metrics["ari"]，并令 metrics["score"] = metrics["ari"]，metrics["score_source"] = "ari"。\n'
            '   - 若 y 为 None，则 metrics["score"] = silhouette_score(X, labels)，metrics["score_source"] = "silhouette"。\n'
            "   - 保留 silhouette / calinski_harabasz / davies_bouldin 等辅助指标。\n"
            "   - 必须在 artifacts 字典中存储结果。示例（y 为 None 时）：\n"
            "artifacts['KMeans'] = {'labels': ..., 'metrics': {'score': 轮廓系数, 'silhouette': 轮廓系数, 'score_source': 'silhouette'}, 'plot_path': f'{ACE_OUTPUT_DIR}/centroid/kmeans.png' if ACE_OUTPUT_DIR else 'outputs/centroid/kmeans.png'}\n"
            "注意：所有 plot PNG 必须保存到 ACE_OUTPUT_DIR/centroid/ 目录下（先 os.makedirs(ACE_OUTPUT_DIR + '/centroid' if ACE_OUTPUT_DIR else 'outputs/centroid', exist_ok=True)）。"
            "只返回 Python 代码，不要有任何解释。"
        )
        user_input = f"数据集画像：{dataset.description}，样本量：{dataset.X.shape[0]}。"
        raw = client.chat_completion([{"role": "user", "content": user_input}], system_prompt)
        return _strip_code_fences(raw or "")
