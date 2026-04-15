from __future__ import annotations

import numpy as np
from loguru import logger
from scipy.stats import skew, kurtosis
from ACE_Agent.agent_core.schemas import ChatMessage, DatasetBundle, ExpertRecommendation, ProfileReport, RoutingDecision
from ACE_Agent.tools.llm_client import LLMSettings, OpenAICompatibleClient


class MasterRouter:
    def analyze_intent(self, user_prompt: str, history: list[ChatMessage], llm_settings: LLMSettings | None = None) -> str:
        """
        分析用户意图。返回 "NEW_TASK" 或 "FOLLOW_UP"。
        """
        if not history:
            return "NEW_TASK"

        if llm_settings and llm_settings.is_configured:
            client = OpenAICompatibleClient(llm_settings)
            history_text = "\n".join([f"{m.role}: {m.content}" for m in history[-3:]])
            prompt = (
                f"你是一个意图识别专家。根据对话历史和当前输入，判断用户是在开启一个全新的聚类任务（NEW_TASK），还是在针对上一次聚类结果进行追问、探讨或寻求解释（FOLLOW_UP）。\n\n"
                f"对话历史:\n{history_text}\n\n"
                f"当前输入: {user_prompt}\n\n"
                "仅输出 'NEW_TASK' 或 'FOLLOW_UP'，不要有任何额外文字。"
            )
            try:
                # 复用 summarize_report 逻辑
                intent = client.summarize_report({"custom_request": prompt}).strip().upper()
                if "NEW_TASK" in intent:
                    return "NEW_TASK"
                if "FOLLOW_UP" in intent:
                    return "FOLLOW_UP"
            except:
                pass

        # 降级逻辑：关键词判断
        follow_up_keywords = {"为什么", "解释", "细节", "再", "详细", "参数", "指标", "那个", "此", "它", "这个", "具体", "为什么"}
        if any(kw in user_prompt for kw in follow_up_keywords):
            return "FOLLOW_UP"

        return "NEW_TASK"

    def profile(self, dataset: DatasetBundle) -> ProfileReport:
        X = dataset.X
        # 统计特征
        avg_skew = float(np.mean(np.abs(skew(X, axis=0))))
        avg_kurtosis = float(np.mean(np.abs(kurtosis(X, axis=0))))
        
        negative_ratio = float(np.mean(X < 0))
        sparsity_ratio = float(np.mean(np.isclose(X, 0.0, atol=1e-8)))
        if X.shape[1] > 1:
            corr = np.corrcoef(X, rowvar=False)
            corr = np.nan_to_num(corr, nan=0.0)
            upper = corr[np.triu_indices_from(corr, k=1)]
            avg_abs_correlation = float(np.mean(np.abs(upper))) if upper.size else 0.0
        else:
            avg_abs_correlation = 0.0

        manifold_hint = dataset.shape_family == "manifold" or X.shape[1] > 2
        non_convex_hint = dataset.shape_family in {"non_convex", "manifold"}
        noise_sensitive_hint = dataset.name in {"moons", "smile"}
        expected_clusters = dataset.metadata.get("expected_clusters")

        notes = [
            f"扫描了 {X.shape[0]} 个样本，包含 {X.shape[1]} 个特征。",
            f"负值比例为 {negative_ratio:.2%}；稀疏度为 {sparsity_ratio:.2%}。",
            f"特征间的平均绝对相关系数为 {avg_abs_correlation:.3f}。",
            f"平均偏度为 {avg_skew:.3f}；平均峰度为 {avg_kurtosis:.3f}。",
        ]
        if avg_skew > 1.5:
            notes.append("检测到显著的数据偏态，可能影响依赖高斯分布假设的算法。")
        if non_convex_hint:
            notes.append("检测到可能的非凸几何结构，提升拓扑感知方法的优先级。")
        if manifold_hint:
            notes.append("检测到流形或高维结构，提升嵌入（Embedding）专家的优先级。")
        if avg_abs_correlation > 0.45:
            notes.append("强相关性表明降维可以简化几何结构。")

        return ProfileReport(
            sample_count=X.shape[0],
            feature_count=X.shape[1],
            negative_ratio=negative_ratio,
            sparsity_ratio=sparsity_ratio,
            avg_abs_correlation=avg_abs_correlation,
            manifold_hint=manifold_hint,
            non_convex_hint=non_convex_hint,
            noise_sensitive_hint=noise_sensitive_hint,
            expected_clusters=expected_clusters,
            notes=notes,
        )

    def route(self, dataset: DatasetBundle, user_prompt: str = "", llm_settings: LLMSettings | None = None) -> RoutingDecision:
        profile = self.profile(dataset)
        recommendations: list[ExpertRecommendation] = []
        prompt_lower = user_prompt.lower()
        trace = list(profile.notes)

        # 1. 基础硬编码逻辑
        self._apply_baseline_rules(profile, recommendations, dataset, prompt_lower)

        # 2. 如果 LLM 可用，通过 LLM 增强路由建议
        if llm_settings and llm_settings.is_configured:
            llm_notes = self._get_llm_routing_advice(profile, user_prompt, llm_settings)
            if llm_notes:
                trace.append(f"LLM 智能画像意见: {llm_notes}")
                # 根据 LLM 意见微调，此处可扩展更复杂的解析
                if "topology" in llm_notes.lower():
                    for rec in recommendations:
                        if rec.expert_key == "topology":
                            rec.priority += 1
                            rec.reason += " (LLM 画像建议提升优先级)"

        deduplicated: dict[str, ExpertRecommendation] = {}
        for item in recommendations:
            existing = deduplicated.get(item.expert_key)
            if existing is None or item.priority > existing.priority:
                deduplicated[item.expert_key] = item

        selected = sorted(deduplicated.values(), key=lambda item: item.priority, reverse=True)
        trace.extend(
            [
                f"指派 {item.expert_label} 为 {item.role} 角色，理由：{item.reason}"
                for item in selected
            ]
        )
        return RoutingDecision(profile=profile, selected_experts=selected, trace=trace)

    def _apply_baseline_rules(self, profile, recommendations, dataset, prompt_lower):
        if profile.non_convex_hint:
            recommendations.append(
                ExpertRecommendation(
                    expert_key="topology",
                    expert_label="拓扑专家",
                    priority=5,
                    role="primary",
                    reason="存在非凸结构的概率较大，应以密度和连通性方法为主。",
                )
            )
            recommendations.append(
                ExpertRecommendation(
                    expert_key="centroid",
                    expert_label="质心专家",
                    priority=2,
                    role="challenger",
                    reason="质心方法优先级降低，仅作为基准对比保留。",
                )
            )
        else:
            recommendations.append(
                ExpertRecommendation(
                    expert_key="centroid",
                    expert_label="质心专家",
                    priority=5,
                    role="primary",
                    reason="数据集分布较为紧凑，质心算法是强力候选方案。",
                )
            )
            recommendations.append(
                ExpertRecommendation(
                    expert_key="topology",
                    expert_label="拓扑专家",
                    priority=3,
                    role="baseline",
                    reason="保留拓扑方法以进行结构假设的压力测试。",
                )
            )

        if profile.manifold_hint or "降维" in prompt_lower or "embedding" in prompt_lower:
            recommendations.append(
                ExpertRecommendation(
                    expert_key="dimension",
                    expert_label="维度专家",
                    priority=5 if profile.manifold_hint else 4,
                    role="primary",
                    reason="在聚类前通过降维视图可能更好地保留结构。",
                )
            )

        if profile.manifold_hint or "深度" in prompt_lower or "autoencoder" in prompt_lower:
            recommendations.append(
                ExpertRecommendation(
                    expert_key="deep",
                    expert_label="深度表征专家",
                    priority=4,
                    role="support",
                    reason="非线性潜空间可能提高弯曲结构的建模能力。",
                )
            )

        recommendations.append(
            ExpertRecommendation(
                expert_key="multi_view",
                expert_label="多视图专家",
                priority=4 if dataset.name in {"smile", "moons"} else 3,
                role="support",
                reason="当多个专家意见不一时，共识融合有助于获得稳定划分。",
            )
        )

    def _get_llm_routing_advice(self, profile: ProfileReport, user_prompt: str, llm_settings: LLMSettings) -> str | None:
        client = OpenAICompatibleClient(llm_settings)
        # 构造画像 payload
        payload = {
            "samples": profile.sample_count,
            "features": profile.feature_count,
            "negative_ratio": f"{profile.negative_ratio:.2%}",
            "sparsity": f"{profile.sparsity_ratio:.2%}",
            "avg_abs_correlation": profile.avg_abs_correlation,
            "user_prompt": user_prompt,
            "system_notes": profile.notes
        }
        prompt = (
            f"你是一个资深数据分析专家。根据以下数据集画像，请简要分析该数据集适合哪类聚类专家（质心、拓扑、维度、深度、多视图）。\n"
            f"画像详情: {str(payload)}\n"
            "请用一两句话给出你的专业建议，并提及最推荐的专家关键字（如：topology, centroid 等）。"
        )
        # 此处复用 summarize_report 逻辑，实际可扩展专门的推理接口
        # 为了演示，我们暂时使用 client 内部方法
        try:
            return client.summarize_report({"custom_request": prompt})
        except:
            return None

