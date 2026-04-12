from __future__ import annotations

import numpy as np

from ACE_Agent.agent_core.schemas import DatasetBundle, ExpertRecommendation, ProfileReport, RoutingDecision


class MasterRouter:
    def profile(self, dataset: DatasetBundle) -> ProfileReport:
        X = dataset.X
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
        ]
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

    def route(self, dataset: DatasetBundle, user_prompt: str = "") -> RoutingDecision:
        profile = self.profile(dataset)
        recommendations: list[ExpertRecommendation] = []
        prompt_lower = user_prompt.lower()

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

        if profile.manifold_hint or "降维" in user_prompt or "embedding" in prompt_lower:
            recommendations.append(
                ExpertRecommendation(
                    expert_key="dimension",
                    expert_label="维度专家",
                    priority=5 if profile.manifold_hint else 4,
                    role="primary",
                    reason="在聚类前通过降维视图可能更好地保留结构。",
                )
            )

        if profile.manifold_hint or "深度" in user_prompt or "autoencoder" in prompt_lower:
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

        deduplicated: dict[str, ExpertRecommendation] = {}
        for item in recommendations:
            existing = deduplicated.get(item.expert_key)
            if existing is None or item.priority > existing.priority:
                deduplicated[item.expert_key] = item

        selected = sorted(deduplicated.values(), key=lambda item: item.priority, reverse=True)
        trace = list(profile.notes)
        trace.extend(
            [
                f"指派 {item.expert_label} 为 {item.role} 角色，理由：{item.reason}"
                for item in selected
            ]
        )
        return RoutingDecision(profile=profile, selected_experts=selected, trace=trace)

