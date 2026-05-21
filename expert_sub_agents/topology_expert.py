from __future__ import annotations

from ACE_Agent.agent_core.schemas import DatasetBundle
from ACE_Agent.expert_sub_agents.base import BaseExpert, _strip_code_fences
from ACE_Agent.tools.llm_client import UniversalLLMClient


class TopologyExpert(BaseExpert):
    """拓扑专家 v2.0：OPTICS 可达性图 + k-NN 局部连接谱聚类"""

    def __init__(self):
        super().__init__(key="topology", label="拓扑专家")

    def _generate_code(
        self,
        client: UniversalLLMClient,
        dataset: DatasetBundle,
        prompt: str,
        constraints=None,
    ) -> str:
        constraint_prompt = self._inject_constraints_prompt(constraints)

        # ---- Time-series modality hint for DTW-aware topology ----------
        _ts_hint = ""
        _md = dataset.metadata or {}
        if _md.get("is_time_series") and _md.get("ts_shape"):
            _ts_T, _ts_F = _md["ts_shape"]
            _n_samp = dataset.X.shape[0]
            _ts_hint = (
                "\n## ⏱️ 时间序列模态 (DTW 拓扑分析)\n\n"
                f"CTX_DATA.metadata 中 is_time_series=True, ts_shape=({_ts_T}, {_ts_F})。\n"
                f"数据已展平为 (N, {_ts_T * _ts_F})。在进行拓扑分析之前：\n"
                f"  X_ts = X.reshape(CTX_DATA.n_samples, {_ts_T}, {_ts_F})\n\n"
                "DTW 距离在时间序列空间中的使用规则：\n"
                "1. tslearn 已预注入：TimeSeriesKMeans 和 tslearn_metrics 可直接使用"
                "（无需 import）。\n"
                "2. 使用 tslearn_metrics.cdist_dtw(X_ts) 计算成对 DTW 距离矩阵。\n"
                "3. 将 DTW 距离矩阵作为 OPTICS/DBSCAN 的 metric='precomputed' 输入：\n"
                "   OPTICS(metric='precomputed', min_samples=...).fit(dtw_dist_matrix)\n"
                "4. 或者转为亲和矩阵用于 SpectralClustering：\n"
                "   aff = np.exp(-dtw_dist / sigma)\n"
                "   SpectralClustering(affinity='precomputed').fit(aff)\n"
                f"5. 样本量 N={_n_samp}。"
            )
            if _n_samp > 500:
                _ts_hint += (
                    f"\n   ⚠️ N={_n_samp}>500：DTW 使用 Sakoe-Chiba 带加速：\n"
                    "   cdist_dtw(X_ts, global_constraint='sakoe_chiba',"
                    " sakoe_chiba_radius=2)\n"
                )
            _ts_hint += (
                "\n优先使用 DTW 距离（而非欧氏距离）进行拓扑分析。\n"
                "欧氏 OPTICS/DBSCAN 仍然作为对比运行，但 DTW 结果写入"
                " artifacts['OPTICS_DTW'] 和 artifacts['SpectralDTW']。\n\n"
            )

        system_prompt = constraint_prompt + _ts_hint + (
            "你是一个高级 Python 数据科学专家（拓扑与密度算法分支 v2.0）。\n"
            "## 核心指令：OPTICS 可达性优先 + Mutual k-NN 局部连接\n\n"
            "### 1. 必须显式包含所有导入（严禁省略！）：\n"
            "```python\n"
            "import warnings\n"
            "import numpy as np\n"
            "from sklearn.preprocessing import StandardScaler\n"
            "from sklearn.neighbors import NearestNeighbors, kneighbors_graph\n"
            "from sklearn.cluster import OPTICS, DBSCAN, SpectralClustering\n"
            "try: from sklearn.cluster import HDBSCAN\n"
            "except ImportError: from hdbscan import HDBSCAN\n"
            "from sklearn.metrics import adjusted_rand_score, silhouette_score\n"
            "warnings.filterwarnings('ignore')\n"
            "```\n\n"
            "### 2. OPTICS 可达性分析（主算法，必须执行）：\n"
            "- 先用 StandardScaler 处理 X。\n"
            "- 运行 OPTICS(min_samples=5, xi=0.05, min_cluster_size=0.05)：\n"
            "  * xi=0.05 控制可达性图中陡峭度阈值，越小越敏感。\n"
            "  * 获取 `reachability_` 和 `ordering_` 属性。\n"
            "- **可达性图山谷检测**：\n"
            "  * 按 ordering_ 排列 reachability_，形成可达性曲线。\n"
            "  * 检测「山谷」— 连续区间内 reachability 显著低于两侧的区域。\n"
            "  * 方法: 对可达性曲线做 1D 高斯平滑 (sigma=2)，在平滑曲线上找\n"
            "    连续低值区 (低于均值 - 0.5*std 且长度 >= min_samples)。\n"
            "  * 山谷的数量即为自然聚类数的估计 `k_valleys`。\n"
            "- **动态 eps 提取**：\n"
            "  * 对每个山谷，取山谷内 reachability 的 90% 分位数作为局部 eps。\n"
            "  * 取所有山谷局部 eps 的加权中位数作为全局 eps_optics。\n"
            "- 将 OPTICS labels 写入 artifacts[\"OPTICS\"]。\n\n"
            "### 3. DBSCAN 参数锁定（K-距离图动态 eps，严禁硬编码）：\n"
            "- **无论 OPTICS 是否成功，DBSCAN 也必须执行**（独立对比验证）。\n"
            "- 用 NearestNeighbors(n_neighbors=min_samples*2, metric='euclidean') 拟合 X_scaled。\n"
            "- 取第 k=min_samples 近邻距离，排序后绘制 K-距离曲线。\n"
            "- **5+ eps 扫描**（非单点肘部！）：\n"
            "  * 在排序后的 k-dist 曲线上，从 5% 到 95% 百分位等距取 **至少 5 个 eps 候选值**。\n"
            "  * 对每个 eps 候选运行 DBSCAN(eps=eps_i, min_samples=min_samples)。\n"
            "  * 记录每个 eps 的: 簇数、噪声率、silhouette_score。\n"
            "  * 排除 噪声率>50% 或 簇数<2 的 eps。\n"
            "  * 在剩余候选中，取 silhouette 最高的 eps 作为 eps_best。\n"
            "  * 若所有候选都被排除，取 k-dist 曲线最大曲率点（二阶差分极大值）作为 eps。\n"
            "- **min_samples 动态选择**: min_samples = max(3, min(2*dim, n//200))。\n"
            "- 运行 DBSCAN(eps=eps_best, min_samples=min_samples)，将 labels 写入 artifacts[\"DBSCAN\"]。\n"
            "- 若 DBSCAN 全噪声或仅 1 簇，回退到 HDBSCAN(min_cluster_size=min_samples)。\n\n"
            "### 4. 谱聚类 Mutual k-NN 局部相似性（必须执行）：\n"
            "- **严禁使用全局 RBF 核**（`SpectralClustering(affinity='rbf')`）！\n"
            "- **强制使用 Mutual k-NN（互近邻图）替代单向 k-NN**：\n"
            "  * 单向 k-NN 会在两条邻近曲线/流形之间建立跨类 short-cut 边。\n"
            "  * Mutual k-NN 仅当 A 是 B 的近邻 AND B 也是 A 的近邻时才连边。\n"
            "  * 这会自然消除不同抛物线/流形之间的「短路边」。\n"
            "  * **自适应 k 值**：k 需要足够大以保证稀疏流形类内连通，mutual 过滤会自然去噪：\n"
            "  ```python\n"
            "  _k_nn = max(8, int(np.sqrt(_n_sample)) // 2)\n"
            "  # 若 spectral 全噪声 → k *= 1.5 重新运行（最多 3 次）\n"
            "  _adj_fwd = kneighbors_graph(_X_scaled, _k_nn, mode='connectivity', include_self=False)\n"
            "  _adj_rev = _adj_fwd.T\n"
            "  # Keep only mutual (bidirectional) edges — natural denoising\n"
            "  _adj_mutual = _adj_fwd.minimum(_adj_rev)\n"
            "  ```\n"
            "- 运行 SpectralClustering(n_clusters=k_valleys, affinity='precomputed') 时传入 _adj_mutual。\n"
            "- 若 adjacency 不连通，对每个连通分量分别 SpectralClustering。\n"
            "- 将 labels 写入 artifacts[\"Spectral_kNN\"]。\n\n"
            "### 5. 可视化合约（每个算法必做）：\n"
            "- 所有 plot PNG 保存到 ACE_OUTPUT_DIR/topology/ 目录下（若 ACE_OUTPUT_DIR 为空则用 outputs/topology/）。\n"
            "  代码开头执行: _out_top = ACE_OUTPUT_DIR + '/topology' if ACE_OUTPUT_DIR else 'outputs/topology'; os.makedirs(_out_top, exist_ok=True)\n"
            '- 散点图 PNG 按算法命名保存到 _out_top 下。\n'
            "- 噪声点 (label=-1) 用灰色，正常簇用不同颜色。\n"
            '- OPTICS 额外生成可达性图 PNG，保存为 _out_top/optics_reachability.png。\n'
            "  ```python\n"
            "  _fig, _ax = plt.subplots(figsize=(8, 3))\n"
            '  _ax.plot(np.arange(len(_reach)), _reach[_opt.ordering_], "b-", lw=0.6)\n'
            '  _ax.set_xlabel("Order"); _ax.set_ylabel("Reachability dist.")\n'
            '  _fig.savefig(_out_top + "/optics_reachability.png", dpi=150, bbox_inches="tight")\n'
            "  plt.close(_fig)\n"
            "  ```\n"
            "### 6. 结构约束：\n"
            "- 代码在顶层直接运行。\n"
            "- 结果写入 artifacts[algo_name] = {labels, metrics, plot_path}，plot_path 指向 _out_top/ 下的文件。\n"
            "- 代码在顶层直接运行。\n"
            "- 结果写入 artifacts[algo_name] = {labels, metrics, plot_path}。\n"
            "- 若 y 非 None，metrics[\"score\"] = ARI(y, labels)，score_source=\"ari\"。\n"
            "- 若 y 为 None（无标签），metrics[\"score\"] = silhouette_score(X_scaled, labels)，score_source=\"silhouette\"。严禁设置 score=None！\n"
            "只返回 Python 代码，不要有任何解释。"
        )
        user_input = (
            f"为样本量 {dataset.X.shape[0]} 的数据集生成 OPTICS 可达性优先的拓扑聚类代码。"
            f"数据集可能为极窄间距的并行曲线/抛物线结构，请严格遵循可达性图山谷检测和 Mutual k-NN 局部连接策略。"
        )
        raw = client.chat_completion(
            [{"role": "user", "content": user_input}],
            system_prompt,
        )
        return _strip_code_fences(raw or "")
