from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Any

from ACE_Agent.agent_core.schemas import AlgorithmRunResult, DatasetBundle
from ACE_Agent.expert_sub_agents.base import BaseExpert
from ACE_Agent.tools.algorithm_zoo import AlgorithmZoo

from concurrent.futures import ThreadPoolExecutor, as_completed

class ZooExpert(BaseExpert):
    key = "zoo"
    label = "全量算法专家"

    def run(self, dataset: DatasetBundle, output_dir: Path, algorithm_names: list[str] | None = None) -> list[AlgorithmRunResult]:
        """
        运行 Zoo 中的指定算法或全部算法。
        """
        all_algos = AlgorithmZoo.get_all_algorithms()
        
        # 如果没有指定，默认运行全部（ExhaustiveMode 核心逻辑）
        if algorithm_names is None:
            target_algos = all_algos
        else:
            target_algos = [a for a in all_algos if a["name"] in algorithm_names]
        
        expected_clusters = dataset.metadata.get("expected_clusters", 3)
        results = []

        # 并行执行算法以提升 ExhaustiveMode 效率
        with ThreadPoolExecutor(max_workers=min(len(target_algos), 8)) as executor:
            future_to_algo = {}
            for algo in target_algos:
                # 动态填充参数
                params = {}
                for k, v in algo["params"].items():
                    if v == "expected_clusters":
                        params[k] = expected_clusters
                    else:
                        params[k] = v
                
                code = AlgorithmZoo.get_algorithm_code(algo["name"], params, f"{self.label} - {algo['name']}")
                
                future = executor.submit(
                    self._execute_code,
                    dataset=dataset,
                    output_dir=output_dir,
                    algorithm_name=algo["name"],
                    params=params,
                    code=code,
                    plot_filename=f"{dataset.name}_zoo_{algo['name'].lower()}.png",
                    trace=[
                        f"从 AlgorithmZoo 自动配置了 {algo['name']}。",
                        f"使用参数: {params}",
                    ]
                )
                future_to_algo[future] = algo["name"]

            for future in as_completed(future_to_algo):
                algo_name = future_to_algo[future]
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                except Exception as e:
                    import logging
                    logging.error(f"ZooExpert failed to run {algo_name}: {e}")

        return results
