from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict, List
from pathlib import Path
from ACE_Agent.agent_core.schemas import AlgorithmRunResult, DatasetBundle
from ACE_Agent.tools.coder_sandbox import CoderSandbox
from ACE_Agent.tools.llm_client import UniversalLLMClient, LLMSettings

class BaseExpert(ABC):
    """专家智能体基类：支持 Think -> Act -> Fix 循环"""
    
    def __init__(self, key: str, label: str):
        self.key = key
        self.label = label
        self.sandbox = CoderSandbox()

    def execute_with_self_correction(self, dataset: DatasetBundle, prompt: str, settings: LLMSettings) -> List[AlgorithmRunResult]:
        """核心方法：具备纠错能力的专家运行逻辑"""
        client = UniversalLLMClient(settings)
        results = []
        logs = [f"【{self.label}】开始分析数据特征并生成代码..."]
        
        # 1. 初始生成代码
        code = self._generate_code(client, dataset, prompt)
        
        # 2. 尝试运行并修复 (最多3次)
        for attempt in range(1, 4):
            logs.append(f"【{self.label}】第 {attempt} 次尝试运行代码...")
            run_result = self.sandbox.execute(code, dataset.X, dataset.y)
            
            if run_result["success"]:
                logs.append(f"【{self.label}】运行成功！")
                # 构造结果对象 (简化实现)
                for algo, info in run_result["artifacts"].items():
                    results.append(AlgorithmRunResult(
                        algorithm_name=algo,
                        expert_key=self.key,
                        expert_label=self.label,
                        labels=info.get("labels"),
                        metrics=info.get("metrics", {}),
                        plot_path=Path(info.get("plot_path", "")),
                        code=code
                    ))
                break
            else:
                logs.append(f"【{self.label}】运行失败，错误信息: {run_result['error'][:200]}")
                if attempt < 3:
                    logs.append(f"【{self.label}】正在分析错误并重写代码...")
                    code = self._fix_code(client, code, run_result["error"])
                else:
                    logs.append(f"【{self.label}】重试次数耗尽，任务失败。")
        
        # 将日志注入结果，供 Supervisor 汇总展示
        self.last_logs = logs
        return results

    @abstractmethod
    def _generate_code(self, client: UniversalLLMClient, dataset: DatasetBundle, prompt: str) -> str:
        """各专家实现自己的代码生成逻辑"""
        pass

    def _fix_code(self, client: UniversalLLMClient, old_code: str, error: str) -> str:
        """通用修复逻辑"""
        system_prompt = "你是一个 Python 调试专家。修正以下代码以解决给出的错误信息。只返回修正后的 Python 代码，不要解释。"
        user_input = f"原始代码：\n{old_code}\n\n错误信息：\n{error}"
        return client.chat_completion([{"role": "user", "content": user_input}], system_prompt).strip("```python").strip("```")
