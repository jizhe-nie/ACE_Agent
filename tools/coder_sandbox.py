from __future__ import annotations

import traceback
from dataclasses import dataclass
from typing import Any
from loguru import logger
import numpy as np

SAFE_BUILTINS = {
    "__build_class__": __build_class__,
    "__import__": __import__,
    "abs": abs,
    "all": all,
    "any": any,
    "Exception": Exception,
    "dict": dict,
    "dir": dir,
    "enumerate": enumerate,
    "float": float,
    "getattr": getattr,
    "globals": globals,
    "hasattr": hasattr,
    "int": int,
    "isinstance": isinstance,
    "len": len,
    "list": list,
    "locals": locals,
    "max": max,
    "min": min,
    "object": object,
    "print": print,
    "range": range,
    "round": round,
    "set": set,
    "str": str,
    "sum": sum,
    "super": super,
    "tuple": tuple,
    "vars": vars,
    "zip": zip,
}


@dataclass
class SandboxExecution:
    code: str
    result: dict[str, Any]
    error: str | None = None


class CoderSandbox:
    """在受限命名空间内执行生成的 Python 代码片段。"""

    def execute(self, code: str, X: np.ndarray, y: np.ndarray | None = None) -> dict[str, Any]:
        """为专家提供的便捷执行接口。"""
        # 准备执行上下文
        # 我们期望生成的代码向 artifacts 字典中写入结果
        artifacts = {}
        exec_env = {
            "__builtins__": SAFE_BUILTINS,
            "__name__": "__ace_sandbox__",
            "X": X,
            "y": y,
            "artifacts": artifacts,
            "np": np, # 允许代码使用 numpy
        }
        
        try:
            # 导入并配置库
            import matplotlib.pyplot as plt
            import platform
            # 解决中文乱码
            if platform.system() == "Windows":
                plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
            plt.rcParams['axes.unicode_minus'] = False
            exec_env["plt"] = plt
            
            # 执行代码
            exec(code, exec_env, exec_env)
            
            # 如果代码没有向 artifacts 写入任何内容，尝试查找执行环境中的变量
            if not artifacts and "result" in exec_env:
                artifacts = exec_env["result"]

            return {
                "success": True,
                "artifacts": artifacts,
                "error": None
            }
        except Exception as e:
            err_msg = f"沙箱执行失败: {str(e)}\n{traceback.format_exc()}"
            logger.error(err_msg)
            return {
                "success": False,
                "artifacts": {},
                "error": err_msg
            }

    def run(self, code: str, context: dict[str, Any]) -> SandboxExecution:
        """旧接口兼容逻辑。"""
        exec_env = {"__builtins__": SAFE_BUILTINS, "__name__": "__ace_sandbox__", **context}
        try:
            exec(code, exec_env, exec_env)
            result = exec_env.get("result", exec_env.get("artifacts", {}))
            return SandboxExecution(code=code, result=result)
        except Exception as e:
            err_msg = f"Sandbox execution failed: {str(e)}\n{traceback.format_exc()}"
            return SandboxExecution(code=code, result={}, error=err_msg)
