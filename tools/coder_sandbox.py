from __future__ import annotations

import traceback
from dataclasses import dataclass
from typing import Any
from loguru import logger

SAFE_BUILTINS = {
    "__build_class__": __build_class__,
    "__import__": __import__,
    "abs": abs,
    "all": all,
    "any": any,
    "Exception": Exception,
    "dict": dict,
    "enumerate": enumerate,
    "float": float,
    "getattr": getattr,
    "hasattr": hasattr,
    "int": int,
    "isinstance": isinstance,
    "len": len,
    "list": list,
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
    "zip": zip,
}


@dataclass
class SandboxExecution:
    code: str
    result: dict[str, Any]
    error: str | None = None


class CoderSandbox:
    """Executes deterministic generated code snippets in a constrained namespace."""

    def run(self, code: str, context: dict[str, Any]) -> SandboxExecution:
        exec_env = {"__builtins__": SAFE_BUILTINS, "__name__": "__ace_sandbox__", **context}
        try:
            # We use exec because we are running generated code in a controlled environment.
            exec(code, exec_env, exec_env)
            result = exec_env.get("result")
            if not isinstance(result, dict):
                raise ValueError("Generated code did not expose a `result` dictionary.")
            return SandboxExecution(code=code, result=result)
        except Exception as e:
            err_msg = f"Sandbox execution failed: {str(e)}\n{traceback.format_exc()}"
            logger.error(err_msg)
            return SandboxExecution(code=code, result={}, error=err_msg)
