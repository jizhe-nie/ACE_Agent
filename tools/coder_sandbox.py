from __future__ import annotations

from dataclasses import dataclass
from typing import Any


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


class CoderSandbox:
    """Executes deterministic generated code snippets in a constrained namespace."""

    def run(self, code: str, context: dict[str, Any]) -> SandboxExecution:
        exec_env = {"__builtins__": SAFE_BUILTINS, "__name__": "__ace_sandbox__", **context}
        exec(code, exec_env, exec_env)
        result = exec_env.get("result")
        if not isinstance(result, dict):
            raise ValueError("Generated code did not expose a `result` dictionary.")
        return SandboxExecution(code=code, result=result)
