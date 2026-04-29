"""
tests/test_fence_stripping.py
=============================
Regression tests for ``_strip_code_fences`` in ``expert_sub_agents/base.py``.

Covers the bug where ``str.strip("```python")`` stripped a *character set*
rather than a substring, leaving markdown fences embedded in LLM-generated
code and causing the sandbox to fail silently (empty ``artifacts``).
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root's parent is on sys.path for ACE_Agent.* imports
_root_parent = str(Path(__file__).resolve().parents[2])
if _root_parent not in sys.path:
    sys.path.insert(0, _root_parent)

from ACE_Agent.expert_sub_agents.base import _strip_code_fences  # noqa: E402


def _looks_executable(code: str) -> bool:
    """Heuristic: stripped code should begin with plausible Python, not a fence."""
    return not code.startswith("`") and (
        code.startswith("import ") or code.startswith("from ") or code.startswith("artifacts[")
    )


def test_python_fence_removed():
    raw = "```python\nimport numpy as np\nartifacts['KMeans'] = {'labels': [], 'metrics': {}, 'plot_path': ''}\n```"
    out = _strip_code_fences(raw)
    assert "```" not in out
    assert "python" not in out.splitlines()[0]  # no leading 'python' residue
    assert out.startswith("import numpy as np")
    assert "artifacts['KMeans']" in out
    assert _looks_executable(out)


def test_bare_fence_removed():
    raw = (
        "```\n"
        "import sklearn\n"
        "artifacts['DBSCAN'] = {'labels': [0, 1], 'metrics': {'score': 0.5}, 'plot_path': 'x.png'}\n"
        "```"
    )
    out = _strip_code_fences(raw)
    assert "```" not in out
    assert out.startswith("import sklearn")
    assert _looks_executable(out)


def test_no_fence_passthrough():
    raw = "import numpy as np\nartifacts['KMeans'] = {'labels': [0], 'metrics': {'score': 1.0}, 'plot_path': 'k.png'}"
    out = _strip_code_fences(raw)
    assert out == raw.strip()
    assert "```" not in out
    assert _looks_executable(out)


def test_extra_whitespace_around_fences():
    raw = (
        "   \n\n```python\n"
        "import pandas as pd\n"
        "artifacts['GMM'] = {'labels': [], 'metrics': {}, 'plot_path': ''}\n"
        "```   \n\n"
    )
    out = _strip_code_fences(raw)
    assert "```" not in out
    assert out.startswith("import pandas as pd")
    assert out.endswith("'plot_path': ''}")
    assert _looks_executable(out)


def test_py_short_fence_removed():
    """Optional: `” ```py ” fences (shorthand) should also be stripped."""
    raw = (
        "```py\n"
        "from sklearn.cluster import KMeans\n"
        "artifacts['KMeans'] = {'labels': [], 'metrics': {}, 'plot_path': ''}\n"
        "```"
    )
    out = _strip_code_fences(raw)
    assert "```" not in out
    assert out.startswith("from sklearn.cluster import KMeans")
    assert _looks_executable(out)


def test_none_and_empty_inputs():
    """Defensive: None / empty should not raise."""
    assert _strip_code_fences(None) == ""  # type: ignore[arg-type]
    assert _strip_code_fences("") == ""
    assert _strip_code_fences("   \n\n") == ""
