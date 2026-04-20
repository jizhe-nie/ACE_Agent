"""
tests/test_sandbox_rescue.py
============================
Unit tests for the broadened sandbox rescue logic in
``tools/coder_sandbox.py``.

Three real LLM-generated code failure modes are covered:

1. ``if __name__ == "__main__":`` guard — under the sandbox ``__name__``
   is ``"__ace_sandbox__"``, so anything behind the guard never runs.
   If the user ALSO wrote to an artifacts-shaped dict at module level
   (before the guard), the rescue should pick it up. If the guard is
   the ONLY place results are populated, artifacts must remain empty
   (we cannot rescue what never executed).

2. Arbitrarily-named result dict — e.g. ``output = {"KMeans": {...}}``.
   The broad scan should find it regardless of variable name.

3. Backward compatibility — the legacy rescue of a dict named
   ``result`` must still work.

Also verifies that innocuous dicts (e.g. ``config = {"seed": 42}``)
are NOT misidentified as artifacts.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_root_parent = str(Path(__file__).resolve().parents[2])
if _root_parent not in sys.path:
    sys.path.insert(0, _root_parent)

from ACE_Agent.tools.coder_sandbox import CoderSandbox, _looks_like_artifacts  # noqa: E402


# ---------------------------------------------------------------------------
# _looks_like_artifacts unit tests
# ---------------------------------------------------------------------------


class TestLooksLikeArtifacts:
    def test_valid_artifacts_shape(self) -> None:
        v = {"KMeans": {"labels": [0, 1, 0], "metrics": {"score": 0.5}, "plot_path": "k.png"}}
        assert _looks_like_artifacts(v) is True

    def test_valid_with_only_labels(self) -> None:
        # metrics / plot_path are optional for detection purposes
        v = {"DBSCAN": {"labels": [0, 1]}}
        assert _looks_like_artifacts(v) is True

    def test_empty_dict_is_not_artifacts(self) -> None:
        assert _looks_like_artifacts({}) is False

    def test_non_dict_values_are_not_artifacts(self) -> None:
        assert _looks_like_artifacts({"seed": 42, "name": "foo"}) is False

    def test_config_dict_is_not_artifacts(self) -> None:
        # This is the kind of dict that must NOT be rescued
        cfg = {"seed": 42, "n_clusters": 3, "algo": "KMeans"}
        assert _looks_like_artifacts(cfg) is False

    def test_nested_dict_without_labels_key(self) -> None:
        v = {"foo": {"bar": 1, "baz": 2}}
        assert _looks_like_artifacts(v) is False

    def test_non_dict_input(self) -> None:
        assert _looks_like_artifacts([1, 2, 3]) is False
        assert _looks_like_artifacts(None) is False
        assert _looks_like_artifacts("hello") is False


# ---------------------------------------------------------------------------
# End-to-end sandbox rescue tests
# ---------------------------------------------------------------------------


def _X() -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.standard_normal((20, 3))


class TestSandboxRescue:
    def test_result_variable_still_rescued(self) -> None:
        """Backward compat: `result = {...}` rescue still works."""
        code = (
            "result = {'KMeans': {'labels': [0, 1, 0], "
            "'metrics': {'score': 0.5}, 'plot_path': 'k.png'}}\n"
        )
        sbx = CoderSandbox()
        out = sbx.execute(code, _X())
        assert out["success"] is True
        assert "KMeans" in out["artifacts"]
        assert out["artifacts"]["KMeans"]["labels"] == [0, 1, 0]

    def test_broad_scan_finds_artifacts_shaped_dict_with_any_name(self) -> None:
        """A dict named anything (not `artifacts`/`result`) should be rescued."""
        code = (
            "output = {'KMeans': {'labels': [0, 1, 2], "
            "'metrics': {'score': 0.7}, 'plot_path': 'k.png'}}\n"
        )
        sbx = CoderSandbox()
        out = sbx.execute(code, _X())
        assert out["success"] is True
        assert "KMeans" in out["artifacts"]
        assert out["artifacts"]["KMeans"]["metrics"]["score"] == 0.7

    def test_non_artifacts_shaped_dict_ignored(self) -> None:
        """A config-style dict must NOT be mistaken for artifacts."""
        code = "config = {'seed': 42, 'n_clusters': 3}\n"
        sbx = CoderSandbox()
        out = sbx.execute(code, _X())
        assert out["success"] is True
        assert out["artifacts"] == {}

    def test_main_guard_not_rescued_but_broad_scan_finds_module_level(self) -> None:
        """Code that writes an artifacts-shaped dict at module level BEFORE
        a ``if __name__ == "__main__":`` guard should be rescued.
        Code that only writes inside the guard should leave artifacts empty.
        """
        # Case A: module-level write BEFORE guard — rescuable.
        code_a = (
            "output = {'KMeans': {'labels': [0, 1, 0], "
            "'metrics': {'score': 0.4}, 'plot_path': 'k.png'}}\n"
            "if __name__ == '__main__':\n"
            "    print('not reached in sandbox')\n"
        )
        sbx = CoderSandbox()
        out_a = sbx.execute(code_a, _X())
        assert out_a["success"] is True
        assert "KMeans" in out_a["artifacts"]

        # Case B: write ONLY inside the guard — unrescuable. Cannot fix what
        # never ran.
        code_b = (
            "if __name__ == '__main__':\n"
            "    artifacts['KMeans'] = {'labels': [0, 1], "
            "'metrics': {'score': 0.3}, 'plot_path': 'k.png'}\n"
        )
        out_b = sbx.execute(code_b, _X())
        assert out_b["success"] is True
        assert out_b["artifacts"] == {}

    def test_def_never_called_not_rescuable(self) -> None:
        """Code that defines a function writing to artifacts but never calls
        it should remain empty — there is nothing to rescue."""
        code = (
            "def run():\n"
            "    artifacts['KMeans'] = {'labels': [0], "
            "'metrics': {'score': 0.1}, 'plot_path': 'k.png'}\n"
        )
        sbx = CoderSandbox()
        out = sbx.execute(code, _X())
        assert out["success"] is True
        # The function was never called; the `run` binding is callable and
        # gets skipped by the scanner. artifacts must stay empty.
        assert out["artifacts"] == {}

    def test_direct_artifacts_write_still_works(self) -> None:
        """Sanity: the normal happy path (writing directly to ``artifacts``)
        is untouched by the broadened rescue."""
        code = (
            "artifacts['KMeans'] = {'labels': [0, 1], "
            "'metrics': {'score': 0.9}, 'plot_path': 'k.png'}\n"
        )
        sbx = CoderSandbox()
        out = sbx.execute(code, _X())
        assert out["success"] is True
        assert out["artifacts"]["KMeans"]["metrics"]["score"] == 0.9
