---
name: ACE Agent Test Conventions
description: How to run tests, what exists, known pitfalls
type: feedback
---

Run tests from `D:/PycharmProject` (not from within `ACE_Agent/`) so that `ACE_Agent.*` package imports resolve correctly:
```
cd /d/PycharmProject
C:/Users/Administrator/anaconda3/envs/Tumor_Subtype_Agent/python.exe -m pytest ACE_Agent/tests/ -v
```

Test files use `sys.path.insert(0, parents[2])` before ACE_Agent imports — this triggers E402/I001 ruff warnings. Fixed via `per-file-ignores` in pyproject.toml: `"tests/*.py" = ["E402", "I001"]`.

`test_follow_up.py` is an integration test that requires a live LLM API key. It is guarded with `@pytest.mark.skipif(not settings.is_configured)` — will skip in CI without credentials.

`test_core.py` dataset names: use only names from `list_demo_datasets()` — does NOT include `circles`. Valid: blobs, moons, s_curve, smile, high_dim, multi_view, iris, wine, digits, mnist, news, mfeat, custom.

**Why**: Original test_core.py used `router.route()` (non-existent) and `supervisor.run(dataset)` (missing required args) — code evolved past the tests.
**How to apply**: Always run a quick sanity import test before writing new tests.
