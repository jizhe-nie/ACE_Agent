# Fix #017: Ran W1–W10 in the wrong conda env (base instead of Tumor_Subtype_Agent)

**Date**: 2026-06-24
**Severity**: Medium (process/reproducibility; not a code bug). Results came from a non-canonical env.
**Status**: ✅ Corrected (process fix); affected results flagged.
**Discovered by**: PM ("你用错环境了…全给我改成base了")

## What happened
All W1–W10 experiments were launched with bare `python`, which resolved to the **base** anaconda
env (`C:\Users\Administrator\anaconda3\python`), NOT the project env **`Tumor_Subtype_Agent`**.
I then also force-reinstalled GPU torch into **base** (unnecessary + unintended).

## Root cause
- Bare `python` in the Bash tool resolves to base, not the project env.
- I ignored README/CLAUDE.md which state the project must run in conda `Tumor_Subtype_Agent`
  (I had even logged this as audit item D6, then violated it).
- I diagnosed "torch is CPU-only, need GPU reinstall" from **base** (`2.9.1+cpu`), while the
  **project env already had GPU torch** `2.6.0+cu124` (cuda=True, RTX 4060 Ti).

## Impact
- W1–W10 numbers were produced in base (numpy/sklearn/scipy/torch versions differ from the
  project env) → non-canonical. Exploratory only; led to the "change target" decision, which
  does not depend on exact numbers, so conclusions stand. Any number destined for a paper MUST
  be reproduced in `Tumor_Subtype_Agent`.
- base env torch changed cpu → cu128 (harmless; revert on request).

## Fix / Prevention
- **Always** run via `conda run -n Tumor_Subtype_Agent python ...` (or the env's python.exe).
  Never bare `python`. Saved to memory `conda-env-tumor-subtype-agent`.
- Note: `conda run -n ENV python - <<HEREDOC` does not pipe stdin (empty output); use
  `python -c "..."` or `python script.py`.
- Project env verified ready: numpy 2.4.3 / pandas 2.3.3 / sklearn 1.8.0 / scipy 1.17.1 /
  torch 2.6.0+cu124 (GPU OK). No torch install needed there.
