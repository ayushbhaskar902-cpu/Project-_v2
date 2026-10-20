# P1-16 Final Verification Report — Quality, Eligibility, and Submission State

**Issue:** P1-16
**Status:** COMPLETE ✅
**Verified:** 2026-10-20

## Implemented boundaries

- `EvaluationScorecard.meets_internal_thresholds` is the explicit internal heuristic result.
- `is_submission_grade` remains only as a deprecated compatibility alias for that internal result.
- `ResearchCandidate.platform_eligibility_status` persists `UNCHECKED`, `CHECK_PENDING`, `ELIGIBLE`, `INELIGIBLE`, or `EXEMPT_OFFLINE` independently of research lifecycle status.
- `EvaluationRecord.internal_checks` stores local threshold checks; `EvaluationRecord.checks` is reserved for authoritative BRAIN check evidence.
- `EvaluationEngine.record_platform_eligibility()` refuses live eligibility for unverified provenance or incomplete/unverified simulation settings and never changes internal acceptance.
- Submission is represented only by an authoritative `Submission` row; internal acceptance, statistical selection, graduation, and platform eligibility do not imply submission.
- `build_canonical_spec()` now uses the candidate's persisted effective simulation settings rather than hard-coded values.
- API and frontend labels identify the compatibility field as internal quality, and expose platform eligibility/submission fields separately.
- Researcher memory and leaderboard counts no longer treat platform-only check payloads as internal quality passes; legacy counter-only rows remain readable.

## Regression evidence

Targeted command:

```text
uv run pytest tests/test_p1_16_platform_states.py tests/test_p1_11_fair_model_leaderboard.py::test_submission_volume_alone_cannot_improve_quality_rank -q --tb=short
Exit code: 0
5 passed in 3.07s
```

Full backend command:

```text
uv run pytest --tb=short -q
Exit code: 0
329 passed, 24 warnings in 98.25s
```

The warnings are existing SQLite/aiosqlite connection teardown warnings and remain tracked under `DEFERRED-TEST-01`; no failures or errors occurred.

## Acceptance criteria

| Criterion | Result |
|---|---|
| Internal threshold pass does not imply platform eligibility | ✅ |
| Statistical/lifecycle acceptance does not imply submission | ✅ |
| Unverified provenance cannot receive live eligibility | ✅ |
| Incomplete settings cannot receive live eligibility | ✅ |
| Canonical graduation settings match persisted candidate settings | ✅ |
| Holdout remains outside eligibility state changes | ✅ |
| Full regression suite | ✅ |
