"""P2-06 formula versioning, integrity checks, and legacy compatibility."""

from __future__ import annotations

import asyncio
import json
import math
import sqlite3
from datetime import date, timedelta
from types import SimpleNamespace
from typing import TYPE_CHECKING

import numpy as np
import pytest
from sqlalchemy import select

from alpha_harness.api.research import (
    CandidateOut,
    get_alpha_leaderboard,
    list_candidates,
    list_evaluations,
)
from alpha_harness.db.models import CandidateStatus, ResearchCandidate, ResearchEvaluation, ResearchSession
from alpha_harness.db.sqlite import Database
from alpha_harness.evaluation.metrics import (
    LEGACY_COMPOSITE_CURRENT_VERSION,
    LEGACY_COMPOSITE_VERSION,
    VALIDATION_COMPOSITE_VERSION,
    composite_score_spec,
    compute_composite_score,
    compute_legacy_composite_score_v2,
    compute_validation_composite_score,
    evaluate_pnl_series,
    recompute_composite_score,
)
from alpha_harness.portfolio.pool import AlphaPoolOptimizer
from alpha_harness.leaderboard.service import LeaderboardService
from alpha_harness.learning.memory import ResearchMemory
from alpha_harness.vault.metrics import _number, stats

if TYPE_CHECKING:
    from pathlib import Path


def test_formula_golden_vectors_and_zero_turnover_compatibility() -> None:
    assert compute_validation_composite_score(2.0, 1.0, 0.0) == 1.35
    assert compute_composite_score(2.0, 1.0, 0.0, 0.0) == 1.28
    assert compute_composite_score(2.0, 1.0, 0.0, None) == 1.28
    assert compute_legacy_composite_score_v2(2.0, 1.0, 0.0, 0.0) == 1.30
    assert compute_legacy_composite_score_v2(2.0, 1.0, 0.0, None) == 1.28


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_formula_boundaries_reject_non_finite_values(value: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        compute_validation_composite_score(value, 1.0, 0.1)
    with pytest.raises(ValueError, match="finite"):
        compute_legacy_composite_score_v2(1.0, value, 0.1, 0.1)
    with pytest.raises(ValueError, match="finite"):
        _number(value)
    with pytest.raises(ValueError, match="finite"):
        stats(np.array([0.0, value]), np.array([0.0, 0.1]))


@pytest.mark.parametrize("turnover", [-0.01, -1.0])
def test_negative_turnover_is_rejected(turnover: float) -> None:
    with pytest.raises(ValueError, match="negative"):
        compute_validation_composite_score(1.0, 1.0, turnover)
    with pytest.raises(ValueError, match="negative"):
        compute_legacy_composite_score_v2(1.0, 1.0, 0.2, turnover)


def test_turnover_above_one_is_capped_and_missing_components_are_explicit() -> None:
    assert compute_validation_composite_score(None, None, 2.0) == -0.15
    assert compute_validation_composite_score(0.0, 0.0, None) == -0.03
    assert compute_legacy_composite_score_v2(None, None, None, 2.0) == 0.0
    assert compute_legacy_composite_score_v2(None, None, None, None) == 0.08


@pytest.mark.parametrize(
    ("formula", "components", "expected"),
    [
        (
            VALIDATION_COMPOSITE_VERSION,
            {
                "validation_after_cost_sharpe": 1.25,
                "validation_fitness": 0.75,
                "validation_turnover": 0.2,
            },
            0.8575,
        ),
        (
            LEGACY_COMPOSITE_VERSION,
            {"after_cost_sharpe": 1.25, "fitness": 0.75, "sharpe_decay": 0.2, "turnover": 0.0},
            0.865,
        ),
        (
            LEGACY_COMPOSITE_CURRENT_VERSION,
            {"after_cost_sharpe": 1.25, "fitness": 0.75, "sharpe_decay": 0.2, "turnover": 0.0},
            0.885,
        ),
    ],
)
def test_persisted_spec_recomputes_exactly(
    formula: str, components: dict[str, float | None], expected: float
) -> None:
    spec = composite_score_spec(formula_id=formula, components=components)
    assert spec["formula_id"] == formula
    assert recompute_composite_score(spec) == expected


def test_evaluation_input_series_reject_non_finite_and_negative_turnover() -> None:
    from datetime import date, timedelta

    dates = [date(2025, 1, 1) + timedelta(days=index) for index in range(60)]
    pnl = [float(index % 3) for index in range(60)]
    turnover = [0.1] * 60
    for invalid in (math.nan, math.inf, -math.inf):
        broken = pnl.copy()
        broken[3] = invalid
        with pytest.raises(ValueError, match="finite"):
            evaluate_pnl_series(1, dates, broken, turnover)
    turnover[3] = -0.1
    with pytest.raises(ValueError, match="negative"):
        evaluate_pnl_series(1, dates, pnl, turnover)


def test_sqlite_migration_preserves_historical_score_as_unknown(tmp_path: Path) -> None:
    async def _run() -> None:
        db_path = tmp_path / "score-migration.sqlite"
        db = Database(db_path)
        await db.create_all()
        async with db.session() as session:
            session.add(
                ResearchEvaluation(
                    id=1,
                    composite_score=0.4321,
                    val_composite_score=0.5432,
                    evaluation_scheme="LEGACY_2WAY",
                )
            )
        await db.dispose()

        with sqlite3.connect(db_path) as connection:
            connection.execute("ALTER TABLE research_evaluation DROP COLUMN composite_score_spec")

        migrated = Database(db_path)
        await migrated.create_all()
        async with migrated.session() as session:
            row = await session.get(ResearchEvaluation, 1)
            assert row is not None
            assert row.composite_score == 0.4321
            assert row.val_composite_score == 0.5432
            assert row.composite_score_spec is None
        await migrated.dispose()

    asyncio.run(_run())


def test_formula_specs_and_historical_scores_survive_database_reopen(tmp_path: Path) -> None:
    async def _run() -> None:
        db_path = tmp_path / "score-reopen.sqlite"
        db = Database(db_path)
        await db.create_all()
        legacy_parts = {
            "after_cost_sharpe": None,
            "fitness": 0.0,
            "sharpe_decay": None,
            "turnover": 0.0,
        }
        legacy_spec = composite_score_spec(
            formula_id=LEGACY_COMPOSITE_CURRENT_VERSION, components=legacy_parts
        )
        legacy_score = recompute_composite_score(legacy_spec)
        validation_parts = {
            "validation_after_cost_sharpe": 0.0,
            "validation_fitness": None,
            "validation_turnover": None,
        }
        validation_spec = composite_score_spec(
            formula_id=VALIDATION_COMPOSITE_VERSION, components=validation_parts
        )
        validation_score = recompute_composite_score(validation_spec)
        async with db.session() as session:
            session.add_all(
                [
                    ResearchEvaluation(
                        id=1,
                        composite_score=legacy_score,
                        composite_score_spec=legacy_spec,
                        evaluation_scheme="LEGACY_2WAY",
                    ),
                    ResearchEvaluation(
                        id=2,
                        composite_score=validation_score,
                        val_composite_score=validation_score,
                        composite_score_spec=validation_spec,
                        evaluation_scheme="THREE_WAY_CHRONOLOGICAL",
                    ),
                    ResearchEvaluation(
                        id=3,
                        composite_score=0.4321,
                        evaluation_scheme="LEGACY_2WAY",
                    ),
                ]
            )
        await db.dispose()

        reopened = Database(db_path)
        await reopened.create_all()
        async with reopened.session() as session:
            legacy = await session.get(ResearchEvaluation, 1)
            validation = await session.get(ResearchEvaluation, 2)
            historical = await session.get(ResearchEvaluation, 3)
            assert legacy is not None and legacy.composite_score_spec == legacy_spec
            assert legacy.composite_score == recompute_composite_score(legacy.composite_score_spec)
            assert validation is not None and validation.composite_score_spec == validation_spec
            assert validation.composite_score == validation.val_composite_score
            assert validation.val_composite_score == recompute_composite_score(
                validation.composite_score_spec
            )
            assert historical is not None
            assert historical.composite_score == 0.4321
            assert historical.composite_score_spec is None
        await reopened.dispose()

    asyncio.run(_run())


def test_api_keeps_existing_score_field_and_exposes_formula_provenance() -> None:
    payload = CandidateOut(
        id=42,
        composite_score=0.125,
        composite_score_spec={"formula_id": VALIDATION_COMPOSITE_VERSION},
    ).model_dump(by_alias=True)
    assert payload["compositeScore"] == 0.125
    assert payload["compositeScoreSpec"]["formula_id"] == VALIDATION_COMPOSITE_VERSION

    quarantined = CandidateOut(id=43, composite_score=math.inf)
    assert quarantined.composite_score is None


def test_portfolio_ties_are_candidate_id_stable_and_non_finite_scores_rejected() -> None:
    optimizer = AlphaPoolOptimizer(None)  # Selection helper does not access the DB.
    series = {1: [0.0, 1.0, 0.0], 2: [1.0, 0.0, 1.0]}
    assert optimizer.select_uncorrelated_pool(
        [2, 1], series, {1: 1.0, 2: 1.0}, max_correlation=1.0, target_pool_size=2
    ) == [1, 2]
    with pytest.raises(ValueError, match="non-finite"):
        optimizer.select_uncorrelated_pool([1], series, {1: math.nan})


def test_persisted_api_leaderboard_and_memory_quarantine_invalid_metrics(
    tmp_path: Path,
) -> None:
    async def _run() -> None:
        db = Database(tmp_path / "consumer-quarantine.sqlite")
        await db.create_all()
        async with db.session() as session:
            research_session = ResearchSession(name="score consumers")
            session.add(research_session)
            await session.flush()
            candidates: dict[str, ResearchCandidate] = {}
            for name, score, full_sharpe, after_cost in (
                ("tie-a", 1.0, 1.2, 0.8),
                ("tie-b", 1.0, 1.1, 0.7),
                ("missing", None, 0.9, 0.5),
                ("invalid", 2.0, math.inf, math.inf),
                ("invalid-score", math.inf, 1.0, 0.5),
            ):
                candidate = ResearchCandidate(
                    session_id=research_session.id,
                    expression=f"rank({name})",
                    status=CandidateStatus.EVALUATED,
                    is_valid=True,
                )
                session.add(candidate)
                await session.flush()
                candidates[name] = candidate
                session.add(
                    ResearchEvaluation(
                        candidate_id=candidate.id,
                        is_verified=True,
                        provenance="BRAIN_VAULT",
                        evaluation_scheme="LEGACY_2WAY",
                        composite_score=score,
                        composite_score_spec=(
                            {"formula_id": LEGACY_COMPOSITE_CURRENT_VERSION}
                            if name == "tie-a"
                            else None
                        ),
                        full_sharpe=full_sharpe,
                        full_fitness=0.8,
                        after_cost_sharpe=after_cost,
                        oos_sharpe=0.6,
                        sharpe_decay=0.1,
                        is_turnover=0.2,
                            checks_passed=0,
                            checks_total=0,
                    )
                )

        leaderboard = LeaderboardService(db)
        ranks = await leaderboard.get_alpha_leaderboard(session_id=research_session.id)
        assert [row.candidate_id for row in ranks] == [
            candidates["tie-a"].id,
            candidates["tie-b"].id,
            candidates["missing"].id,
        ]
        assert ranks[0].composite_score_spec["formula_id"] == LEGACY_COMPOSITE_CURRENT_VERSION
        assert ranks[1].composite_score_spec is None
        assert ranks[2].composite_score == 0.0

        memory = ResearchMemory(db)
        records = await memory.get_eligible_candidates(session_id=research_session.id)
        assert [row["candidate_id"] for row in records] == [
            candidates["tie-a"].id,
            candidates["tie-b"].id,
            candidates["missing"].id,
        ]
        assert records[0]["composite_score_spec"]["formula_id"] == LEGACY_COMPOSITE_CURRENT_VERSION
        assert records[1]["composite_score_spec"] is None

        state = SimpleNamespace(db=db, leaderboard=leaderboard)
        candidate_payloads = await list_candidates(session_id=research_session.id, state=state)
        invalid_payload = next(row for row in candidate_payloads if row.id == candidates["invalid"].id)
        assert invalid_payload.full_sharpe is None
        assert invalid_payload.after_cost_sharpe is None
        assert invalid_payload.composite_score == 2.0
        invalid_score_payload = next(
            row for row in candidate_payloads if row.id == candidates["invalid-score"].id
        )
        assert invalid_score_payload.composite_score is None
        assert next(row for row in candidate_payloads if row.id == candidates["missing"].id).composite_score_spec is None

        evaluation_payloads = await list_evaluations(session_id=research_session.id, state=state)
        invalid_evaluation = next(
            row for row in evaluation_payloads if row["candidateId"] == candidates["invalid"].id
        )
        assert invalid_evaluation["fullSharpe"] is None
        assert invalid_evaluation["afterCostSharpe"] is None
        assert invalid_evaluation["compositeScoreSpec"] is None
        invalid_score_evaluation = next(
            row for row in evaluation_payloads if row["candidateId"] == candidates["invalid-score"].id
        )
        assert invalid_score_evaluation["compositeScore"] is None
        json.dumps(evaluation_payloads, allow_nan=False)

        alpha_payloads = await get_alpha_leaderboard(state=state, session_id=research_session.id)
        assert [row["candidateId"] for row in alpha_payloads] == [
            candidates["tie-a"].id,
            candidates["tie-b"].id,
            candidates["missing"].id,
        ]
        assert alpha_payloads[1]["compositeScoreSpec"] is None
        await db.dispose()

    asyncio.run(_run())


@pytest.mark.parametrize(
    ("candidate", "pnl", "turnover", "message"),
    [
        ({"id": 1, "composite_score": math.inf}, [0.0, 1.0], [0.1, 0.1], "composite_score"),
        ({"id": 1, "composite_score": 1.0, "sharpe": math.nan}, [0.0, 1.0], [0.1, 0.1], "sharpe"),
        ({"id": 1, "composite_score": 1.0, "turnover": -0.1}, [0.0, 1.0], [0.1, 0.1], "negative turnover"),
        ({"id": 1, "composite_score": 1.0}, [0.0, math.inf], [0.1, 0.1], "non-finite portfolio observations"),
        ({"id": 1, "composite_score": 1.0}, [0.0, 1.0], [0.1, math.nan], "non-finite portfolio observations"),
    ],
)
def test_portfolio_rejects_invalid_scores_and_metric_inputs(
    candidate: dict[str, object],
    pnl: list[float],
    turnover: list[float],
    message: str,
) -> None:
    optimizer = AlphaPoolOptimizer(None)
    with pytest.raises(ValueError, match=message):
        optimizer.curate_pool(
            [candidate],
            {1: pnl},
            {1: turnover},
            target_size=1,
        )


def test_holdout_mutation_does_not_change_p2_06_consumers(tmp_path: Path) -> None:
    async def _run() -> None:
        dates = [date(2025, 1, 1) + timedelta(days=index) for index in range(100)]
        pnl = [0.01 * (1 + index % 7) for index in range(100)]
        turnover = [0.1] * 100
        baseline = evaluate_pnl_series(1, dates, pnl, turnover)
        mutated_pnl = pnl.copy()
        mutated_pnl[80:] = [-100.0 - index for index in range(20)]
        mutated = evaluate_pnl_series(1, dates, mutated_pnl, turnover)
        assert baseline.val_composite_score == mutated.val_composite_score
        assert baseline.composite_score_spec == mutated.composite_score_spec

        db = Database(tmp_path / "holdout-score-consumers.sqlite")
        await db.create_all()
        async with db.session() as session:
            research_session = ResearchSession(name="holdout score invariance")
            session.add(research_session)
            await session.flush()
            candidate_ids: list[int] = []
            for index in range(2):
                candidate = ResearchCandidate(
                    session_id=research_session.id,
                    expression=f"rank(field_{index})",
                    status=CandidateStatus.GRADUATED,
                    is_valid=True,
                )
                session.add(candidate)
                await session.flush()
                candidate_ids.append(candidate.id)
                session.add(
                    ResearchEvaluation(
                        candidate_id=candidate.id,
                        is_verified=True,
                        provenance="BRAIN_VAULT",
                        evaluation_scheme="THREE_WAY_CHRONOLOGICAL",
                        composite_score=baseline.composite_score,
                        val_composite_score=baseline.val_composite_score,
                        composite_score_spec=baseline.composite_score_spec,
                        full_sharpe=baseline.full_sharpe,
                        val_sharpe=baseline.val_sharpe,
                        val_fitness=baseline.val_fitness,
                        val_turnover=baseline.val_turnover,
                        val_after_cost_sharpe=baseline.val_after_cost_sharpe,
                        is_val_decay=baseline.is_val_decay,
                        holdout_sharpe=5.0 + index,
                        holdout_fitness=4.0 + index,
                        holdout_turnover=0.1 + index,
                        checks_passed=0,
                        checks_total=0,
                    )
                )

        leaderboard = LeaderboardService(db)
        memory = ResearchMemory(db)
        optimizer = AlphaPoolOptimizer(None)
        state = SimpleNamespace(db=db, leaderboard=leaderboard)
        pnl_map = {
            candidate_ids[0]: [0.0, 0.01, 0.0, -0.01],
            candidate_ids[1]: [0.01, 0.0, -0.01, 0.0],
        }
        turnover_map = {candidate_id: [0.1] * 4 for candidate_id in candidate_ids}

        async def snapshot() -> tuple[object, ...]:
            candidate_rows = await list_candidates(session_id=research_session.id, state=state)
            evaluation_rows = await list_evaluations(session_id=research_session.id, state=state)
            ranks = await leaderboard.get_alpha_leaderboard(session_id=research_session.id)
            records = await memory.get_eligible_candidates(session_id=research_session.id)
            pool = optimizer.curate_pool(
                [
                    {
                        "id": row["candidate_id"],
                        "composite_score": row["composite_score"],
                        "composite_score_spec": row["composite_score_spec"],
                        "sharpe": row["sharpe"],
                        "fitness": row["fitness"],
                        "turnover": row["turnover"],
                    }
                    for row in records
                ],
                pnl_map,
                turnover_map,
                max_pairwise_corr=1.0,
                target_size=2,
            )
            return (
                tuple((row.id, row.composite_score, row.composite_score_spec) for row in candidate_rows),
                tuple(
                    (row["candidateId"], row["compositeScore"], row["compositeScoreSpec"])
                    for row in evaluation_rows
                ),
                tuple((row.candidate_id, row.composite_score, row.composite_score_spec) for row in ranks),
                tuple(
                    (row["candidate_id"], row["composite_score"], row["composite_score_spec"])
                    for row in records
                ),
                tuple((member.candidate_id, member.composite_score, member.composite_score_spec) for member in pool.members),
            )

        before = await snapshot()
        async with db.session() as session:
            for candidate_id in candidate_ids:
                evaluation = await session.scalar(
                    select(ResearchEvaluation).where(
                        ResearchEvaluation.candidate_id == candidate_id
                    )
                )
                assert evaluation is not None
                evaluation.holdout_sharpe = -999.0
                evaluation.holdout_fitness = 999.0
                evaluation.holdout_turnover = 99.0
        after = await snapshot()
        assert after == before
        await db.dispose()

    asyncio.run(_run())
