"""Regression test suite for P0-04: Genuine Robustness Perturbation Simulations."""

from __future__ import annotations

import asyncio
import json
import sqlite3
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from sqlalchemy import select, text

from alpha_harness.brain.errors import BrainValidationError
from alpha_harness.brain.schemas import SimulationRequest, SimulationSettings, SimulationType
from alpha_harness.db.duck import Catalog
from alpha_harness.db.models import (
    CandidateStatus,
    ResearchCandidate,
    ResearchEvaluation,
    ResearchRobustness,
    ResearchSession,
    ResearchStatus,
    SimStatus,
    utcnow,
)
from alpha_harness.db.sqlite import Database
from alpha_harness.evaluation.robustness import (
    RobustnessEngine,
    RobustnessTestPlan,
    RobustnessTestResult,
)
from alpha_harness.vault.store import AlphaVault


def generate_series_rows(
    num_days: int = 100,
    base_pnl: float = 1000.0,
    turnover: float = 0.15,
    start_date: date | None = None,
):
    start = start_date or date(2023, 1, 2)
    pnl_rows = []
    to_rows = []
    cur = start
    cumulative_pnl = 0.0
    for i in range(num_days):
        cumulative_pnl += base_pnl + (i % 5) * 10.0
        pnl_rows.append({"date": cur, "pnl": cumulative_pnl})
        to_rows.append({"date": cur, "turnover": turnover})
        cur += timedelta(days=1)
    return pnl_rows, to_rows


class MockBrainResponse:
    def __init__(
        self,
        status: int = 201,
        location: str = "/simulations/sim_mock_001",
        body: Any = None,
    ):
        self.status = status
        self.location = location
        self.body = body or {}
        self.retry_after = None


class MockBrainEndpoints:
    def __init__(
        self,
        fail_create: bool = False,
        fail_sim: bool = False,
        sim_id: str = "sim_mock_001",
        alpha_id: str = "alpha_perturbed_001",
        pnl_rows: list[dict[str, Any]] | None = None,
        to_rows: list[dict[str, Any]] | None = None,
    ):
        self.fail_create = fail_create
        self.fail_sim = fail_sim
        self.sim_id = sim_id
        self.alpha_id = alpha_id
        self.pnl_rows = pnl_rows or []
        self.to_rows = to_rows or []
        self.submitted_requests: list[SimulationRequest] = []
        self.client = object()

    async def create_simulation(self, request: SimulationRequest) -> MockBrainResponse:
        self.submitted_requests.append(request)
        if self.fail_create:
            raise BrainValidationError(
                "Mock validation error",
                fields={"settings": {"neutralization": ["Invalid setting on platform"]}},
            )
        return MockBrainResponse(status=201, location=f"/simulations/{self.sim_id}")

    async def read_simulation(self, simulation_id: str) -> MockBrainResponse:
        if self.fail_sim:
            return MockBrainResponse(
                status=200,
                body={"status": "ERROR", "message": "Simulation failed on compute node"},
            )
        return MockBrainResponse(
            status=200,
            body={"status": "COMPLETE", "alpha": self.alpha_id},
        )

    async def get_recordset(self, alpha_id: str, name: str) -> Any:
        class FakeRecordSet:
            def __init__(self, rows_data):
                self._rows = rows_data

            def rows(self):
                return self._rows

        if name == "pnl":
            return FakeRecordSet(self.pnl_rows)
        if name == "turnover":
            return FakeRecordSet(self.to_rows)
        if name == "yearly-stats":
            return FakeRecordSet([])
        return FakeRecordSet([])


async def setup_test_environment(tmp_path: Path):
    db = Database(tmp_path / "app.db")
    await db.create_all()

    cat = Catalog(tmp_path / "catalog.duckdb")
    await cat.open()
    vault = AlphaVault(cat)

    async with db.session() as session:
        sess = ResearchSession(
            name="P0-04 Test Session",
            prompt="Robustness Verification",
            status=ResearchStatus.RUNNING,
        )
        session.add(sess)
        await session.flush()

        cand = ResearchCandidate(
            session_id=sess.id,
            alpha_id="alpha_base_001",
            expression="rank(ts_decay_linear(volume, 10))",
            status=CandidateStatus.EVALUATED,
            sim_region="USA",
            sim_universe="TOP3000",
            sim_delay=1,
            sim_decay=4,
            sim_neutralization="INDUSTRY",
            language="FASTEXPR",
        )
        session.add(cand)
        await session.flush()

        # Dates: 60 IS, 20 Val (2023-03-03 to 2023-03-23), 20 Holdout
        start_d = date(2023, 1, 2)
        is_end = start_d + timedelta(days=60)
        val_end = is_end + timedelta(days=20)

        ev = ResearchEvaluation(
            candidate_id=cand.id,
            evaluation_scheme="SPLIT_3WAY_FINAL",
            is_end_date=is_end,
            val_end_date=val_end,
            val_sharpe=1.85,
            val_fitness=1.40,
            full_sharpe=1.75,
            full_fitness=1.30,
        )
        session.add(ev)
        await session.commit()
        cand_id = cand.id
        ev_id = ev.id

    return db, cat, vault, cand_id, ev_id, is_end, val_end


def test_no_random_degradation_metrics(tmp_path: Path):
    """P0-04 Req 1: Robustness path deterministically calculates metrics from genuine simulation data."""
    async def _run():
        db, cat, vault, cand_id, ev_id, is_end, val_end = await setup_test_environment(tmp_path)
        try:
            pnl_rows, to_rows = generate_series_rows(num_days=100, base_pnl=500.0)
            mock_ep = MockBrainEndpoints(
                sim_id="sim_neut_01",
                alpha_id="alpha_neut_01",
                pnl_rows=pnl_rows,
                to_rows=to_rows,
            )
            engine = RobustnessEngine(db, vault=vault, endpoints=mock_ep)

            plan = RobustnessTestPlan(
                candidate_id=cand_id,
                test_type="neutralization",
                test_value="SECTOR",
                perturbed_settings={"neutralization": "SECTOR"},
            )

            res1 = await engine.execute_perturbation(cand_id, plan, baseline_eval_id=ev_id)
            res2 = await engine.execute_perturbation(cand_id, plan, baseline_eval_id=ev_id)

            assert res1.status == "COMPLETED"
            assert res2.status == "COMPLETED"
            assert res1.is_synthetic is False
            assert res2.is_synthetic is False
            assert res1.result_kind == "GENUINE"
            assert res2.result_kind == "GENUINE"

            # Must be completely deterministic, identical, and reflect genuine series
            assert res1.perturbed_sharpe == res2.perturbed_sharpe
            assert res1.perturbed_fitness == res2.perturbed_fitness
            assert res1.sharpe_delta == res2.sharpe_delta
            assert res1.perturbed_sharpe is not None
            assert res1.brain_simulation_id == "sim_neut_01"
            assert res1.perturbed_alpha_id == "alpha_neut_01"
        finally:
            await cat.close()
            await db.dispose()

    asyncio.run(_run())


def test_perturbation_submits_correct_parameter_diff(tmp_path: Path):
    """P0-04 Req 2: Submits simulation request varying ONLY the intended parameter."""
    async def _run():
        db, cat, vault, cand_id, ev_id, _, _ = await setup_test_environment(tmp_path)
        try:
            pnl_rows, to_rows = generate_series_rows(num_days=100)
            mock_ep = MockBrainEndpoints(pnl_rows=pnl_rows, to_rows=to_rows)
            engine = RobustnessEngine(db, vault=vault, endpoints=mock_ep)

            # Test 1: Neutralization perturbation (INDUSTRY -> SECTOR)
            plan_neut = RobustnessTestPlan(
                candidate_id=cand_id,
                test_type="neutralization",
                test_value="SECTOR",
                perturbed_settings={"neutralization": "SECTOR"},
            )
            await engine.execute_perturbation(cand_id, plan_neut, baseline_eval_id=ev_id)

            req_neut = mock_ep.submitted_requests[-1]
            assert req_neut.settings.neutralization == "SECTOR"
            assert req_neut.settings.decay == 4  # Untouched
            assert req_neut.settings.universe == "TOP3000"  # Untouched
            assert req_neut.settings.region == "USA"  # Untouched
            assert req_neut.regular == "rank(ts_decay_linear(volume, 10))"

            # Test 2: Decay perturbation (4 -> 8)
            plan_decay = RobustnessTestPlan(
                candidate_id=cand_id,
                test_type="decay",
                test_value="8",
                perturbed_settings={"decay": 8},
            )
            await engine.execute_perturbation(cand_id, plan_decay, baseline_eval_id=ev_id)

            req_decay = mock_ep.submitted_requests[-1]
            assert req_decay.settings.decay == 8
            assert req_decay.settings.neutralization == "INDUSTRY"  # Untouched
            assert req_decay.settings.universe == "TOP3000"  # Untouched
            assert req_decay.settings.region == "USA"  # Untouched

            # Test 3: Universe perturbation (TOP3000 -> TOP1000)
            plan_univ = RobustnessTestPlan(
                candidate_id=cand_id,
                test_type="universe",
                test_value="TOP1000",
                perturbed_settings={"universe": "TOP1000"},
            )
            await engine.execute_perturbation(cand_id, plan_univ, baseline_eval_id=ev_id)

            req_univ = mock_ep.submitted_requests[-1]
            assert req_univ.settings.universe == "TOP1000"
            assert req_univ.settings.decay == 4  # Untouched
            assert req_univ.settings.neutralization == "INDUSTRY"  # Untouched
        finally:
            await cat.close()
            await db.dispose()

    asyncio.run(_run())


def test_completed_results_retain_simulation_id_and_provenance(tmp_path: Path):
    """P0-04 Req 3 & 10: Retains genuine simulation ID, baseline evaluation ID, and provenance."""
    async def _run():
        db, cat, vault, cand_id, ev_id, is_end, val_end = await setup_test_environment(tmp_path)
        try:
            pnl_rows, to_rows = generate_series_rows(num_days=100)
            mock_ep = MockBrainEndpoints(
                sim_id="sim_prov_999",
                alpha_id="alpha_prov_999",
                pnl_rows=pnl_rows,
                to_rows=to_rows,
            )
            engine = RobustnessEngine(db, vault=vault, endpoints=mock_ep)

            plan = RobustnessTestPlan(
                candidate_id=cand_id,
                test_type="neutralization",
                test_value="MARKET",
                perturbed_settings={"neutralization": "MARKET"},
            )
            res = await engine.execute_perturbation(cand_id, plan, baseline_eval_id=ev_id)

            assert res.status == "COMPLETED"
            assert res.brain_simulation_id == "sim_prov_999"
            assert res.perturbed_alpha_id == "alpha_prov_999"
            assert res.baseline_eval_id == ev_id
            assert res.val_start_date == is_end
            assert res.val_end_date == val_end
            assert res.perturbation_config is not None
            assert res.perturbation_config["parameter_changed"] == "neutralization"
            assert res.perturbation_config["perturbed_value"] == "MARKET"

            # Verify persisted row in SQLite
            async with db.session() as session:
                rows = (
                    await session.scalars(
                        select(ResearchRobustness).where(
                            ResearchRobustness.brain_simulation_id == "sim_prov_999"
                        )
                    )
                ).all()
                assert len(rows) == 1
                row = rows[0]
                assert row.candidate_id == cand_id
                assert row.baseline_eval_id == ev_id
                assert row.status == "COMPLETED"
                assert row.result_kind == "GENUINE"
                assert row.is_synthetic is False
                assert row.val_start_date == is_end
                assert row.val_end_date == val_end
        finally:
            await cat.close()
            await db.dispose()

    asyncio.run(_run())


def test_perturbed_performance_calculated_using_baseline_validation_window(tmp_path: Path):
    """P0-04 Req 4: Perturbed metrics calculated over identical baseline validation dates."""
    async def _run():
        db, cat, vault, cand_id, ev_id, is_end, val_end = await setup_test_environment(tmp_path)
        try:
            # Construct series where IS has zero pnl, Val has positive pnl, and Holdout has zero pnl
            pnl_rows = []
            to_rows = []
            cur = date(2023, 1, 2)
            cum = 0.0
            for i in range(100):
                if is_end <= cur < val_end:
                    # Pure validation period gets distinct return with variance
                    cum += 5000.0 + (i % 5) * 100.0
                pnl_rows.append({"date": cur, "pnl": cum})
                to_rows.append({"date": cur, "turnover": 0.10})
                cur += timedelta(days=1)

            mock_ep = MockBrainEndpoints(
                sim_id="sim_val_window",
                alpha_id="alpha_val_window",
                pnl_rows=pnl_rows,
                to_rows=to_rows,
            )
            engine = RobustnessEngine(db, vault=vault, endpoints=mock_ep)

            plan = RobustnessTestPlan(
                candidate_id=cand_id,
                test_type="decay",
                test_value="6",
                perturbed_settings={"decay": 6},
            )
            res = await engine.execute_perturbation(cand_id, plan, baseline_eval_id=ev_id)

            assert res.status == "COMPLETED"
            # Validation period had strictly positive cumulative gains -> high positive Sharpe
            assert res.perturbed_sharpe is not None
            assert res.perturbed_sharpe > 0.8
            assert res.val_start_date == is_end
            assert res.val_end_date == val_end
        finally:
            await cat.close()
            await db.dispose()

    asyncio.run(_run())


def test_missing_credentials_or_unavailable_brain_produces_not_run(tmp_path: Path):
    """P0-04 Req 5: Missing credentials / unavailable BRAIN produces NOT_RUN without metrics."""
    async def _run():
        db, cat, vault, cand_id, ev_id, _, _ = await setup_test_environment(tmp_path)
        try:
            # Endpoints is None -> execution unavailable
            engine = RobustnessEngine(db, vault=vault, endpoints=None)

            plan = RobustnessTestPlan(
                candidate_id=cand_id,
                test_type="neutralization",
                test_value="SECTOR",
                perturbed_settings={"neutralization": "SECTOR"},
            )
            res = await engine.execute_perturbation(cand_id, plan, baseline_eval_id=ev_id)

            assert res.status == "NOT_RUN"
            assert res.result_kind == "GENUINE"
            assert res.is_synthetic is False
            assert res.passed is False
            assert res.perturbed_sharpe is None
            assert res.perturbed_fitness is None
            assert res.sharpe_delta is None
            assert "unavailable" in res.message.lower()

            # Verify summary treats candidate as untested
            summary = await engine.get_candidate_robustness_summary(cand_id)
            assert summary["tested"] is False
            assert summary["total_tests"] == 0
            assert summary["all_passed"] is False
        finally:
            await cat.close()
            await db.dispose()

    asyncio.run(_run())


def test_invalid_or_unsupported_perturbations_produce_unsupported(tmp_path: Path):
    """P0-04 Req 6: Invalid or unsupported perturbation requests produce UNSUPPORTED."""
    async def _run():
        db, cat, vault, cand_id, ev_id, _, _ = await setup_test_environment(tmp_path)
        try:
            pnl_rows, to_rows = generate_series_rows(num_days=100)
            mock_ep = MockBrainEndpoints(pnl_rows=pnl_rows, to_rows=to_rows)
            engine = RobustnessEngine(db, vault=vault, endpoints=mock_ep)

            # 1. Invalid neutralization
            plan_bad_neut = RobustnessTestPlan(
                candidate_id=cand_id,
                test_type="neutralization",
                test_value="INVALID_SECTOR_999",
                perturbed_settings={"neutralization": "INVALID_SECTOR_999"},
            )
            res1 = await engine.execute_perturbation(cand_id, plan_bad_neut, baseline_eval_id=ev_id)
            assert res1.status == "UNSUPPORTED"
            assert res1.passed is False
            assert res1.perturbed_sharpe is None
            assert "unsupported" in res1.message.lower()

            # 2. Invalid decay
            plan_bad_decay = RobustnessTestPlan(
                candidate_id=cand_id,
                test_type="decay",
                test_value="-10",
                perturbed_settings={"decay": -10},
            )
            res2 = await engine.execute_perturbation(cand_id, plan_bad_decay, baseline_eval_id=ev_id)
            assert res2.status == "UNSUPPORTED"
            assert res2.passed is False
            assert res2.perturbed_sharpe is None

            # 3. BRAIN API rejects settings on platform
            mock_ep.fail_create = True
            plan_valid = RobustnessTestPlan(
                candidate_id=cand_id,
                test_type="neutralization",
                test_value="SECTOR",
                perturbed_settings={"neutralization": "SECTOR"},
            )
            res3 = await engine.execute_perturbation(cand_id, plan_valid, baseline_eval_id=ev_id)
            assert res3.status == "UNSUPPORTED"
            assert res3.passed is False
            assert "rejected" in res3.message.lower()
        finally:
            await cat.close()
            await db.dispose()

    asyncio.run(_run())


def test_failed_simulation_produces_failed_with_explanation(tmp_path: Path):
    """P0-04 Req 7: Submitted simulation failure produces FAILED with explanation."""
    async def _run():
        db, cat, vault, cand_id, ev_id, _, _ = await setup_test_environment(tmp_path)
        try:
            mock_ep = MockBrainEndpoints(fail_sim=True, sim_id="sim_fail_123")
            engine = RobustnessEngine(db, vault=vault, endpoints=mock_ep)

            plan = RobustnessTestPlan(
                candidate_id=cand_id,
                test_type="universe",
                test_value="TOP1000",
                perturbed_settings={"universe": "TOP1000"},
            )
            res = await engine.execute_perturbation(cand_id, plan, baseline_eval_id=ev_id)

            assert res.status == "FAILED"
            assert res.passed is False
            assert res.perturbed_sharpe is None
            assert res.brain_simulation_id == "sim_fail_123"
            assert res.error_details is not None
            assert "failed on compute node" in res.error_details.lower()
        finally:
            await cat.close()
            await db.dispose()

    asyncio.run(_run())


def test_legacy_synthetic_robustness_records_excluded_from_genuine_statistics(tmp_path: Path):
    """P0-04 Req 8 & 9: Legacy synthetic records excluded from genuine robustness statistics."""
    async def _run():
        db, cat, vault, cand_id, ev_id, _, _ = await setup_test_environment(tmp_path)
        try:
            # Seed legacy synthetic records directly into the database
            async with db.session() as session:
                r1 = ResearchRobustness(
                    candidate_id=cand_id,
                    test_type="neutralization",
                    test_value="SECTOR",
                    perturbed_sharpe=2.15,
                    perturbed_fitness=1.80,
                    sharpe_delta=0.30,
                    passed=True,
                    message="Passed legacy synthetic test",
                    status="COMPLETED",
                    result_kind="LEGACY_SYNTHETIC",
                    is_synthetic=True,
                    brain_simulation_id=None,
                )
                r2 = ResearchRobustness(
                    candidate_id=cand_id,
                    test_type="decay",
                    test_value="6",
                    perturbed_sharpe=1.95,
                    perturbed_fitness=1.60,
                    sharpe_delta=0.10,
                    passed=True,
                    message="Passed legacy synthetic test",
                    status="COMPLETED",
                    result_kind="LEGACY_SYNTHETIC",
                    is_synthetic=True,
                    brain_simulation_id=None,
                )
                session.add_all([r1, r2])
                await session.commit()

            engine = RobustnessEngine(db, vault=vault)
            summary = await engine.get_candidate_robustness_summary(cand_id)

            # Legacy synthetic records must NOT count as genuine tested evidence
            assert summary["tested"] is False
            assert summary["total_tests"] == 0
            assert summary["passed_tests"] == 0
            assert summary["pass_rate"] == 0.0
            assert summary["all_passed"] is False
            assert summary["legacy_synthetic_count"] == 2

            # Historical records are preserved for audit
            assert len(summary["tests"]) == 2
            assert all(t["is_synthetic"] is True for t in summary["tests"])
            assert all(t["result_kind"] == "LEGACY_SYNTHETIC" for t in summary["tests"])

            # Add ONE genuine completed test
            pnl_rows, to_rows = generate_series_rows(num_days=100)
            mock_ep = MockBrainEndpoints(
                sim_id="sim_genuine_01",
                alpha_id="alpha_genuine_01",
                pnl_rows=pnl_rows,
                to_rows=to_rows,
            )
            engine.endpoints = mock_ep
            plan = RobustnessTestPlan(
                candidate_id=cand_id,
                test_type="neutralization",
                test_value="MARKET",
                perturbed_settings={"neutralization": "MARKET"},
            )
            await engine.execute_perturbation(cand_id, plan, baseline_eval_id=ev_id)

            # Summary now includes ONLY the genuine test
            summary2 = await engine.get_candidate_robustness_summary(cand_id)
            assert summary2["tested"] is True
            assert summary2["total_tests"] == 1
            assert summary2["passed_tests"] == 1
            assert summary2["pass_rate"] == 1.0
            assert summary2["all_passed"] is True
            assert summary2["legacy_synthetic_count"] == 2
        finally:
            await cat.close()
            await db.dispose()

    asyncio.run(_run())


def test_incomplete_or_synthetic_results_cannot_masquerade_as_completed(tmp_path: Path):
    """P0-04 Req 9: Incomplete, failed, or synthetic results cannot masquerade as completed genuine evaluations."""
    async def _run():
        db, cat, vault, cand_id, _, _, _ = await setup_test_environment(tmp_path)
        try:
            async with db.session() as session:
                records = [
                    ResearchRobustness(
                        candidate_id=cand_id,
                        test_type="neutralization",
                        test_value="SECTOR",
                        status="NOT_RUN",
                        result_kind="GENUINE",
                        is_synthetic=False,
                        passed=False,
                    ),
                    ResearchRobustness(
                        candidate_id=cand_id,
                        test_type="decay",
                        test_value="999",
                        status="UNSUPPORTED",
                        result_kind="GENUINE",
                        is_synthetic=False,
                        passed=False,
                    ),
                    ResearchRobustness(
                        candidate_id=cand_id,
                        test_type="universe",
                        test_value="TOP1000",
                        status="FAILED",
                        result_kind="GENUINE",
                        is_synthetic=False,
                        passed=False,
                    ),
                ]
                session.add_all(records)
                await session.commit()

            engine = RobustnessEngine(db, vault=vault)
            summary = await engine.get_candidate_robustness_summary(cand_id)

            assert summary["tested"] is False
            assert summary["total_tests"] == 0
            assert summary["passed_tests"] == 0
            assert summary["pass_rate"] == 0.0
            assert summary["all_passed"] is False
        finally:
            await cat.close()
            await db.dispose()

    asyncio.run(_run())


def test_additive_migration_preserves_existing_records(tmp_path: Path):
    """P0-04 Req 11: Additive SQLite migration preserves existing records and marks legacy synthetic records."""
    db_path = tmp_path / "legacy.db"

    # 1. Create a pre-P0-04 schema table directly in SQLite
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE research_robustness (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            candidate_id INTEGER,
            test_type TEXT,
            test_value TEXT,
            simulation_record_id INTEGER,
            perturbed_sharpe REAL,
            perturbed_fitness REAL,
            perturbed_turnover REAL,
            sharpe_delta REAL,
            fitness_delta REAL,
            passed INTEGER,
            message TEXT,
            tested_at TEXT
        )
        """
    )
    conn.execute(
        """
        INSERT INTO research_robustness (candidate_id, test_type, test_value, perturbed_sharpe, passed, message)
        VALUES (1, 'neutralization', 'SECTOR', 1.88, 1, 'Passed old test')
        """
    )
    conn.commit()
    conn.close()

    # 2. Run Database.migrate()
    async def _migrate():
        db = Database(db_path)
        await db.create_all()

        async with db.session() as session:
            rows = (await session.scalars(select(ResearchRobustness))).all()
            assert len(rows) == 1
            legacy_row = rows[0]

            # Preserved values
            assert legacy_row.candidate_id == 1
            assert legacy_row.test_type == "neutralization"
            assert legacy_row.test_value == "SECTOR"
            assert legacy_row.perturbed_sharpe == 1.88
            assert legacy_row.passed is True

            # Migrated safeguard columns
            assert legacy_row.is_synthetic is True
            assert legacy_row.result_kind == "LEGACY_SYNTHETIC"
            assert legacy_row.status == "COMPLETED"
            assert legacy_row.brain_simulation_id is None

        await db.dispose()

    asyncio.run(_migrate())


def test_api_endpoint_production_path_no_random_fabrication(tmp_path: Path):
    """P0-04 Production API: run_candidate_suite via RobustnessEngine does not fabricate random numbers."""
    async def _run():
        db, cat, vault, cand_id, ev_id, _, _ = await setup_test_environment(tmp_path)
        try:
            # When BRAIN is unavailable, API reports honest NOT_RUN without random metrics
            engine = RobustnessEngine(db, vault=vault, endpoints=None)

            suite = engine.generate_test_suite(cand_id, base_neutralization="INDUSTRY", base_decay=4)
            results = await engine.run_candidate_suite(cand_id, suite=suite, baseline_eval_id=ev_id)

            assert len(results) > 0
            for r in results:
                assert r.status == "NOT_RUN"
                assert r.is_synthetic is False
                assert r.perturbed_sharpe is None
                assert r.sharpe_delta is None
                assert r.passed is False
                assert "unavailable" in r.message.lower()
        finally:
            await cat.close()
            await db.dispose()

    asyncio.run(_run())
