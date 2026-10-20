"""Regression coverage for durable evaluation context in background vault backfills."""

import asyncio

from alpha_harness.vault.backfill import Backfill


def test_fetch_each_reports_the_matching_evaluation_job_and_continues_siblings() -> None:
    async def run() -> None:
        backfill = object.__new__(Backfill)
        backfill._evaluation_jobs = {}
        fetched: list[str] = []
        reported: list[tuple[int, int, str, BaseException]] = []

        async def fetch_returns(alpha_id: str) -> None:
            fetched.append(alpha_id)
            if alpha_id == "alpha-fails":
                raise RuntimeError("temporary vault outage")

        async def on_failure(
            job_id: int, simulation_attempt: int, alpha_id: str, exc: BaseException
        ) -> None:
            reported.append((job_id, simulation_attempt, alpha_id, exc))

        backfill.fetch_returns = fetch_returns
        backfill.register_evaluation_job("alpha-fails", 73, 4, on_failure)

        await backfill._fetch_each(["alpha-fails", "alpha-succeeds"])

        assert fetched == ["alpha-fails", "alpha-succeeds"]
        assert len(reported) == 1
        job_id, simulation_attempt, alpha_id, error = reported[0]
        assert (job_id, simulation_attempt, alpha_id) == (73, 4, "alpha-fails")
        assert str(error) == "temporary vault outage"
        assert "alpha-fails" not in backfill._evaluation_jobs

    asyncio.run(run())
