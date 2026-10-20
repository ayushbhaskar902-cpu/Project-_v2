import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

from alpha_harness.application.research import ResearchSessionService, ResearchWorkflowService


class _EmptyResult:
    def all(self):
        return []


class _DatabaseSession:
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        return False

    async def execute(self, statement):
        return _EmptyResult()


class _Database:
    def session(self):
        return _DatabaseSession()


def test_session_service_enqueue_delegates_without_http_concerns():
    async def run():
        orchestrator = SimpleNamespace(enqueue_candidates_for_simulation=AsyncMock(return_value=[11, 12]))
        state = SimpleNamespace(orchestrator=orchestrator, engine="engine")
        payload = SimpleNamespace(candidate_ids=[11, 12], task="evaluate")

        result = await ResearchSessionService(state).enqueue(7, payload)

        assert result == [11, 12]
        orchestrator.enqueue_candidates_for_simulation.assert_awaited_once_with(
            session_id=7,
            candidate_ids=[11, 12],
            task="evaluate",
            batch_engine="engine",
        )

    asyncio.run(run())


def test_session_service_generate_delegates_exact_generation_target():
    async def run():
        orchestrator = SimpleNamespace(run_generation_round=AsyncMock(return_value=[21]))
        state = SimpleNamespace(orchestrator=orchestrator)

        result = await ResearchSessionService(state).generate(9, 80)

        assert result == [21]
        orchestrator.run_generation_round.assert_awaited_once_with(
            session_id=9, total_target_candidates=80
        )

    asyncio.run(run())


def test_learning_service_preserves_response_and_snapshot_contract():
    async def run():
        learning_result = SimpleNamespace(
        generation=3,
        batch_size=20,
        winner_count=4,
        loser_count=16,
        avg_sharpe=1.2,
        best_sharpe=2.4,
        insights_generated=["one", "two"],
        updated_prompt_strategy_id=41,
        updated_prompt_strategy_ids=[41, 42],
        categories_evolved=["momentum"],
    )
        learning = SimpleNamespace(execute_learning_cycle=AsyncMock(return_value=learning_result))
        leaderboard = SimpleNamespace(capture_snapshot=AsyncMock())
        state = SimpleNamespace(learning=learning, leaderboard=leaderboard)

        result = await ResearchWorkflowService(state).learning_cycle(5, "momentum")

        assert result == {
        "status": "success",
        "sessionId": 5,
        "generation": 3,
        "batchSize": 20,
        "winnerCount": 4,
        "loserCount": 16,
        "avgSharpe": 1.2,
        "bestSharpe": 2.4,
        "insightsGenerated": 2,
        "updatedPromptStrategyId": 41,
        "updatedPromptStrategyIds": [41, 42],
        "categoriesEvolved": ["momentum"],
    }
        learning.execute_learning_cycle.assert_awaited_once_with(session_id=5, category="momentum")
        leaderboard.capture_snapshot.assert_awaited_once_with(session_id=5, generation=3)

    asyncio.run(run())


def test_robustness_service_empty_result_contract():
    async def run():
        state = SimpleNamespace(db=_Database())

        result = await ResearchWorkflowService(state).robustness(4, SimpleNamespace(candidate_ids=[]))

        assert result == {
        "status": "success",
        "sessionId": 4,
        "candidatesTested": 0,
        "totalTests": 0,
        "passedTests": 0,
        "results": [],
        }

    asyncio.run(run())


def test_curation_service_empty_result_contract():
    async def run():
        state = SimpleNamespace(db=_Database())

        result = await ResearchWorkflowService(state).curate_pool(SimpleNamespace(session_id=None))

        assert result == {
        "poolSize": 0,
        "members": [],
        "combinedSharpe": 0.0,
        "maxCorrelation": 0.0,
        "avgCorrelation": 0.0,
        }

    asyncio.run(run())
