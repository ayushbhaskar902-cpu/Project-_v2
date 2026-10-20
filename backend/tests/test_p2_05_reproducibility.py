"""P2-05: stable experiment seeds and the P0-04 genuine-evidence boundary."""

from __future__ import annotations

import asyncio
import os
import random
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from alpha_harness.brain.schemas import SimulationRequest, SimulationSettings
from alpha_harness.db.models import Study
from alpha_harness.db.sqlite import Database
from alpha_harness.labs import ga, power_pool
from alpha_harness.labs.fastexpr import Node
from alpha_harness.labs.launch import add_study, preview_samples
from alpha_harness.labs.params import SEARCH_SAMPLER, PowerPoolParams, SearchParams
from alpha_harness.labs.seeding import (
    SEED_DERIVATION_VERSION,
    derive_seed,
    reproducibility_known,
    reproducibility_metadata,
)
from alpha_harness.tools.settings_sampler import expand, shuffle_batches


def test_seed_derivation_is_canonical_and_configuration_sensitive() -> None:
    cfg = {"region": "USA", "delay": 1, "fields": ["close", "volume"]}
    first = derive_seed("run-a", cfg, explicit_seed=73)
    assert first == derive_seed(
        "run-a", {"fields": ["close", "volume"], "delay": 1, "region": "USA"}, explicit_seed=73
    )
    assert first != derive_seed("run-b", cfg, explicit_seed=73)
    assert first != derive_seed("run-a", {**cfg, "delay": 0}, explicit_seed=73)
    assert first != derive_seed("run-a", cfg, explicit_seed=74)


def test_seed_derivation_rejects_missing_identity_and_negative_seed() -> None:
    with pytest.raises(ValueError, match="identity"):
        derive_seed("", {"region": "USA"})
    with pytest.raises(ValueError, match="non-negative"):
        derive_seed("run-a", {}, explicit_seed=-1)


def test_generated_preview_is_stable_across_process_hash_seeds() -> None:
    code = (
        "import json; from alpha_harness.brain.schemas import SimulationRequest, SimulationSettings; "
        "from alpha_harness.labs.launch import preview_samples; "
        "from alpha_harness.labs.seeding import derive_seed; "
        "identity=__import__('sys').argv[1]; config={'b': 2, 'a': ['x', 'y']}; "
        "seed=derive_seed(identity, config, explicit_seed=9); "
        "draw=lambda rng: SimulationRequest(regular=f'rank(close) + {rng.randrange(1_000_000)}', "
        "settings=SimulationSettings(region='USA', delay=1, universe='TOP3000')); "
        "result=preview_samples(draw, tries=5, seed=seed); "
        "print(json.dumps([item.model_dump(mode='json') for item in result], sort_keys=True))"
    )
    backend = Path(__file__).resolve().parents[1]
    outputs = []
    for hash_seed, identity in (
        ("1", "experiment-unique-1"),
        ("987654", "experiment-unique-1"),
        ("1", "experiment-unique-2"),
    ):
        env = os.environ.copy()
        env["PYTHONHASHSEED"] = hash_seed
        proc = subprocess.run(
            [sys.executable, "-c", code, identity],
            cwd=backend,
            env=env,
            capture_output=True,
            check=True,
            text=True,
            timeout=20,
        )
        outputs.append(proc.stdout.strip())
    assert outputs[0] == outputs[1]
    assert outputs[0] != "[]"
    assert outputs[0] != outputs[2]


def test_seed_metadata_records_unknown_legacy_as_unknown() -> None:
    metadata = reproducibility_metadata("run-x", {"delay": 1}, 5, explicit_seed=2)
    assert metadata["effective_seed"] == 5
    assert metadata["derivation_version"] == SEED_DERIVATION_VERSION
    assert '"delay":1' in metadata["canonical_configuration"]
    assert reproducibility_known(5, metadata)
    assert not reproducibility_known(None, None)
    assert not reproducibility_known(5, None)


def test_settings_batch_shuffle_replays_and_new_seed_changes_order() -> None:
    full = [(i,) for i in range(12)]
    tails = [(i,) for i in range(4)]
    first = shuffle_batches(full.copy(), tails.copy(), seed=123)
    replay = shuffle_batches(full.copy(), tails.copy(), seed=123)
    different = shuffle_batches(full.copy(), tails.copy(), seed=456)
    assert first == replay
    assert first != different


def test_settings_sampler_rejects_unseeded_expansion() -> None:
    with pytest.raises(TypeError, match="seed"):
        expand([], set(), set(), set(), {})


def test_all_random_lab_draws_accept_repeatable_local_rngs() -> None:
    def preview(rng: random.Random) -> SimulationRequest:
        return SimulationRequest(
            regular=f"rank(close) + {rng.randrange(100)}",
            settings=SimulationSettings(region="USA", delay=1, universe="TOP3000"),
        )

    assert preview_samples(preview, tries=5, seed=21) == preview_samples(
        preview, tries=5, seed=21
    )

    market = ga.Market(
        table={},
        swappable={},
        dataset_of={},
        by_dataset={},
        names=frozenset(),
        neutralizations=("INDUSTRY", "SECTOR"),
    )
    tree = Node(kind="name", value="close")
    settings = {"neutralization": "INDUSTRY", "decay": 0, "truncation": 0.08}
    left = ga.mutate(random.Random(9), tree, settings, market, 0.5, force=True)
    right = ga.mutate(random.Random(9), tree, settings, market, 0.5, force=True)
    assert left == right

    context = power_pool.Context(
        id="fundamental6",
        name="Fundamentals",
        category="Fundamentals",
        description="fixture only",
        fields=(),
        basics=(),
        groups=(),
        types={},
        own=frozenset(),
        names=frozenset({"close"}),
        held={"close": frozenset({"U1", "U2"})},
    )
    run = PowerPoolParams(
        region="USA",
        delay=1,
        universes=["U1", "U2"],
        neutralizations=["INDUSTRY", "SECTOR"],
    )
    assert power_pool.draw(random.Random(33), frozenset({"close"}), context, run) == (
        power_pool.draw(random.Random(33), frozenset({"close"}), context, run)
    )


def test_power_pool_pending_response_survives_param_round_trip() -> None:
    pending = {
        "dataset": "fundamental6",
        "offset": 200,
        "fields": 40,
        "items": [{"expression": "rank(close)"}],
        "entry": {"model": "provider/model", "tokens": 123},
    }
    reproducibility = {
        "experiment_id": "power-pool-unique-run",
        "derivation_version": SEED_DERIVATION_VERSION,
        "effective_seed": 912,
        "canonical_configuration": '{"sampler":"power-pool"}',
    }
    run = PowerPoolParams(
        region="USA",
        delay=1,
        universes=["TOP3000"],
        neutralizations=["INDUSTRY"],
        model="provider/model",
        dataset_ids=["fundamental6"],
        llm={"calls": 2, "pendingBatch": pending},
        _reproducibility=reproducibility,
    )
    restored = PowerPoolParams.model_validate(run.dump())
    assert restored.llm["pendingBatch"] == pending
    assert restored.dump()["_reproducibility"] == reproducibility


def test_add_study_persists_effective_seed_and_configuration(tmp_path: Path) -> None:
    async def _run() -> None:
        db = Database(tmp_path / "p2-05-seed.sqlite")
        await db.create_all()

        async def notify() -> None:
            return None

        state = SimpleNamespace(db=db, optimizer=SimpleNamespace(notify=notify))
        try:
            now = datetime(2026, 10, 20, tzinfo=UTC)
            params = SearchParams(region="USA", delay=1, space={})
            added = await add_study(
                state,
                now=now,
                sampler=SEARCH_SAMPLER,
                params=params,
                simulations=10,
                batch_size=10,
                template_source="# test",
                seed=73,
            )
            async with db.session() as session:
                row = await session.get(Study, added.id)
                assert row is not None
                original_task = row.task
                original_seed = row.seed
                original_metadata = row.sampler_params["_reproducibility"]
                configuration = {"sampler": SEARCH_SAMPLER, "params": params.dump()}
                assert original_seed == derive_seed(
                    original_task, configuration, explicit_seed=73
                )
                assert original_metadata["effective_seed"] == original_seed
                assert original_metadata["derivation_version"] == SEED_DERIVATION_VERSION
                assert original_metadata["canonical_configuration"]
        finally:
            await db.dispose()

        # A new engine/session against the same SQLite file models an application restart.
        restarted = Database(tmp_path / "p2-05-seed.sqlite")
        await restarted.create_all()
        try:
            async with restarted.session() as session:
                row = await session.get(Study, added.id)
                assert row is not None
                assert row.task == original_task
                assert row.seed == original_seed
                assert row.sampler_params["_reproducibility"] == original_metadata
        finally:
            await restarted.dispose()

    asyncio.run(_run())
