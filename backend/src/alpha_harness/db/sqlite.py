"""Async SQLAlchemy engine, session factory, and the additive schema migration.

WAL mode plus a busy timeout: the background simulation tracker writes while HTTP
handlers read, and SQLite's default rollback journal would make them block each other.

``create_all`` only creates missing *tables*, so a column added to a model after the
database was created never appears and surfaces much later as ``no such column``.
:func:`migrate` closes that gap on every startup by comparing the live schema against
``Base.metadata``.

**Additive, with two exceptions.** A column that has *disappeared* from a model is reported
rather than quietly reconciled, unless it is listed in :data:`RETIRED`, or left ``NOT NULL``
with no default: nothing writes that one, so it blocks every insert into its table, and it
is dropped because nothing reads it either.
"""

from __future__ import annotations

from datetime import UTC, datetime
import json
import re
import sqlite3
from contextlib import asynccontextmanager, closing
from typing import TYPE_CHECKING, Any

import structlog
from sqlalchemy import Index, Table, event, inspect, text
from sqlalchemy.dialects import sqlite
from sqlalchemy.exc import OperationalError
from sqlalchemy.ext.asyncio import (
    AsyncConnection,
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.schema import CreateColumn

from .models import Base

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator
    from pathlib import Path


def _apply_pragmas(dbapi_connection: Any, _record: object) -> None:
    cursor = dbapi_connection.cursor()
    try:
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA synchronous=NORMAL")
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.execute("PRAGMA busy_timeout=5000")
    finally:
        cursor.close()


class Database:
    """Owns the engine and hands out sessions."""

    def __init__(self, path: Path) -> None:
        self._engine: AsyncEngine = create_async_engine(f"sqlite+aiosqlite:///{path}")
        event.listen(self._engine.sync_engine, "connect", _apply_pragmas)
        self._sessionmaker = async_sessionmaker(self._engine, expire_on_commit=False)

    async def create_all(self) -> None:
        """Bring the schema up to the models, dropping only what :func:`migrate` says.

        Not just ``create_all``: that leaves an existing table alone, so a column added to
        a model later never appears. See :func:`migrate`.
        """
        async with self._engine.begin() as conn:
            await migrate(conn)

    @asynccontextmanager
    async def session(self) -> AsyncGenerator[AsyncSession]:
        """A session that commits on success and rolls back on failure."""
        async with self._sessionmaker() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise

    async def healthcheck(self) -> bool:
        async with self._engine.connect() as conn:
            result = await conn.execute(text("SELECT 1"))
            return result.scalar() == 1

    async def dispose(self) -> None:
        await self._engine.dispose()


#: Snapshots kept, one per version that started against this database.
SNAPSHOTS_KEPT = 3


def snapshot(path: Path, version: str) -> Path | None:
    """Copy the database aside the first time ``version`` starts, before it migrates.

    Migration can drop columns, and falling back to the previous build restores its code
    but not this file. SQLite's backup API rather than a file copy, which can tear a WAL
    database. Returns the new snapshot, or ``None`` when there was nothing to do.
    """
    if not path.exists():
        return None
    folder = path.parent / "backups"
    target = folder / f"{path.stem}-before-{re.sub(r'[^\w.+-]', '_', version)}.db"
    if target.exists():
        return None
    folder.mkdir(mode=0o700, exist_ok=True)
    partial = target.with_suffix(".partial")
    with closing(sqlite3.connect(path)) as source, closing(sqlite3.connect(partial)) as copy:
        source.backup(copy)
    partial.replace(target)
    older = sorted(folder.glob(f"{path.stem}-before-*.db"), key=lambda p: p.stat().st_mtime)
    for stale in older[:-SNAPSHOTS_KEPT]:
        stale.unlink()
    log.info("db.snapshot", path=str(target))
    return target


# --- migration -------------------------------------------------------------------

log = structlog.get_logger(__name__)


class MigrationError(RuntimeError):
    """A schema change this migrator will not make on its own."""


#: Columns a model gave up on purpose, dropped wherever they are found. Only ones an older
#: build can put back, nullable or with a server default: the launcher falls back to the
#: previous build when a new one fails to start, and that build re-adds what its models declare.
RETIRED = {"sync_run": {"cursor_dataset", "fields_expected"}}


async def migrate(connection: AsyncConnection) -> None:
    """Create what is missing and log what was done."""
    live_tables = set(await connection.run_sync(_table_names))
    await connection.run_sync(Base.metadata.create_all)

    added_columns: list[str] = []
    added_indexes: list[str] = []
    unknown_columns: list[str] = []
    dropped_columns: list[str] = []
    retired_columns: list[str] = []

    for table in Base.metadata.sorted_tables:
        if table.name not in live_tables:
            continue  # create_all just made it, so it is current by construction.

        present = await _columns(connection, table.name)

        for column in table.columns:
            if column.name in present:
                continue
            ddl = _add_column_sql(table, column)
            await connection.execute(text(ddl))
            added_columns.append(f"{table.name}.{column.name}")

        expected = {c.name for c in table.columns}
        for name in sorted(present.keys() - expected):
            retired = name in RETIRED.get(table.name, ())
            # A leftover column is harmless unless it is NOT NULL with no default: nothing
            # writes it, so every insert into the table fails forever. Measured on a
            # consultant's machine, where `study.sampler_notes` from an older build made
            # every lab's Add Task answer 500.
            if (retired or present[name]) and await _drop_column(connection, table.name, name):
                (retired_columns if retired else dropped_columns).append(f"{table.name}.{name}")
            else:
                unknown_columns.append(f"{table.name}.{name}")

        added_indexes.extend(
            [index.name or "" for index in table.indexes if await _create_index(connection, index)]
        )

    # P0-04: Non-destructive migration marking legacy synthetic robustness records
    # Legacy records lack genuine brain_simulation_id and were produced by random.uniform
    if "research_robustness" in live_tables:
        await connection.execute(
            text(
                "UPDATE research_robustness "
                "SET is_synthetic = 1, result_kind = 'LEGACY_SYNTHETIC' "
                "WHERE (brain_simulation_id IS NULL OR brain_simulation_id = '') "
                "AND (result_kind = 'GENUINE' OR result_kind IS NULL)"
            )
        )

    # P1-01: Non-destructive migration marking legacy heuristic research insights
    if "research_insight" in live_tables:
        await connection.execute(
            text(
                "UPDATE research_insight "
                "SET execution_status = 'LEGACY_HEURISTIC', is_fallback = 1 "
                "WHERE execution_status IS NULL OR execution_status = ''"
            )
        )

    # P1-02: Non-destructive legacy category normalization and active deduplication
    if "prompt_strategy" in live_tables:
        await connection.execute(
            text(
                "UPDATE prompt_strategy "
                "SET category = lower(trim(category)) "
                "WHERE category IS NOT NULL AND category != lower(trim(category))"
            )
        )
        # Normalize explicitly supported canonical aliases (e.g. mean-reversion -> mean_reversion)
        # while preserving distinct categories (e.g. reversal) and custom identities
        await connection.execute(
            text(
                "UPDATE prompt_strategy "
                "SET category = 'mean_reversion' "
                "WHERE category IN ('mean-reversion', 'mean reversion', 'meanreversion')"
            )
        )
        # Deactivate duplicate active session-specific strategies, retaining highest (version DESC, id DESC)
        await connection.execute(
            text(
                """
                UPDATE prompt_strategy
                SET is_active = 0
                WHERE is_active = 1 AND session_id IS NOT NULL AND id NOT IN (
                    SELECT id FROM (
                        SELECT id,
                               ROW_NUMBER() OVER (
                                   PARTITION BY session_id, category
                                   ORDER BY version DESC, id DESC
                               ) as rn
                        FROM prompt_strategy
                        WHERE is_active = 1 AND session_id IS NOT NULL
                    ) WHERE rn = 1
                )
                """
            )
        )
        # Deactivate duplicate active global strategies (session_id IS NULL), retaining highest (version DESC, id DESC)
        await connection.execute(
            text(
                """
                UPDATE prompt_strategy
                SET is_active = 0
                WHERE is_active = 1 AND session_id IS NULL AND id NOT IN (
                    SELECT id FROM (
                        SELECT id,
                               ROW_NUMBER() OVER (
                                   PARTITION BY category
                                   ORDER BY version DESC, id DESC
                               ) as rn
                        FROM prompt_strategy
                        WHERE is_active = 1 AND session_id IS NULL
                    ) WHERE rn = 1
                )
                """
            )
        )

    if "research_researcher" in live_tables:
        await connection.execute(
            text(
                "UPDATE research_researcher "
                "SET strategy_focus = lower(trim(strategy_focus)) "
                "WHERE strategy_focus IS NOT NULL AND strategy_focus != lower(trim(strategy_focus))"
            )
        )
        await connection.execute(
            text(
                "UPDATE research_researcher "
                "SET strategy_focus = 'mean_reversion' "
                "WHERE strategy_focus IN ('mean-reversion', 'mean reversion', 'meanreversion')"
            )
        )

    # Refresh live tables after create_all to capture newly created tables
    current_tables = set(await connection.run_sync(_table_names))

    # P1-03: Immutable prompt version triggers, partial unique indexes, and canonical seed versions
    if "prompt_version" in current_tables:
        # 1. Database-enforced immutability triggers (reject UPDATE and DELETE)
        await connection.execute(
            text(
                """
                CREATE TRIGGER IF NOT EXISTS trg_prompt_version_no_update
                BEFORE UPDATE ON prompt_version
                BEGIN
                    SELECT RAISE(ABORT, 'PromptVersion records are immutable: UPDATE operations are strictly prohibited');
                END;
                """
            )
        )
        await connection.execute(
            text(
                """
                CREATE TRIGGER IF NOT EXISTS trg_prompt_version_no_delete
                BEFORE DELETE ON prompt_version
                BEGIN
                    SELECT RAISE(ABORT, 'PromptVersion records are immutable: DELETE operations are strictly prohibited');
                END;
                """
            )
        )

        # 2. Database-enforced partial unique indexes for version identity
        # A. Strategy-associated versions: unique per (origin_strategy_id, version)
        await connection.execute(
            text(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS uq_prompt_version_strategy_ver
                ON prompt_version(origin_strategy_id, version)
                WHERE origin_strategy_id IS NOT NULL;
                """
            )
        )
        # B. Session-standalone versions (no strategy row): unique per (origin_session_id, category, version)
        await connection.execute(
            text(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS uq_prompt_version_session_cat_ver
                ON prompt_version(origin_session_id, category, version)
                WHERE origin_session_id IS NOT NULL AND origin_strategy_id IS NULL;
                """
            )
        )
        # C. Global baseline versions (no session, no strategy): unique per (category, version)
        await connection.execute(
            text(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS uq_prompt_version_global_cat_ver
                ON prompt_version(category, version)
                WHERE origin_session_id IS NULL AND origin_strategy_id IS NULL;
                """
            )
        )

        # 3. Canonical seed PromptVersion creation and global strategy linkage
        from ..prompts.library import SEED_TEMPLATES, normalize_category
        from ..prompts.versioning import compute_prompt_content_hash, normalize_prompt_components

        for cat_enum, tmpl in SEED_TEMPLATES.items():
            cat_norm = normalize_category(cat_enum)
            res = await connection.execute(
                text(
                    """
                    SELECT id FROM prompt_version
                    WHERE origin_session_id IS NULL
                      AND origin_strategy_id IS NULL
                      AND category = :cat
                      AND version = 1
                    LIMIT 1
                    """
                ),
                {"cat": cat_norm},
            )
            seed_row = res.fetchone()
            if seed_row:
                seed_pv_id = seed_row[0]
            else:
                t_text, n_guidance, n_examples, n_anti_patterns = normalize_prompt_components(
                    template_text=tmpl.template_text,
                    guidance={
                        "strategy": tmpl.strategy_guidance,
                        "operators": tmpl.operator_guidance,
                    },
                    examples=tmpl.default_examples,
                    anti_patterns=tmpl.anti_patterns,
                )
                c_hash = compute_prompt_content_hash(
                    template_text=t_text,
                    guidance=n_guidance,
                    examples=n_examples,
                    anti_patterns=n_anti_patterns,
                )
                now_str = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")
                import json
                await connection.execute(
                    text(
                        """
                        INSERT INTO prompt_version (
                            origin_session_id, origin_strategy_id, category, version,
                            authoring_source, template_text, guidance, examples,
                            anti_patterns, content_hash, parent_version_id, created_at
                        ) VALUES (
                            NULL, NULL, :category, 1, 'SEED', :template_text, :guidance,
                            :examples, :anti_patterns, :content_hash, NULL, :created_at
                        )
                        """
                    ),
                    {
                        "category": cat_norm,
                        "template_text": t_text,
                        "guidance": n_guidance,
                        "examples": json.dumps(n_examples),
                        "anti_patterns": json.dumps(n_anti_patterns),
                        "content_hash": c_hash,
                        "created_at": now_str,
                    },
                )
                res_inserted = await connection.execute(
                    text("SELECT last_insert_rowid()")
                )
                seed_pv_id = res_inserted.scalar()

            # Link active global prompt_strategy rows if currently unversioned
            if "prompt_strategy" in current_tables and seed_pv_id:
                await connection.execute(
                    text(
                        """
                        UPDATE prompt_strategy
                        SET current_prompt_version_id = :pv_id
                        WHERE session_id IS NULL
                          AND category = :cat
                          AND (current_prompt_version_id IS NULL OR current_prompt_version_id = '')
                        """
                    ),
                    {"pv_id": seed_pv_id, "cat": cat_norm},
                )

    # P1-04: Non-destructive migration marking legacy generation attempts and candidates
    if "generation_attempt" in current_tables:
        await connection.execute(
            text(
                """
                UPDATE generation_attempt
                SET dataset_selection_policy = 'LEGACY_UNSPECIFIED'
                WHERE dataset_selection_policy IS NULL
                """
            )
        )
    if "research_candidate" in current_tables:
        await connection.execute(
            text(
                """
                UPDATE research_candidate
                SET dataset_selection_policy = 'LEGACY_UNSPECIFIED'
                WHERE dataset_selection_policy IS NULL
                """
            )
        )

    if added_columns or added_indexes or retired_columns:
        log.info(
            "db.migrated", columns=added_columns, indexes=added_indexes, retired=retired_columns
        )
    if dropped_columns:
        log.warning(
            "db.dropped_blocking_columns",
            columns=dropped_columns,
            detail=(
                "Left by an older build, NOT NULL with no default, so nothing could be "
                "inserted into their tables. Nothing read them."
            ),
        )
    if unknown_columns:
        # Not fatal: a nullable extra column costs nothing at runtime, and dropping it is
        # the destructive act this migrator refuses. Still worth saying once per start.
        log.warning(
            "db.unknown_columns",
            columns=unknown_columns,
            detail=(
                "These exist in the database but not in the models. Nothing reads them. "
                "Listing one in RETIRED drops it."
            ),
        )


def _table_names(sync_connection: Any) -> list[str]:
    return list(inspect(sync_connection).get_table_names())


async def _columns(connection: AsyncConnection, table: str) -> dict[str, bool]:
    """Every live column, mapped to whether a row could be inserted without naming it.

    ``PRAGMA table_info`` answers ``(cid, name, type, notnull, dflt_value, pk)``. A primary
    key is exempt: SQLite fills an ``INTEGER PRIMARY KEY`` itself.
    """
    result = await connection.execute(text(f'PRAGMA table_info("{table}")'))
    return {row[1]: bool(row[3]) and row[4] is None and not row[5] for row in result.fetchall()}


async def _drop_column(connection: AsyncConnection, table: str, column: str) -> bool:
    """Drop a column, reporting whether SQLite allowed it.

    It refuses one an index or a generated column depends on, which is a refusal to report
    rather than a reason to stop starting up.
    """
    try:
        await connection.execute(text(f'ALTER TABLE "{table}" DROP COLUMN "{column}"'))
    except OperationalError as exc:
        log.warning("db.drop_column_refused", table=table, column=column, error=str(exc))
        return False
    return True


async def _create_index(connection: AsyncConnection, index: Index) -> bool:
    """Create an index if it is missing. Returns whether it was created."""
    existing = await connection.execute(
        text("SELECT name FROM sqlite_master WHERE type = 'index' AND name = :name"),
        {"name": index.name},
    )
    if existing.first() is not None:
        return False
    await connection.run_sync(index.create)
    return True


def _add_column_sql(table: Table, column: Any) -> str:
    """``ALTER TABLE ... ADD COLUMN`` for one column.

    SQLite will not add a ``NOT NULL`` column to a table that already has rows without a
    constant default, so a new column is nullable or declares a ``server_default``;
    anything else gets a real migration.
    """
    if not column.nullable and column.server_default is None:
        raise MigrationError(
            f"Cannot add {table.name}.{column.name}: it is NOT NULL with no server_default, "
            "so existing rows have no value to take. Make it nullable, give it a "
            "server_default, or write a real migration."
        )
    rendered = CreateColumn(column).compile(dialect=sqlite.dialect())
    return f'ALTER TABLE "{table.name}" ADD COLUMN {rendered}'
