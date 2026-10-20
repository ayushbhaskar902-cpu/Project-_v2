"""Operational state, stored in SQLite.

Everything here is small, transactional, and must survive a crash. The bulk analytical
data — the data-field catalog — lives in DuckDB instead (see :mod:`.duck`).

Schema evolution is additive only, applied by :func:`.sqlite.migrate` on every startup:
missing columns and indexes as well as missing tables, which ``create_all`` alone does
*not* add.

Never drop or rename a column on ``simulation_record``. Losing a row there means losing
the ability to cancel a running simulation, which is the exact failure this project
exists to prevent.
"""

from datetime import UTC, date, datetime
from enum import StrEnum
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    LargeBinary,
    String,
    Text,
    TypeDecorator,
    UniqueConstraint,
    text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


def utcnow() -> datetime:
    return datetime.now(UTC)


class UtcDateTime(TypeDecorator[datetime]):
    """A timestamp stored as UTC and read back UTC-aware.

    SQLite has no zone: the stock type drops an aware value's offset without converting,
    so ``12:00-04:00`` came back as ``12:00``. Converting on the way in makes the stored
    text UTC; a naive value is refused rather than guessed at.
    """

    impl = DateTime(timezone=True)
    cache_ok = True

    def process_bind_param(self, value: datetime | None, dialect: Any) -> datetime | None:  # noqa: ARG002
        if value is None:
            return None
        if value.tzinfo is None:
            raise ValueError(f"Refusing to store naive datetime {value!r}; use utcnow().")
        return value.astimezone(UTC)

    def process_result_value(self, value: datetime | None, dialect: Any) -> datetime | None:  # noqa: ARG002
        if value is None:
            return None
        return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


class Base(DeclarativeBase):
    type_annotation_map = {dict[str, Any]: JSON, list[Any]: JSON, datetime: UtcDateTime}


# --- credentials & session ------------------------------------------------


class Credential(Base):
    """A BRAIN login. The password is sealed by :class:`~alpha_harness.sealing.Sealer`."""

    __tablename__ = "credential"

    id: Mapped[int] = mapped_column(primary_key=True)
    email: Mapped[str] = mapped_column(String(320), unique=True)
    password_sealed: Mapped[bytes] = mapped_column(LargeBinary)
    created_at: Mapped[datetime] = mapped_column(UtcDateTime, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(UtcDateTime, default=utcnow, onupdate=utcnow)
    last_login_at: Mapped[datetime | None] = mapped_column(UtcDateTime)


class BrainSessionRow(Base):
    """A cached cookie jar, so a restart does not cost another proof-of-work solve.

    Cookies are bearer credentials, so the jar is sealed just like the password.
    """

    __tablename__ = "brain_session"

    id: Mapped[int] = mapped_column(primary_key=True)
    credential_id: Mapped[int] = mapped_column(ForeignKey("credential.id", ondelete="CASCADE"))
    cookies_sealed: Mapped[bytes] = mapped_column(LargeBinary)
    user_id: Mapped[str | None] = mapped_column(String(64))
    permissions: Mapped[list[Any]] = mapped_column(JSON, default=list)
    expires_at: Mapped[datetime | None] = mapped_column(UtcDateTime)
    created_at: Mapped[datetime] = mapped_column(UtcDateTime, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(UtcDateTime, default=utcnow, onupdate=utcnow)


# --- simulations ----------------------------------------------------------


class SimStatus(StrEnum):
    """Local lifecycle. Distinct from the platform's own status.

    ``PENDING`` is the crash window: the request is in flight and the platform id is not
    known yet, so a row left there after a restart may be a real running simulation and
    must be reconciled rather than discarded.

    ``ORPHANED`` means it was sent but no id came back. Not final:
    :mod:`alpha_harness.engine.reconcile` adopts the alpha it produced or queues the
    request again, because treating it as final either loses the alpha or spends the
    quota twice.
    """

    QUEUED = "QUEUED"
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETE = "COMPLETE"
    WARNING = "WARNING"
    ERROR = "ERROR"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    TIMEOUT = "TIMEOUT"
    ORPHANED = "ORPHANED"
    REJECTED = "REJECTED"
    SKIPPED = "SKIPPED"

    @property
    def terminal(self) -> bool:
        return self not in ACTIVE


#: Not yet finished; every other status is final. One exception: a cancel BRAIN refuses
#: moves ``CANCELLED`` back to ``RUNNING``, because the simulation is in fact still running
#: and still has to be polled (:func:`.engine.lifecycle.cancel_after_lost_race`).
#: ``ORPHANED`` belongs here because an identical request arriving while its outcome is
#: reconciled must share the row rather than pay for a second run.
ACTIVE = (SimStatus.QUEUED, SimStatus.PENDING, SimStatus.RUNNING, SimStatus.ORPHANED)


class SimulationRecord(Base):
    """One simulation we asked BRAIN to run.

    The row is written **before** the HTTP request and updated with ``platform_id`` the
    moment a ``201`` lands, so a simulation can always be cancelled.

    For a multi-simulation, ``platform_id`` is the *parent* id — the only cancellable
    handle. Child ids only exist once the parent completes and are stored in
    ``child_ids``.
    """

    __tablename__ = "simulation_record"

    id: Mapped[int] = mapped_column(primary_key=True)

    # Platform identity
    platform_id: Mapped[str | None] = mapped_column(String(64), index=True)
    alpha_id: Mapped[str | None] = mapped_column(String(64), index=True)
    parent_record_id: Mapped[int | None] = mapped_column(
        ForeignKey("simulation_record.id", ondelete="SET NULL")
    )
    child_ids: Mapped[list[Any]] = mapped_column(JSON, default=list)
    is_batch: Mapped[bool] = mapped_column(default=False)
    #: Child ids exist only once the parent finishes, and resolving them costs one
    #: request each. This marks that work as done so it happens exactly once.
    children_expanded: Mapped[bool] = mapped_column(default=False)

    # What we asked for
    request_hash: Mapped[str] = mapped_column(String(64), index=True)
    payload: Mapped[dict[str, Any]] = mapped_column(JSON)
    expression: Mapped[str | None] = mapped_column(Text)

    # Batch-packing key: children of one multi-simulation must agree on all of these.
    sim_type: Mapped[str] = mapped_column(String(16), default="REGULAR")
    instrument_type: Mapped[str] = mapped_column(String(16), default="EQUITY")
    region: Mapped[str] = mapped_column(String(16))
    delay: Mapped[int] = mapped_column(Integer)
    language: Mapped[str] = mapped_column(String(16), default="FASTEXPR")
    universe: Mapped[str | None] = mapped_column(String(32))

    # Where it came from — lets the UI group work and apply per-task slot quotas.
    task: Mapped[str] = mapped_column(String(64), default="manual", index=True)

    # Lifecycle
    status: Mapped[str] = mapped_column(String(16), default=SimStatus.PENDING, index=True)
    platform_status: Mapped[str | None] = mapped_column(String(16))
    progress: Mapped[float | None] = mapped_column(Float)
    message: Mapped[str | None] = mapped_column(Text)
    #: Times reconcile queued this row again after its send's answer was lost.
    orphan_requeues: Mapped[int] = mapped_column(Integer, default=0, server_default="0")

    created_at: Mapped[datetime] = mapped_column(UtcDateTime, default=utcnow, index=True)
    submitted_at: Mapped[datetime | None] = mapped_column(UtcDateTime)
    #: When the POST that carried this row began — ``created_at`` is when it was queued,
    #: which can be hours earlier. Bounds the search for a send whose answer was lost.
    sent_at: Mapped[datetime | None] = mapped_column(UtcDateTime)
    finished_at: Mapped[datetime | None] = mapped_column(UtcDateTime)
    last_polled_at: Mapped[datetime | None] = mapped_column(UtcDateTime)

    __table_args__ = (
        Index("ix_sim_active", "status", "created_at"),
        Index("ix_sim_batch_key", "sim_type", "instrument_type", "region", "delay", "language"),
    )

    @property
    def elapsed_seconds(self) -> float | None:
        """Wall time since submission — what the matrix cell counts up."""
        if self.submitted_at is None:
            return None
        return ((self.finished_at or utcnow()) - self.submitted_at).total_seconds()


class DedupEntry(Base):
    """Hash of a canonical simulation payload -> the alpha it produced.

    Guards the daily quota against re-simulating something already run, as BRAIN recommends.
    """

    __tablename__ = "dedup_entry"

    request_hash: Mapped[str] = mapped_column(String(64), primary_key=True)
    alpha_id: Mapped[str] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(UtcDateTime, default=utcnow)


class TaskQuota(Base):
    """How many of the concurrent slots a named task may hold at once.

    Research runs in parallel streams — a manual experiment, a template sweep, an
    optimizer study — and without a cap one of them starves the others. A task with no
    row here competes freely for whatever is unallocated.
    """

    __tablename__ = "task_quota"

    name: Mapped[str] = mapped_column(String(64), primary_key=True)
    max_slots: Mapped[int] = mapped_column(Integer, default=1)
    enabled: Mapped[bool] = mapped_column(default=True)
    created_at: Mapped[datetime] = mapped_column(UtcDateTime, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(UtcDateTime, default=utcnow, onupdate=utcnow)


class Preference(Base):
    """One choice made in Settings, by name. A missing row means the default.

    A row per choice rather than a column each: a choice added later needs no migration, and
    a build older than it simply never reads the row.
    """

    __tablename__ = "preference"

    key: Mapped[str] = mapped_column(String(64), primary_key=True)
    value: Mapped[Any] = mapped_column(JSON)


class QuotaSnapshot(Base):
    """A reading of ``X-RateLimit-*`` from a simulation POST.

    Kept as a series rather than a single row so the UI can show quota burn over the day.
    """

    __tablename__ = "quota_snapshot"

    id: Mapped[int] = mapped_column(primary_key=True)
    limit_total: Mapped[int | None] = mapped_column(Integer)
    remaining: Mapped[int | None] = mapped_column(Integer)
    reset_seconds: Mapped[float | None] = mapped_column(Float)
    observed_at: Mapped[datetime] = mapped_column(UtcDateTime, default=utcnow, index=True)


class BrainCache(Base):
    """A BRAIN read kept so it is not asked for again: an Alpha's series, its correlations.

    Correlations and recordsets are Retry-After jobs, while an Alpha's in-sample history
    never changes until the period rolls. So the Alpha page reads from here and only goes
    back to BRAIN when asked to refresh.
    """

    __tablename__ = "brain_cache"

    key: Mapped[str] = mapped_column(String(160), primary_key=True)
    body: Mapped[Any] = mapped_column(JSON)
    fetched_at: Mapped[datetime] = mapped_column(UtcDateTime, default=utcnow)


# --- catalog sync ---------------------------------------------------------


class SyncStatus(StrEnum):
    RUNNING = "RUNNING"
    COMPLETE = "COMPLETE"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class SyncRun(Base):
    """One catalog crawl of a single (instrumentType, region, delay, universe) tuple."""

    __tablename__ = "sync_run"

    id: Mapped[int] = mapped_column(primary_key=True)
    instrument_type: Mapped[str] = mapped_column(String(16), default="EQUITY")
    region: Mapped[str] = mapped_column(String(16))
    delay: Mapped[int] = mapped_column(Integer)
    universe: Mapped[str] = mapped_column(String(32))

    status: Mapped[str] = mapped_column(String(16), default=SyncStatus.RUNNING, index=True)
    phase: Mapped[str | None] = mapped_column(String(32))
    #: Unused. Retired once every install has run a build with these server defaults: a build
    #: without them cannot put the columns back, so the launcher could not fall back to it.
    cursor_offset: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    truncated_datasets: Mapped[list[Any]] = mapped_column(JSON, default=list, server_default="[]")

    datasets_synced: Mapped[int] = mapped_column(Integer, default=0)
    fields_synced: Mapped[int] = mapped_column(Integer, default=0)
    categories_synced: Mapped[int] = mapped_column(Integer, default=0)

    error: Mapped[str | None] = mapped_column(Text)
    started_at: Mapped[datetime] = mapped_column(UtcDateTime, default=utcnow)
    finished_at: Mapped[datetime | None] = mapped_column(UtcDateTime)

    __table_args__ = (Index("ix_sync_tuple", "instrument_type", "region", "delay", "universe"),)


# --- platform metadata cache ---------------------------------------------


class MetadataCache(Base):
    """Cached platform metadata (``OPTIONS /simulations``, ``/operators``).

    Refreshed on login rather than hardcoded — regions and universes change.
    """

    __tablename__ = "metadata_cache"

    key: Mapped[str] = mapped_column(String(64), primary_key=True)
    value: Mapped[dict[str, Any]] = mapped_column(JSON)
    fetched_at: Mapped[datetime] = mapped_column(UtcDateTime, default=utcnow)


# --- template studio (Phase 3) -------------------------------------------


class Template(Base):
    """A YAML alpha template with typed variables."""

    __tablename__ = "template"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(128), unique=True)
    description: Mapped[str | None] = mapped_column(Text)
    source: Mapped[str] = mapped_column(Text)
    parsed: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    origin: Mapped[str] = mapped_column(String(16), default="human")  # human | llm
    tags: Mapped[list[Any]] = mapped_column(JSON, default=list)
    created_at: Mapped[datetime] = mapped_column(UtcDateTime, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(UtcDateTime, default=utcnow, onupdate=utcnow)


# --- optimization (Phase 4) ----------------------------------------------


class StudyStatus(StrEnum):
    IDLE = "IDLE"
    #: Told to run, waiting for its cores to fit in the free slots.
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    COMPLETE = "COMPLETE"
    FAILED = "FAILED"


class TrialState(StrEnum):
    """Mirrors Optuna's states, plus the two this application adds.

    ``QUEUED`` and ``RUNNING`` both mean "asked for, not yet told" as far as Optuna is
    concerned; the distinction is whether a simulation has actually started, which is
    what the UI needs to show.
    """

    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    COMPLETE = "COMPLETE"
    FAIL = "FAIL"
    PRUNED = "PRUNED"


class Study(Base):
    """One optimization run over one template.

    ``template_source`` is a *snapshot*, not a reference. A study is a record of a search
    over a particular space; if the template is edited afterwards the trials stop meaning
    anything, so the space is frozen at the moment the study is created.

    Optuna itself is rebuilt from the trial rows on demand rather than persisted
    separately. One source of truth, and the trials stay joinable to the simulations that
    produced them.
    """

    __tablename__ = "study"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(128), unique=True)
    template_id: Mapped[int | None] = mapped_column(ForeignKey("template.id", ondelete="SET NULL"))
    template_name: Mapped[str | None] = mapped_column(String(128))
    #: What the user calls the task, shown in place of its lab and template.
    label: Mapped[str | None] = mapped_column(String(128))
    template_source: Mapped[str] = mapped_column(Text)

    sampler: Mapped[str] = mapped_column(String(32), default="nsga3")
    sampler_params: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    objectives: Mapped[list[Any]] = mapped_column(JSON, default=list)
    directions: Mapped[list[Any]] = mapped_column(JSON, default=list)
    seed: Mapped[int | None] = mapped_column(Integer)

    #: Trials asked for per round. Kept a multiple of ten so a round fills whole
    #: multi-simulations rather than leaving slots holding part-empty batches.
    batch_size: Mapped[int] = mapped_column(Integer, default=10)
    max_trials: Mapped[int] = mapped_column(Integer, default=80)
    #: Slot-quota group, so a study cannot starve manual work.
    task: Mapped[str] = mapped_column(String(64), default="study")

    status: Mapped[str] = mapped_column(String(16), default=StudyStatus.IDLE, index=True)
    message: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(UtcDateTime, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(UtcDateTime, default=utcnow, onupdate=utcnow)
    #: First time it began running. A task can wait as IDLE for days, so this is what "how
    #: long did it take" means; ``created_at`` would count the waiting too.
    started_at: Mapped[datetime | None] = mapped_column(UtcDateTime)
    finished_at: Mapped[datetime | None] = mapped_column(UtcDateTime)


class Trial(Base):
    """One point in the search space, and what it turned into.

    The link to ``simulation_record`` is the whole reason trials live here rather than in
    Optuna's own storage: "which parameters produced this alpha" and "which simulation is
    this trial waiting on" are the two questions actually asked of a running study.
    """

    __tablename__ = "trial"

    id: Mapped[int] = mapped_column(primary_key=True)
    study_id: Mapped[int] = mapped_column(ForeignKey("study.id", ondelete="CASCADE"), index=True)
    number: Mapped[int] = mapped_column(Integer)

    params: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    #: What Optuna drew each value from: every choice of every categorical, so a field searched
    #: over a few thousand fields stores them all, on every trial — 127 KB a trial was measured.
    #: Deferred, so only the two readers that rebuild the search load it; decoding it on every
    #: row a screen or a harvest reads held the event loop for seconds.
    distributions: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, deferred=True)
    expression: Mapped[str | None] = mapped_column(Text)
    settings: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)

    state: Mapped[str] = mapped_column(String(16), default=TrialState.QUEUED, index=True)
    values: Mapped[list[Any] | None] = mapped_column(JSON)
    #: Submission-check violations, keyed by the platform's own check name. Zero or
    #: less is feasible, following Optuna's convention.
    constraint: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    feasible: Mapped[bool | None] = mapped_column()
    result: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    message: Mapped[str | None] = mapped_column(Text)

    simulation_record_id: Mapped[int | None] = mapped_column(
        ForeignKey("simulation_record.id", ondelete="SET NULL"), index=True
    )
    alpha_id: Mapped[str | None] = mapped_column(String(64), index=True)
    #: Evolution Lab only: 0 for the seeds, then one per bred generation.
    generation: Mapped[int | None] = mapped_column(Integer)

    created_at: Mapped[datetime] = mapped_column(UtcDateTime, default=utcnow)
    finished_at: Mapped[datetime | None] = mapped_column(UtcDateTime)

    __table_args__ = (
        UniqueConstraint("study_id", "number", name="uq_trial_number"),
        Index("ix_trial_open", "study_id", "state"),
    )


# --- conversations --------------------------------------------------------


class ChatThread(Base):
    """One conversation with the assistant.

    Persisted because a consultant who closes the laptop mid-thought should not lose it,
    and because the scope is part of the conversation: "which fields" means nothing
    without knowing which market is being talked about.
    """

    __tablename__ = "chat_thread"

    id: Mapped[int] = mapped_column(primary_key=True)
    title: Mapped[str] = mapped_column(String(160))
    instrument_type: Mapped[str] = mapped_column(String(16), default="EQUITY")
    region: Mapped[str] = mapped_column(String(16))
    delay: Mapped[int] = mapped_column(Integer)
    universe: Mapped[str] = mapped_column(String(32))
    created_at: Mapped[datetime] = mapped_column(UtcDateTime, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        UtcDateTime, default=utcnow, onupdate=utcnow, index=True
    )


class ChatMessage(Base):
    """One turn. ``meta`` carries the picks, model and cost of an assistant turn."""

    __tablename__ = "chat_message"

    id: Mapped[int] = mapped_column(primary_key=True)
    thread_id: Mapped[int] = mapped_column(
        ForeignKey("chat_thread.id", ondelete="CASCADE"), index=True
    )
    role: Mapped[str] = mapped_column(String(16))  # user | assistant
    text: Mapped[str] = mapped_column(Text)
    meta: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(UtcDateTime, default=utcnow)


class ApiKey(Base):
    """An assistant key. Sealed at rest; only a masked hint is ever shown.

    ``provider`` says whose key it is. Rotation is per provider: a Groq key cannot answer
    a request for a Gemini model, and offering it would spend a retry to learn that.
    """

    __tablename__ = "api_key"

    id: Mapped[int] = mapped_column(primary_key=True)
    label: Mapped[str] = mapped_column(String(64))
    provider: Mapped[str] = mapped_column(String(32), default="google", server_default="google")
    key_sealed: Mapped[bytes] = mapped_column(LargeBinary)
    hint: Mapped[str] = mapped_column(String(32))
    fingerprint: Mapped[str] = mapped_column(String(64), unique=True)
    enabled: Mapped[bool] = mapped_column(default=True)
    #: The user's own daily request ceiling, overriding the model's when set. Required for a
    #: paid provider, which has no ceiling of its own, and optional for a free one.
    daily_limit: Mapped[int | None] = mapped_column(nullable=True)
    last_ok_at: Mapped[datetime | None] = mapped_column(UtcDateTime)
    last_error: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(UtcDateTime, default=utcnow)


class KeyUsage(Base):
    """Local budget ledger per (key, model, quota day).

    Providers expose no remaining-quota endpoint, so requests per minute and per day are
    tracked client-side or rotation is guesswork. ``day`` is the date in the model's own
    reset time zone, since providers do not agree on when a day starts.
    """

    __tablename__ = "key_usage"

    id: Mapped[int] = mapped_column(primary_key=True)
    api_key_id: Mapped[int] = mapped_column(ForeignKey("api_key.id", ondelete="CASCADE"))
    model: Mapped[str] = mapped_column(String(64))
    day: Mapped[str] = mapped_column(String(10))  # YYYY-MM-DD, America/Los_Angeles
    requests: Mapped[int] = mapped_column(Integer, default=0)
    tokens: Mapped[int] = mapped_column(Integer, default=0)
    last_request_at: Mapped[datetime | None] = mapped_column(UtcDateTime)

    __table_args__ = (UniqueConstraint("api_key_id", "model", "day", name="uq_key_model_day"),)


class LLMModel(Base):
    """A model the user has set up, with the limits they read off their provider.

    Nothing about models is built in: which ones exist and what each allows change faster
    than any table here could keep up. Limits apply per key, so a second account doubles
    them. A model id is unique per provider: two providers may serve the same one.
    """

    __tablename__ = "llm_model"

    id: Mapped[int] = mapped_column(primary_key=True)
    provider: Mapped[str] = mapped_column(String(32))
    model: Mapped[str] = mapped_column(String(200))
    requests_per_minute: Mapped[int] = mapped_column(Integer)
    requests_per_day: Mapped[int] = mapped_column(Integer)
    #: The IANA time zone whose midnight starts the provider's new day for this model.
    reset_timezone: Mapped[str] = mapped_column(
        String(64), default="America/Los_Angeles", server_default="America/Los_Angeles"
    )
    #: The most tokens one Power Pool prompt may spend on this model, or null for the default.
    max_prompt_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(UtcDateTime, default=utcnow)

    # An index rather than a table constraint, so a database that already has the table
    # gains it on startup: SQLite cannot add a constraint to a live table.
    __table_args__ = (Index("uq_llm_model_provider_model", "provider", "model", unique=True),)


class Submission(Base):
    """An Alpha the consultant has told us they submitted on BRAIN.

    Submission happens by hand on the platform, so nothing here can observe it. The row is
    the user's own record, and the Submission Planner treats it as a permanent pool member
    every later candidate has to clear.
    """

    __tablename__ = "submission"

    alpha_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    submitted_at: Mapped[datetime] = mapped_column(UtcDateTime, default=utcnow)


# --- research engine ------------------------------------------------------
#
# The self-learning multi-LLM research engine. A ``ResearchSession`` is one autonomous
# campaign: the user gives a prompt, a budget, and a set of models, and the engine runs
# generation → simulation → evaluation → learning cycles until the budget runs out.
#
# Every table here is new. Columns are nullable or carry a ``server_default`` so the
# additive migrator can add them to a database that already has the rest of Alpha Harness.


class ResearchStatus(StrEnum):
    """Lifecycle of a research session or a generation within it."""

    IDLE = "IDLE"
    RUNNING = "RUNNING"
    GENERATING = "GENERATING"
    SIMULATING = "SIMULATING"
    EVALUATING = "EVALUATING"
    PAUSED = "PAUSED"
    LEARNING = "LEARNING"
    COMPLETE = "COMPLETE"
    FAILED = "FAILED"


class CandidateStatus(StrEnum):
    """Lifecycle of one generated alpha candidate."""

    DRAFT = "DRAFT"
    VALID = "VALID"
    INVALID = "INVALID"
    QUEUED = "QUEUED"
    SIMULATING = "SIMULATING"
    UNAVAILABLE = "UNAVAILABLE"
    EVALUATED = "EVALUATED"
    REJECTED = "REJECTED"
    ACCEPTED = "ACCEPTED"
    GRADUATED = "GRADUATED"


class PlatformEligibilityStatus(StrEnum):
    """Evidence-backed BRAIN eligibility, independent of research quality."""

    UNCHECKED = "UNCHECKED"
    CHECK_PENDING = "CHECK_PENDING"
    ELIGIBLE = "ELIGIBLE"
    INELIGIBLE = "INELIGIBLE"
    EXEMPT_OFFLINE = "EXEMPT_OFFLINE"


class ResearchSession(Base):
    """One autonomous multi-LLM research campaign.

    Created when the user hits Start on the Research Studio. Owns every researcher,
    candidate, evaluation and learning insight produced during its run. The ``prompt``
    is the user's seed idea; the engine refines it through self-learning.

    ``learning_frequency`` is the number of completed simulations between strategy
    updates. With 8 BRAIN slots × 10 items per multi-sim = 80 per round, the default
    is 80 so a full round finishes before the learning loop fires.
    """

    __tablename__ = "research_session"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str | None] = mapped_column(String(200))
    prompt: Mapped[str | None] = mapped_column(Text)

    #: Market scope, frozen when the session starts.
    instrument_type: Mapped[str] = mapped_column(String(16), default="EQUITY", server_default="EQUITY")
    region: Mapped[str | None] = mapped_column(String(16))
    universe: Mapped[str | None] = mapped_column(String(32))
    delay: Mapped[int | None] = mapped_column(Integer, server_default="1")

    #: Budget and cadence.
    simulation_budget: Mapped[int | None] = mapped_column(Integer, server_default="500")
    simulations_used: Mapped[int | None] = mapped_column(Integer, server_default="0")
    learning_frequency: Mapped[int | None] = mapped_column(Integer, server_default="80")
    current_generation: Mapped[int | None] = mapped_column(Integer, server_default="0")

    #: Which LLM models to use, as a JSON list of ``provider:model`` refs.
    model_refs: Mapped[list[Any] | None] = mapped_column(JSON, server_default="[]")
    #: Extra constraints the user sets: min Sharpe, max turnover, target datasets, etc.
    constraints: Mapped[dict[str, Any] | None] = mapped_column(JSON, server_default="{}")

    status: Mapped[str | None] = mapped_column(
        String(32), default=ResearchStatus.IDLE, server_default="IDLE", index=True
    )
    message: Mapped[str | None] = mapped_column(Text)

    #: P0-05: Autonomous experiment runner durable checkpointing.
    #: The current phase within the autonomous loop, persisted so a restart can resume.
    current_phase: Mapped[str | None] = mapped_column(String(32), server_default="IDLE")
    #: Candidate IDs generated in the current round, JSON list for crash recovery.
    current_round_candidate_ids: Mapped[list[int] | None] = mapped_column(JSON, server_default="[]")
    #: Consecutive unrecoverable errors; reset on every successful phase completion.
    error_count: Mapped[int | None] = mapped_column(Integer, server_default="0")
    #: Timestamp of last successful phase checkpoint.
    last_checkpoint_at: Mapped[datetime | None] = mapped_column(UtcDateTime)

    created_at: Mapped[datetime] = mapped_column(UtcDateTime, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(UtcDateTime, default=utcnow, onupdate=utcnow)
    started_at: Mapped[datetime | None] = mapped_column(UtcDateTime)
    finished_at: Mapped[datetime | None] = mapped_column(UtcDateTime)


class ResearchResearcher(Base):
    """One LLM agent persona inside a research session.

    Each row configures a single model instance with its own system prompt persona,
    temperature, and strategy focus. The session's orchestrator dispatches work to every
    active researcher in parallel and tracks per-model statistics here.
    """

    __tablename__ = "research_researcher"

    id: Mapped[int] = mapped_column(primary_key=True)
    session_id: Mapped[int | None] = mapped_column(
        ForeignKey("research_session.id", ondelete="CASCADE"), index=True
    )

    #: ``provider:model`` reference, matching :func:`~.llm.registry.model_ref`.
    model_ref: Mapped[str | None] = mapped_column(String(200))
    #: Human label for this agent (e.g. "Momentum Explorer", "Quality Analyst").
    label: Mapped[str | None] = mapped_column(String(128))
    #: Research focus: momentum, mean_reversion, value, quality, volatility, general.
    strategy_focus: Mapped[str | None] = mapped_column(String(64), server_default="general")
    #: The system prompt persona injected before every generation call.
    system_prompt: Mapped[str | None] = mapped_column(Text)
    temperature: Mapped[float | None] = mapped_column(Float, server_default="0.8")
    #: P1-04: Explicit target dataset and deliberate allocation policy
    target_dataset_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    dataset_allocation_policy: Mapped[str | None] = mapped_column(String(32), nullable=True)

    #: Running statistics, updated after each evaluation cycle.
    total_generated: Mapped[int | None] = mapped_column(Integer, server_default="0")
    total_valid: Mapped[int | None] = mapped_column(Integer, server_default="0")
    total_simulated: Mapped[int | None] = mapped_column(Integer, server_default="0")
    total_accepted: Mapped[int | None] = mapped_column(Integer, server_default="0")
    avg_sharpe: Mapped[float | None] = mapped_column(Float)
    avg_fitness: Mapped[float | None] = mapped_column(Float)
    best_sharpe: Mapped[float | None] = mapped_column(Float)
    total_tokens: Mapped[int | None] = mapped_column(Integer, server_default="0")

    enabled: Mapped[bool | None] = mapped_column(default=True, server_default="1")
    created_at: Mapped[datetime] = mapped_column(UtcDateTime, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(UtcDateTime, default=utcnow, onupdate=utcnow)


class PromptVersion(Base):
    """Immutable point-in-time snapshot of an LLM alpha generation prompt.

    Append-only: Protected by SQLite triggers rejecting all UPDATE and DELETE operations.
    Archival origins decouple PromptVersion lifecycle from mutable sessions or strategies.
    """

    __tablename__ = "prompt_version"

    id: Mapped[int] = mapped_column(primary_key=True)

    #: Archival origin identifiers (decoupled, plain indexed integers, no ON DELETE cascades)
    origin_session_id: Mapped[int | None] = mapped_column(Integer, nullable=True, index=True)
    origin_strategy_id: Mapped[int | None] = mapped_column(Integer, nullable=True, index=True)

    #: Classification & Identity
    category: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    version: Mapped[int] = mapped_column(Integer, nullable=False)
    authoring_source: Mapped[str] = mapped_column(
        String(32), default="SYSTEM", server_default="SYSTEM"
    )  #: "SEED", "SYSTEM", "EVOLUTION", "MANUAL"

    #: Canonical Prompt Snapshot Components (Option A)
    template_text: Mapped[str] = mapped_column(Text, nullable=False)
    guidance: Mapped[str | None] = mapped_column(Text, nullable=True)
    examples: Mapped[list[Any]] = mapped_column(JSON, default=list, server_default="[]")
    anti_patterns: Mapped[list[Any]] = mapped_column(JSON, default=list, server_default="[]")

    #: Content-Only Cryptographic Integrity Hash (SHA-256)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False, index=True)

    #: Lineage (Immutable self-referential lineage protected by RESTRICT)
    parent_version_id: Mapped[int | None] = mapped_column(
        ForeignKey("prompt_version.id", ondelete="RESTRICT"), nullable=True, index=True
    )

    created_at: Mapped[datetime] = mapped_column(UtcDateTime, default=utcnow, nullable=False)

    __table_args__ = (
        Index("ix_prompt_version_cat_ver", "category", "version"),
    )


class PromptStrategy(Base):
    """A versioned prompt template for alpha generation.

    Prompt strategies are the unit of prompt experimentation. Each version captures a
    snapshot of the system prompt additions, example patterns, operator guidance, and
    dataset hints that the generation pipeline injects. The self-learning engine creates
    new versions when it discovers better patterns.

    Performance statistics are updated after evaluation so prompts can be compared and
    ranked on the leaderboard.
    """

    __tablename__ = "prompt_strategy"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str | None] = mapped_column(String(200))
    version: Mapped[int | None] = mapped_column(Integer, server_default="1")
    #: Category of research strategy: momentum, mean_reversion, value, quality, etc.
    category: Mapped[str | None] = mapped_column(String(64))
    #: The full prompt template text, with ``{placeholders}`` for dynamic injection.
    template_text: Mapped[str | None] = mapped_column(Text)
    #: Structured guidance extracted by the learning engine: operator tips, field tips, etc.
    guidance: Mapped[dict[str, Any] | None] = mapped_column(JSON, server_default="{}")
    #: Example expressions that performed well, injected as few-shot context.
    examples: Mapped[list[Any] | None] = mapped_column(JSON, server_default="[]")
    #: Anti-patterns: expressions or patterns to explicitly avoid.
    anti_patterns: Mapped[list[Any] | None] = mapped_column(JSON, server_default="[]")

    #: Performance tracking, updated by the evaluation engine.
    times_used: Mapped[int | None] = mapped_column(Integer, server_default="0")
    total_candidates: Mapped[int | None] = mapped_column(Integer, server_default="0")
    valid_rate: Mapped[float | None] = mapped_column(Float)
    avg_sharpe: Mapped[float | None] = mapped_column(Float)
    avg_fitness: Mapped[float | None] = mapped_column(Float)
    top_decile_rate: Mapped[float | None] = mapped_column(Float)
    oos_stability: Mapped[float | None] = mapped_column(Float)

    #: Which session created this version, if any (null for seed prompts).
    session_id: Mapped[int | None] = mapped_column(
        ForeignKey("research_session.id", ondelete="SET NULL")
    )
    #: The previous version this one was derived from, for lineage tracking.
    parent_id: Mapped[int | None] = mapped_column(
        ForeignKey("prompt_strategy.id", ondelete="SET NULL")
    )
    #: P1-03: Immutable snapshot linkage
    current_prompt_version_id: Mapped[int | None] = mapped_column(
        ForeignKey("prompt_version.id", ondelete="RESTRICT"), nullable=True, index=True
    )

    is_active: Mapped[bool | None] = mapped_column(default=True, server_default="1")
    #: P1-13: Validation status & promotion lifecycle (EXPERIMENTAL, SHADOW_TESTING, PROMOTED, REJECTED, ROLLED_BACK)
    validation_status: Mapped[str | None] = mapped_column(
        String(32), default="EXPERIMENTAL", server_default="EXPERIMENTAL"
    )
    previous_promoted_strategy_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    previous_promoted_version_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    promotion_criteria: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    promoted_at: Mapped[datetime | None] = mapped_column(UtcDateTime, nullable=True)
    rolled_back_at: Mapped[datetime | None] = mapped_column(UtcDateTime, nullable=True)
    rejection_reason: Mapped[str | None] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(UtcDateTime, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(UtcDateTime, default=utcnow, onupdate=utcnow)

    __table_args__ = (
        Index("ix_prompt_strategy_active", "is_active", "category"),
        Index("ix_prompt_strategy_status", "validation_status", "category"),
    )


class GenerationAttempt(Base):
    """Durable record of a single LLM generation dispatch attempt.

    Preserves the exact rendered system prompt (including catalog context and operator tables),
    user prompt, response schema, and model configuration once per researcher dispatch.
    """

    __tablename__ = "generation_attempt"

    id: Mapped[int] = mapped_column(primary_key=True)
    session_id: Mapped[int] = mapped_column(
        ForeignKey("research_session.id", ondelete="CASCADE"), index=True
    )
    researcher_id: Mapped[int | None] = mapped_column(
        ForeignKey("research_researcher.id", ondelete="SET NULL"), nullable=True, index=True
    )
    prompt_strategy_id: Mapped[int | None] = mapped_column(
        ForeignKey("prompt_strategy.id", ondelete="SET NULL"), nullable=True
    )
    #: P1-03: Exact immutable snapshot attribution
    prompt_version_id: Mapped[int | None] = mapped_column(
        ForeignKey("prompt_version.id", ondelete="RESTRICT"), nullable=True, index=True
    )
    #: Resolution tier: EXACT_SESSION, GLOBAL_CATEGORY, SESSION_GENERAL, GLOBAL_GENERAL, TEMPLATE_FALLBACK
    resolution_tier: Mapped[str | None] = mapped_column(String(32), nullable=True)
    generation: Mapped[int | None] = mapped_column(Integer, server_default="0")
    model_ref: Mapped[str | None] = mapped_column(String(64), nullable=True)
    temperature: Mapped[float | None] = mapped_column(Float, nullable=True)
    system_prompt: Mapped[str | None] = mapped_column(Text, nullable=True)
    user_prompt: Mapped[str | None] = mapped_column(Text, nullable=True)
    response_schema: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    #: P1-04: Dataset selection context and lineage
    target_dataset_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    catalog_version: Mapped[str | None] = mapped_column(String(64), nullable=True)
    dataset_selection_policy: Mapped[str | None] = mapped_column(String(32), nullable=True)
    dataset_selection_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    #: P1-05: Authoritative datafield catalog version and allowed field lineage
    datafield_catalog_version: Mapped[str | None] = mapped_column(String(64), nullable=True)
    allowed_fields: Mapped[list[Any] | None] = mapped_column(JSON, nullable=True)
    #: P1-06: Authoritative operator catalog version
    operator_catalog_version: Mapped[str | None] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(UtcDateTime, default=utcnow)


class ResearchCandidate(Base):
    """One alpha expression generated by an LLM researcher.

    Tracks the full lineage: which session, which researcher, which prompt strategy, and
    which generation produced it. The ``hypothesis`` is the LLM's stated reasoning; the
    ``expression`` is the BRAIN-ready code. Validation results from :mod:`.labs.fastexpr`
    are stored so a syntax error can be fed back to the LLM for retry.

    Links to ``simulation_record`` once the candidate reaches the BatchEngine queue, and
    to ``alpha_id`` once BRAIN completes the simulation.
    """

    __tablename__ = "research_candidate"

    id: Mapped[int] = mapped_column(primary_key=True)
    session_id: Mapped[int | None] = mapped_column(
        ForeignKey("research_session.id", ondelete="CASCADE"), index=True
    )
    researcher_id: Mapped[int | None] = mapped_column(
        ForeignKey("research_researcher.id", ondelete="SET NULL"), index=True
    )
    prompt_strategy_id: Mapped[int | None] = mapped_column(
        ForeignKey("prompt_strategy.id", ondelete="SET NULL")
    )
    #: P1-03: Exact immutable snapshot attribution
    prompt_version_id: Mapped[int | None] = mapped_column(
        ForeignKey("prompt_version.id", ondelete="RESTRICT"), nullable=True, index=True
    )
    generation: Mapped[int | None] = mapped_column(Integer, server_default="0")

    #: The LLM's stated reasoning for this alpha.
    hypothesis: Mapped[str | None] = mapped_column(Text)
    #: The WorldQuant BRAIN Fast Expression code. Fast Expression ('FASTEXPR') is the
    #: sole supported generation language; Python alpha generation is explicitly out of scope.
    expression: Mapped[str | None] = mapped_column(Text)
    #: Generation/simulation language. Strictly restricted to 'FASTEXPR'.
    language: Mapped[str | None] = mapped_column(String(16), server_default="FASTEXPR")

    #: Which datasets and fields the expression references, extracted by the parser.
    datasets_used: Mapped[list[Any] | None] = mapped_column(JSON, server_default="[]")
    fields_used: Mapped[list[Any] | None] = mapped_column(JSON, server_default="[]")
    operator_count: Mapped[int | None] = mapped_column(Integer)
    field_count: Mapped[int | None] = mapped_column(Integer)

    #: AST validation result.
    is_valid: Mapped[bool | None] = mapped_column()
    validation_error: Mapped[str | None] = mapped_column(Text)
    #: How many LLM retry attempts were needed (0 = first try was valid).
    retry_count: Mapped[int | None] = mapped_column(Integer, server_default="0")

    #: Simulation linkage, set once the candidate enters the BatchEngine queue.
    simulation_record_id: Mapped[int | None] = mapped_column(
        ForeignKey("simulation_record.id", ondelete="SET NULL"), index=True
    )
    alpha_id: Mapped[str | None] = mapped_column(String(64), index=True)

    #: BRAIN simulation settings used.
    sim_region: Mapped[str | None] = mapped_column(String(16))
    sim_universe: Mapped[str | None] = mapped_column(String(32))
    sim_neutralization: Mapped[str | None] = mapped_column(String(32))
    sim_decay: Mapped[int | None] = mapped_column(Integer)
    sim_delay: Mapped[int | None] = mapped_column(Integer)

    #: P1-15: Explicit, validated, and persisted simulation settings per candidate
    simulation_settings: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    settings_hash: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    settings_profile_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    settings_profile_version: Mapped[str | None] = mapped_column(String(32), nullable=True)
    settings_schema_version: Mapped[str | None] = mapped_column(String(32), nullable=True)
    settings_verification_status: Mapped[str | None] = mapped_column(
        String(32), nullable=True, server_default="UNVERIFIED", default="UNVERIFIED"
    )
    candidate_provenance_hash: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)

    @property
    def is_simulation_settings_complete(self) -> bool:
        """Returns True only when candidate has complete, verified simulation settings.

        Legacy rows without complete simulation_settings cannot claim complete reproducibility.
        """
        if self.simulation_settings is None:
            return False
        return self.settings_verification_status not in ("LEGACY_UNVERIFIED", "INVALID")

    def effective_simulation_settings(self) -> dict[str, Any]:
        """Returns the effective simulation settings.

        For legacy rows with missing complete settings, returns reconstructed partial scalar
        settings with an explicit legacy_unverified marker; missing values are NOT fabricated.
        """
        if self.simulation_settings is not None:
            return dict(self.simulation_settings)
        return {
            "region": self.sim_region,
            "universe": self.sim_universe,
            "delay": self.sim_delay,
            "decay": self.sim_decay,
            "neutralization": self.sim_neutralization,
            "is_legacy_unverified": True,
            "verification_status": "LEGACY_UNVERIFIED",
        }

    def to_simulation_settings(self) -> Any:
        """Constructs a SimulationSettings wire object for simulation dispatch."""
        from ..brain.schemas import SimulationSettings

        if self.simulation_settings:
            s = self.simulation_settings
            return SimulationSettings(
                instrument_type=s.get("instrument_type") or "EQUITY",
                region=s.get("region") or self.sim_region or "USA",
                universe=s.get("universe") or self.sim_universe or "TOP3000",
                delay=int(s.get("delay") if s.get("delay") is not None else (self.sim_delay or 1)),
                decay=int(s.get("decay") if s.get("decay") is not None else (self.sim_decay or 0)),
                neutralization=s.get("neutralization") or self.sim_neutralization or "NONE",
                truncation=float(s.get("truncation", 0.08)),
                pasteurization=str(s.get("pasteurization", "ON")),
                unit_handling=str(s.get("unit_handling", "VERIFY")),
                nan_handling=str(s.get("nan_handling", "OFF")),
                language=str(s.get("language", "FASTEXPR")),
                visualization=bool(s.get("visualization", False)),
                test_period=s.get("test_period"),
                max_trade=s.get("max_trade"),
                max_position=s.get("max_position"),
                simulation_mode=s.get("simulation_mode"),
                selection_handling=s.get("selection_handling"),
                selection_limit=s.get("selection_limit"),
                component_activation=s.get("component_activation"),
            )
        return SimulationSettings(
            instrument_type="EQUITY",
            region=self.sim_region or "USA",
            universe=self.sim_universe or "TOP3000",
            delay=self.sim_delay or 1,
            decay=self.sim_decay or 0,
            neutralization=self.sim_neutralization or "NONE",
            truncation=0.08,
            pasteurization="ON",
            unit_handling="VERIFY",
            nan_handling="OFF",
            language="FASTEXPR",
            visualization=False,
        )

    #: Exact generation-attempt lineage and point-in-time model configuration snapshot
    generation_attempt_id: Mapped[int | None] = mapped_column(
        ForeignKey("generation_attempt.id", ondelete="SET NULL"), nullable=True, index=True
    )
    model_ref: Mapped[str | None] = mapped_column(String(64), nullable=True)
    temperature: Mapped[float | None] = mapped_column(Float, nullable=True)
    raw_expression: Mapped[str | None] = mapped_column(Text, nullable=True)
    repair_history: Mapped[list[Any] | None] = mapped_column(JSON, nullable=True)
    #: P1-04: Selected target dataset and catalog provenance (distinguished from datasets_used)
    target_dataset_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    catalog_version: Mapped[str | None] = mapped_column(String(64), nullable=True)
    dataset_selection_policy: Mapped[str | None] = mapped_column(String(32), nullable=True)
    dataset_selection_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    #: P1-05: Datafield catalog version and field families
    datafield_catalog_version: Mapped[str | None] = mapped_column(String(64), nullable=True)
    field_families: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    #: P1-06: Authoritative operator catalog version and operators used
    operator_catalog_version: Mapped[str | None] = mapped_column(String(64), nullable=True)
    operators_used: Mapped[list[Any] | None] = mapped_column(JSON, nullable=True)

    #: P1-14: Deterministic deduplication, AST skeleton fingerprint, and pruning reason
    expression_hash: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    ast_skeleton: Mapped[str | None] = mapped_column(Text, nullable=True)
    pruning_reason: Mapped[str | None] = mapped_column(String(64), nullable=True)

    status: Mapped[str | None] = mapped_column(
        String(16), default=CandidateStatus.DRAFT, server_default="DRAFT", index=True
    )
    #: P1-16: never inferred from internal metrics or statistical acceptance.
    platform_eligibility_status: Mapped[str] = mapped_column(
        String(24), default=PlatformEligibilityStatus.UNCHECKED,
        server_default=PlatformEligibilityStatus.UNCHECKED, index=True,
    )
    platform_eligibility_checked_at: Mapped[datetime | None] = mapped_column(UtcDateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(UtcDateTime, default=utcnow)
    finished_at: Mapped[datetime | None] = mapped_column(UtcDateTime)

    __table_args__ = (
        Index("ix_candidate_session_gen", "session_id", "generation"),
        Index("ix_candidate_session_status", "session_id", "status"),
    )


class ResearchEvaluation(Base):
    """Post-simulation quantitative evaluation of one alpha candidate.

    Separates in-sample (IS) from out-of-sample (OOS) metrics. The IS/OOS gap is the
    primary overfitting signal: ``(is_sharpe - oos_sharpe) / is_sharpe``. A composite
    score ranks alphas on the leaderboard.

    Always one-to-one with a :class:`ResearchCandidate` that reached COMPLETE.
    """

    __tablename__ = "research_evaluation"

    id: Mapped[int] = mapped_column(primary_key=True)
    candidate_id: Mapped[int | None] = mapped_column(
        ForeignKey("research_candidate.id", ondelete="CASCADE"),
        unique=True,
        index=True,
    )

    #: In-sample metrics (train period, typically first 8 of 10 years).
    is_sharpe: Mapped[float | None] = mapped_column(Float)
    is_fitness: Mapped[float | None] = mapped_column(Float)
    is_turnover: Mapped[float | None] = mapped_column(Float)
    is_returns: Mapped[float | None] = mapped_column(Float)
    is_drawdown: Mapped[float | None] = mapped_column(Float)
    is_margin: Mapped[float | None] = mapped_column(Float)

    #: Out-of-sample metrics (test period, typically last 2 of 10 years).
    oos_sharpe: Mapped[float | None] = mapped_column(Float)
    oos_fitness: Mapped[float | None] = mapped_column(Float)
    oos_turnover: Mapped[float | None] = mapped_column(Float)
    oos_returns: Mapped[float | None] = mapped_column(Float)
    oos_drawdown: Mapped[float | None] = mapped_column(Float)

    #: Full-period metrics (all 10 years combined).
    full_sharpe: Mapped[float | None] = mapped_column(Float)
    full_fitness: Mapped[float | None] = mapped_column(Float)
    after_cost_sharpe: Mapped[float | None] = mapped_column(Float)

    #: Overfitting signals.
    is_oos_gap: Mapped[float | None] = mapped_column(Float)
    #: Sharpe decay ratio: (IS Sharpe - OOS Sharpe) / IS Sharpe, 0 is perfect, 1 is total.
    sharpe_decay: Mapped[float | None] = mapped_column(Float)

    #: Composite leaderboard score:
    #:   0.4 * after_cost_sharpe + 0.3 * fitness + 0.2 * (1 - sharpe_decay) - 0.1 * turnover
    composite_score: Mapped[float | None] = mapped_column(Float)

    #: BRAIN submission check results, if fetched from an authoritative platform response.
    checks: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    #: Internal validation thresholds. Legacy rows may have these in ``checks``.
    internal_checks: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    checks_passed: Mapped[int | None] = mapped_column(Integer)
    checks_total: Mapped[int | None] = mapped_column(Integer)

    #: Provenance and verification: strictly True only when evaluated against genuine BRAIN vault series.
    is_verified: Mapped[bool] = mapped_column(Boolean, default=False, server_default=text("0"), index=True)
    provenance: Mapped[str | None] = mapped_column(String(64), nullable=True)

    #: Exact quantitative evaluation parameters used to compute metrics
    cost_bps: Mapped[float | None] = mapped_column(Float, nullable=True)
    split_date: Mapped[date | None] = mapped_column(Date, nullable=True)

    #: P0-03: Evaluation methodology scheme and 3-way chronological partitions
    evaluation_scheme: Mapped[str] = mapped_column(String(32), default="SPLIT_3WAY_DEV", server_default="LEGACY_2WAY")
    is_end_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    val_end_date: Mapped[date | None] = mapped_column(Date, nullable=True)

    #: Pure Validation partition metrics (selection and tuning)
    val_sharpe: Mapped[float | None] = mapped_column(Float, nullable=True)
    val_fitness: Mapped[float | None] = mapped_column(Float, nullable=True)
    val_turnover: Mapped[float | None] = mapped_column(Float, nullable=True)
    val_returns: Mapped[float | None] = mapped_column(Float, nullable=True)
    val_drawdown: Mapped[float | None] = mapped_column(Float, nullable=True)
    val_after_cost_sharpe: Mapped[float | None] = mapped_column(Float, nullable=True)
    val_composite_score: Mapped[float | None] = mapped_column(Float, nullable=True)

    #: Diagnostic only (IS-vs-Validation decay; strictly excluded from selection score)
    is_val_decay: Mapped[float | None] = mapped_column(Float, nullable=True)

    #: Final Holdout partition metrics (strictly uncomputed/NULL until explicit graduation)
    holdout_sharpe: Mapped[float | None] = mapped_column(Float, nullable=True)
    holdout_fitness: Mapped[float | None] = mapped_column(Float, nullable=True)
    holdout_turnover: Mapped[float | None] = mapped_column(Float, nullable=True)
    holdout_returns: Mapped[float | None] = mapped_column(Float, nullable=True)
    holdout_drawdown: Mapped[float | None] = mapped_column(Float, nullable=True)
    holdout_after_cost_sharpe: Mapped[float | None] = mapped_column(Float, nullable=True)
    holdout_evaluated_at: Mapped[datetime | None] = mapped_column(UtcDateTime, nullable=True)

    #: Frozen configuration fingerprint
    config_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    canonical_spec: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)

    evaluated_at: Mapped[datetime | None] = mapped_column(UtcDateTime, default=utcnow)


class HoldoutAssessment(Base):
    """Immutable persistent registry of consumed and revealed market holdout windows.

    Guarantees that once a market holdout period is unsealed anywhere in the research program,
    it cannot be reused as an untouched final holdout by another candidate or session.
    """

    __tablename__ = "holdout_assessment"

    id: Mapped[int] = mapped_column(primary_key=True)

    #: Assessment classification: "FINAL_ASSESSMENT", "LEGACY_CONTAMINATED", or "LEGACY_CONTAMINATED_UNKNOWN"
    assessment_type: Mapped[str] = mapped_column(String(32), default="FINAL_ASSESSMENT", nullable=False)

    #: Exact temporal boundaries of the consumed holdout window (NULL if unknown legacy)
    holdout_start_date: Mapped[date | None] = mapped_column(Date, index=True, nullable=True)
    holdout_end_date: Mapped[date | None] = mapped_column(Date, index=True, nullable=True)
    observation_count: Mapped[int | None] = mapped_column(Integer, nullable=True)

    #: Provenance and linkage
    market_scope: Mapped[str] = mapped_column(String(64), index=True, nullable=False)
    candidate_id: Mapped[int | None] = mapped_column(
        ForeignKey("research_candidate.id", ondelete="RESTRICT"),
        unique=True,
        nullable=True,
    )
    session_id: Mapped[int | None] = mapped_column(
        ForeignKey("research_session.id", ondelete="RESTRICT"),
        nullable=True,
    )
    alpha_id: Mapped[str | None] = mapped_column(String(64), nullable=True)

    #: Frozen configuration fingerprint (populated strictly for FINAL_ASSESSMENT)
    config_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    canonical_spec: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)

    #: Unsealed performance metrics (NULL for legacy contaminated records)
    holdout_sharpe: Mapped[float | None] = mapped_column(Float, nullable=True)
    holdout_fitness: Mapped[float | None] = mapped_column(Float, nullable=True)
    holdout_turnover: Mapped[float | None] = mapped_column(Float, nullable=True)
    holdout_returns: Mapped[float | None] = mapped_column(Float, nullable=True)
    holdout_drawdown: Mapped[float | None] = mapped_column(Float, nullable=True)

    evaluated_at: Mapped[datetime] = mapped_column(UtcDateTime, default=utcnow, nullable=False)

    __table_args__ = (
        Index("ix_holdout_dates", "holdout_start_date", "holdout_end_date"),
    )


class ResearchRobustness(Base):
    """Perturbation and sensitivity tests for an evaluated alpha.

    Each row is one test applied to one candidate. The test type names what was varied
    (neutralization, decay, universe, region) and the result says whether the alpha
    survived the perturbation. A candidate must pass all mandatory tests to be ACCEPTED.

    Kept separate from :class:`ResearchEvaluation` because a candidate may undergo many
    tests, each with its own settings and outcome.
    """

    __tablename__ = "research_robustness"

    id: Mapped[int] = mapped_column(primary_key=True)
    candidate_id: Mapped[int | None] = mapped_column(
        ForeignKey("research_candidate.id", ondelete="CASCADE"), index=True
    )

    #: What was tested: "neutralization", "decay", "universe", "region", "time_period".
    test_type: Mapped[str | None] = mapped_column(String(32))
    #: The specific perturbation: e.g. "SECTOR" for neutralization, "10" for decay.
    test_value: Mapped[str | None] = mapped_column(String(64))
    #: The simulation that ran the perturbed alpha, if any.
    simulation_record_id: Mapped[int | None] = mapped_column(
        ForeignKey("simulation_record.id", ondelete="SET NULL")
    )

    #: Metrics under the perturbed setting.
    perturbed_sharpe: Mapped[float | None] = mapped_column(Float)
    perturbed_fitness: Mapped[float | None] = mapped_column(Float)
    perturbed_turnover: Mapped[float | None] = mapped_column(Float)

    #: How much the metric changed relative to the baseline.
    sharpe_delta: Mapped[float | None] = mapped_column(Float)
    fitness_delta: Mapped[float | None] = mapped_column(Float)

    passed: Mapped[bool | None] = mapped_column()
    message: Mapped[str | None] = mapped_column(Text)
    tested_at: Mapped[datetime | None] = mapped_column(UtcDateTime, default=utcnow)

    # Execution status: PENDING, RUNNING, COMPLETED, NOT_RUN, UNSUPPORTED, FAILED
    status: Mapped[str] = mapped_column(
        String(32), default="COMPLETED", server_default=text("'COMPLETED'")
    )
    # Result kind: GENUINE, LEGACY_SYNTHETIC
    result_kind: Mapped[str] = mapped_column(
        String(32), default="GENUINE", server_default=text("'GENUINE'")
    )
    # Explicit flag indicating whether record was generated by synthetic simulation/approximation
    is_synthetic: Mapped[bool] = mapped_column(
        Boolean, default=False, server_default=text("0")
    )

    # Provenance linkage
    baseline_eval_id: Mapped[int | None] = mapped_column(
        ForeignKey("research_evaluation.id", ondelete="SET NULL"), nullable=True
    )
    brain_simulation_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    perturbed_alpha_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    perturbation_config: Mapped[str | None] = mapped_column(Text, nullable=True)
    error_details: Mapped[str | None] = mapped_column(Text, nullable=True)
    val_start_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    val_end_date: Mapped[date | None] = mapped_column(Date, nullable=True)

    # Execution status: PENDING, RUNNING, COMPLETED, NOT_RUN, UNSUPPORTED, FAILED
    status: Mapped[str] = mapped_column(
        String(32), default="COMPLETED", server_default=text("'COMPLETED'")
    )
    # Result kind: GENUINE, LEGACY_SYNTHETIC
    result_kind: Mapped[str] = mapped_column(
        String(32), default="GENUINE", server_default=text("'GENUINE'")
    )
    # Explicit flag indicating whether record was generated by synthetic simulation/approximation
    is_synthetic: Mapped[bool] = mapped_column(
        Boolean, default=False, server_default=text("0")
    )

    # Provenance linkage
    baseline_eval_id: Mapped[int | None] = mapped_column(
        ForeignKey("research_evaluation.id", ondelete="SET NULL"), nullable=True
    )
    brain_simulation_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    perturbed_alpha_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    perturbation_config: Mapped[str | None] = mapped_column(Text, nullable=True)
    error_details: Mapped[str | None] = mapped_column(Text, nullable=True)
    val_start_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    val_end_date: Mapped[date | None] = mapped_column(Date, nullable=True)

    __table_args__ = (
        Index("ix_robustness_candidate_type", "candidate_id", "test_type"),
    )


class ResearchInsight(Base):
    """One piece of knowledge extracted by the self-learning engine.

    After every ``learning_frequency`` simulations, the engine analyses the batch of
    results, identifies patterns in winners and losers, and records them here. Each
    insight is versioned by ``generation`` and typed by ``insight_type``.

    Insights feed back into prompt strategies: winning patterns become examples, failure
    modes become anti-patterns, and operator/field statistics update the guidance block.
    The engine can answer "what did it learn?" by reading these rows chronologically.
    """

    __tablename__ = "research_insight"

    id: Mapped[int] = mapped_column(primary_key=True)
    session_id: Mapped[int | None] = mapped_column(
        ForeignKey("research_session.id", ondelete="CASCADE"), index=True
    )
    generation: Mapped[int | None] = mapped_column(Integer, server_default="0")

    #: What kind of insight: winning_pattern, failure_mode, operator_stat, field_stat,
    #: strategy_update, prompt_mutation.
    insight_type: Mapped[str | None] = mapped_column(String(32))
    #: Human-readable summary of what was learned.
    summary: Mapped[str | None] = mapped_column(Text)
    #: Machine-readable rules, e.g. {"operator": "ts_zscore", "avg_sharpe_lift": 0.4}.
    rules: Mapped[dict[str, Any] | None] = mapped_column(JSON, server_default="{}")
    #: Supporting evidence: expression snippets, metric ranges, etc.
    evidence: Mapped[dict[str, Any] | None] = mapped_column(JSON, server_default="{}")

    #: Which researcher or prompt produced the batch this insight came from (antecedent strategy).
    researcher_id: Mapped[int | None] = mapped_column(
        ForeignKey("research_researcher.id", ondelete="SET NULL")
    )
    prompt_strategy_id: Mapped[int | None] = mapped_column(
        ForeignKey("prompt_strategy.id", ondelete="SET NULL")
    )
    #: Resulting strategy evolved from this insight cycle.
    evolved_strategy_id: Mapped[int | None] = mapped_column(
        ForeignKey("prompt_strategy.id", ondelete="SET NULL"), nullable=True
    )

    #: P1-01 Reflection metadata & execution tracking
    model_ref: Mapped[str | None] = mapped_column(String(128), nullable=True)
    execution_status: Mapped[str | None] = mapped_column(String(32), nullable=True)
    is_fallback: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    evidence_candidate_ids: Mapped[list[int] | None] = mapped_column(JSON, nullable=True)
    structured_output: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    error_details: Mapped[str | None] = mapped_column(Text, nullable=True)

    #: Batch-level statistics at the time of this insight.
    batch_size: Mapped[int | None] = mapped_column(Integer)
    batch_avg_sharpe: Mapped[float | None] = mapped_column(Float)
    batch_best_sharpe: Mapped[float | None] = mapped_column(Float)
    batch_valid_rate: Mapped[float | None] = mapped_column(Float)

    #: Whether this insight was applied (used to update a prompt) or only recorded.
    applied: Mapped[bool | None] = mapped_column(default=False, server_default="0")
    created_at: Mapped[datetime] = mapped_column(UtcDateTime, default=utcnow)

    __table_args__ = (
        Index("ix_insight_session_gen", "session_id", "generation"),
    )


class LeaderboardSnapshot(Base):
    """A periodic snapshot of leaderboard rankings.

    Taken after each learning cycle so the dashboard can plot ranking trends over time.
    The ``dimension`` says what is being ranked (model, prompt, alpha, experiment), and
    ``entries`` is the ordered list of scored items at that moment.

    Not a live view: the service computes rankings on demand. These rows are the archive
    that lets a chart show "model X climbed from rank 4 to rank 1 over five generations".
    """

    __tablename__ = "leaderboard_snapshot"

    id: Mapped[int] = mapped_column(primary_key=True)
    session_id: Mapped[int | None] = mapped_column(
        ForeignKey("research_session.id", ondelete="CASCADE"), index=True
    )
    generation: Mapped[int | None] = mapped_column(Integer)

    #: What is ranked: "model", "prompt", "alpha", "experiment".
    dimension: Mapped[str | None] = mapped_column(String(32))
    #: Ordered list of {ref, rank, score, metrics...} dictionaries.
    entries: Mapped[list[Any] | None] = mapped_column(JSON, server_default="[]")

    snapshot_at: Mapped[datetime | None] = mapped_column(UtcDateTime, default=utcnow)

    __table_args__ = (
        Index("ix_leaderboard_session_dim", "session_id", "dimension"),
    )


class DatasetAllocationState(Base):
    """Persisted allocation sequence tracking for autonomous dataset exploration.

    Guarantees balanced, non-starving dataset exploration across process restarts
    and concurrent researchers without relying on volatile in-memory counters.
    """

    __tablename__ = "dataset_allocation_state"

    id: Mapped[int] = mapped_column(primary_key=True)
    session_id: Mapped[int | None] = mapped_column(Integer, index=True, nullable=True)
    scope_key: Mapped[str] = mapped_column(String(64), index=True, nullable=False)
    allocation_count: Mapped[int] = mapped_column(Integer, default=0, server_default="0", nullable=False)
    last_dataset_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    updated_at: Mapped[datetime] = mapped_column(UtcDateTime, default=utcnow, onupdate=utcnow)

    __table_args__ = (
        UniqueConstraint("session_id", "scope_key", name="uq_dataset_alloc_session_scope"),
        Index("ix_dataset_alloc_session_scope", "session_id", "scope_key"),
    )


class OperatorEfficacyRecord(Base):
    """P1-09: Persisted controlled operator attribution and shrinkage efficacy record.

    Stores deconfounded, shrinkage-regularized statistical efficacy metrics for Fast Expression
    operators across research cohorts, supporting empirical prompt evolution and learning loops.
    """

    __tablename__ = "operator_efficacy_record"

    id: Mapped[int] = mapped_column(primary_key=True)
    session_id: Mapped[int | None] = mapped_column(
        ForeignKey("research_session.id", ondelete="CASCADE"), index=True, nullable=True
    )
    generation: Mapped[int | None] = mapped_column(Integer, nullable=True)
    operator_name: Mapped[str] = mapped_column(String(64), index=True, nullable=False)
    category: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)

    sample_count: Mapped[int] = mapped_column(Integer, default=0, server_default="0", nullable=False)
    winner_count: Mapped[int] = mapped_column(Integer, default=0, server_default="0", nullable=False)
    loser_count: Mapped[int] = mapped_column(Integer, default=0, server_default="0", nullable=False)

    raw_avg_sharpe: Mapped[float] = mapped_column(Float, default=0.0, server_default="0.0", nullable=False)
    baseline_sharpe: Mapped[float] = mapped_column(Float, default=0.0, server_default="0.0", nullable=False)
    observed_lift: Mapped[float] = mapped_column(Float, default=0.0, server_default="0.0", nullable=False)
    marginal_lift: Mapped[float] = mapped_column(Float, default=0.0, server_default="0.0", nullable=False)
    shrunk_lift: Mapped[float] = mapped_column(Float, default=0.0, server_default="0.0", nullable=False)

    std_error: Mapped[float] = mapped_column(Float, default=0.0, server_default="0.0", nullable=False)
    t_stat: Mapped[float | None] = mapped_column(Float, nullable=True)
    lower_confidence_bound: Mapped[float | None] = mapped_column(Float, nullable=True)
    upper_confidence_bound: Mapped[float | None] = mapped_column(Float, nullable=True)

    confidence_level: Mapped[str] = mapped_column(
        String(32), default="INSUFFICIENT_SAMPLE", server_default="INSUFFICIENT_SAMPLE", nullable=False
    )
    evidence_status: Mapped[str] = mapped_column(
        String(32), default="INCONCLUSIVE", server_default="INCONCLUSIVE", nullable=False
    )

    method_version: Mapped[str] = mapped_column(String(32), default="p1_09_v1", server_default="p1_09_v1", nullable=False)
    provenance: Mapped[str | None] = mapped_column(String(64), nullable=True)
    evaluation_window: Mapped[str | None] = mapped_column(String(64), nullable=True)
    co_occurring_operators: Mapped[list[Any] | None] = mapped_column(JSON, nullable=True)
    details: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)

    created_at: Mapped[datetime] = mapped_column(UtcDateTime, default=utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(UtcDateTime, default=utcnow, onupdate=utcnow, nullable=False)

    __table_args__ = (
        Index("ix_op_eff_session_cat_op", "session_id", "category", "operator_name"),
    )


class ResearchEpisode(Base):
    """P1-12: Persistent, structured researcher-specific episodic memory.

    Stores detailed experiment records (hypotheses, candidate expressions, execution/simulation
    funnel stage, non-holdout validation metrics, statistical evidence, failure categories, and lessons)
    scoped strictly to an individual researcher persona or promoted to shared knowledge.
    """

    __tablename__ = "research_episode"

    id: Mapped[int] = mapped_column(primary_key=True)
    session_id: Mapped[int] = mapped_column(
        ForeignKey("research_session.id", ondelete="CASCADE"), index=True, nullable=False
    )
    researcher_id: Mapped[int] = mapped_column(
        ForeignKey("research_researcher.id", ondelete="CASCADE"), index=True, nullable=False
    )
    candidate_id: Mapped[int | None] = mapped_column(
        ForeignKey("research_candidate.id", ondelete="SET NULL"), index=True, nullable=True
    )
    prompt_version_id: Mapped[int | None] = mapped_column(
        ForeignKey("prompt_version.id", ondelete="SET NULL"), index=True, nullable=True
    )

    # Context & Environment Metadata
    generation: Mapped[int] = mapped_column(Integer, default=0, server_default="0", nullable=False)
    strategy_focus: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    dataset_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    region: Mapped[str | None] = mapped_column(String(32), nullable=True)
    universe: Mapped[str | None] = mapped_column(String(32), nullable=True)
    delay: Mapped[int | None] = mapped_column(Integer, nullable=True)
    neutralization: Mapped[str | None] = mapped_column(String(32), nullable=True)
    decay: Mapped[int | None] = mapped_column(Integer, nullable=True)

    hypothesis_family: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    hypothesis: Mapped[str | None] = mapped_column(Text, nullable=True)
    rationale: Mapped[str | None] = mapped_column(Text, nullable=True)
    expression: Mapped[str | None] = mapped_column(Text, nullable=True)
    expression_hash: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)

    # Execution & Lifecycle
    funnel_stage: Mapped[str] = mapped_column(
        String(32), default="GENERATED", server_default="GENERATED", nullable=False
    )  # GENERATED, AST_VALID, PREFLIGHT_VALID, SIMULATED, EVALUATED
    outcome: Mapped[str] = mapped_column(
        String(32), default="PENDING", server_default="PENDING", nullable=False
    )  # STATISTICAL_WINNER, STATISTICAL_LOSER, PROVISIONAL, SYNTAX_ERROR, PREFLIGHT_ERROR, SIMULATION_ERROR, UNTESTED
    selection_tier: Mapped[str | None] = mapped_column(String(32), nullable=True)
    provenance: Mapped[str | None] = mapped_column(String(32), default="LOCAL_PROXY", nullable=True)
    brain_simulation_id: Mapped[str | None] = mapped_column(String(64), nullable=True)

    # Non-holdout Validation Performance & Sample Info (P0-03 Holdout Isolation)
    # Strictly in-sample and validation metrics only. NEVER holdout metrics.
    validation_sharpe: Mapped[float | None] = mapped_column(Float, nullable=True)
    validation_fitness: Mapped[float | None] = mapped_column(Float, nullable=True)
    validation_turnover: Mapped[float | None] = mapped_column(Float, nullable=True)
    validation_decay: Mapped[float | None] = mapped_column(Float, nullable=True)
    in_sample_sharpe: Mapped[float | None] = mapped_column(Float, nullable=True)
    t_years: Mapped[float | None] = mapped_column(Float, nullable=True)
    val_observations: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # Statistical Evidence
    sharpe_se: Mapped[float | None] = mapped_column(Float, nullable=True)
    t_stat: Mapped[float | None] = mapped_column(Float, nullable=True)
    p_value: Mapped[float | None] = mapped_column(Float, nullable=True)
    lcb95: Mapped[float | None] = mapped_column(Float, nullable=True)
    z_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    fdr_q_value: Mapped[float | None] = mapped_column(Float, nullable=True)
    is_statistically_significant: Mapped[bool] = mapped_column(
        Boolean, default=False, server_default="0", nullable=False
    )

    # Failure Classification & Diagnostic Learning
    failure_category: Mapped[str | None] = mapped_column(
        String(64), nullable=True
    )  # None, "SYNTAX", "OPERATOR_TYPE", "PREFLIGHT", "SIMULATION_EXECUTION", "NEGATIVE_SHARPE", "HIGH_TURNOVER", "SEVERE_DECAY"
    error_details: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    lessons_learned: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Versioning & Lineage
    catalog_version: Mapped[str | None] = mapped_column(String(64), nullable=True)
    method_version: Mapped[str] = mapped_column(
        String(32), default="p1_12_v1", server_default="p1_12_v1", nullable=False
    )

    # Privacy & Knowledge Sharing Boundary
    is_shared_knowledge: Mapped[bool] = mapped_column(
        Boolean, default=False, server_default="0", nullable=False, index=True
    )

    created_at: Mapped[datetime] = mapped_column(UtcDateTime, default=utcnow, nullable=False, index=True)
    updated_at: Mapped[datetime] = mapped_column(UtcDateTime, default=utcnow, onupdate=utcnow, nullable=False)

    __table_args__ = (
        Index("ix_res_ep_researcher_gen", "researcher_id", "generation"),
        Index("ix_res_ep_res_outcome", "researcher_id", "outcome"),
        Index("ix_res_ep_shared", "is_shared_knowledge", "session_id"),
        Index("ix_res_ep_expr_hash", "researcher_id", "expression_hash"),
    )


class PromptLifecycleEvent(Base):
    """P1-13: Immutable audit log of prompt lifecycle events and promotion decisions."""

    __tablename__ = "prompt_lifecycle_event"

    id: Mapped[int] = mapped_column(primary_key=True)
    session_id: Mapped[int | None] = mapped_column(Integer, index=True, nullable=True)
    category: Mapped[str] = mapped_column(String(64), index=True, nullable=False)
    strategy_id: Mapped[int] = mapped_column(Integer, index=True, nullable=False)
    prompt_version_id: Mapped[int | None] = mapped_column(Integer, index=True, nullable=True)
    target_strategy_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    target_version_id: Mapped[int | None] = mapped_column(Integer, nullable=True)

    event_type: Mapped[str] = mapped_column(
        String(32), index=True, nullable=False
    )  # "CREATED_EXPERIMENTAL", "BENCHMARKED", "PROMOTED", "REJECTED", "ROLLED_BACK"
    from_status: Mapped[str | None] = mapped_column(String(32), nullable=True)
    to_status: Mapped[str] = mapped_column(String(32), nullable=False)

    # Statistical Evaluation Evidence (P0-03: Strictly validation partition data)
    delta_sharpe: Mapped[float | None] = mapped_column(Float, nullable=True)
    delta_se: Mapped[float | None] = mapped_column(Float, nullable=True)
    lcb95_delta: Mapped[float | None] = mapped_column(Float, nullable=True)
    non_inferiority_margin: Mapped[float | None] = mapped_column(Float, nullable=True)
    non_inferiority_passed: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    ast_valid_rate: Mapped[float | None] = mapped_column(Float, nullable=True)
    token_usage_avg: Mapped[float | None] = mapped_column(Float, nullable=True)
    sample_size: Mapped[int | None] = mapped_column(Integer, nullable=True)
    reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    details: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)

    created_at: Mapped[datetime] = mapped_column(UtcDateTime, default=utcnow, nullable=False, index=True)

    __table_args__ = (
        Index("ix_prompt_life_sess_cat", "session_id", "category"),
        Index("ix_prompt_life_event_time", "event_type", "created_at"),
    )

