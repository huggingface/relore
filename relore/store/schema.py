"""The portable core of the schema, as SQLAlchemy Core tables.

The SQL in the build plan section 4 is the specification; this is its rendering.
Core, not the ORM: section 5.1's guarantee -- a thread's documents written and its stale
ones deleted in one transaction -- is the central claim of this system and has to be
readable as SQL rather than inferred from a flush order.

**Everything here must behave identically on Postgres and SQLite.** The columns and
indexes that do not port -- ``documents.fts``, ``documents.embedding``, the ``pg_trgm``
indexes -- are deliberately absent from this file and are created by the Postgres-only
migration instead. See section 4.1.
"""

from __future__ import annotations

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    Column,
    ForeignKey,
    Index,
    Integer,
    MetaData,
    Table,
    Text,
    UniqueConstraint,
)

from relore.store.dialect import UTCDateTime, json_type, pk_type

metadata = MetaData()


def _fk(target: str) -> Column:
    return Column("thread_id", pk_type(), ForeignKey(target, ondelete="CASCADE"), nullable=False)


schema_migrations = Table(
    "schema_migrations",
    metadata,
    Column("version", Integer, primary_key=True, autoincrement=False),
    Column("applied_at", UTCDateTime, nullable=False),
)

threads = Table(
    "threads",
    metadata,
    Column("id", pk_type(), primary_key=True, autoincrement=True),
    Column("repo", Text, nullable=False),
    Column("github_number", Integer, nullable=False),
    Column("thread_type", Text, nullable=False),
    Column("work_kind", Text),  # section 9; NULL until the files pass runs
    Column("title", Text, nullable=False),
    Column("author", Text),
    Column("state", Text),
    Column("url", Text),
    Column("created_at", UTCDateTime),
    Column("updated_at", UTCDateTime),
    Column("closed_at", UTCDateTime),
    Column("merged_at", UTCDateTime),
    Column("github_node_id", Text),
    Column("indexed_at", UTCDateTime, nullable=False),
    Column("metadata", json_type(), nullable=False),
    CheckConstraint("thread_type IN ('issue','pr')", name="threads_type_ck"),
    UniqueConstraint("repo", "github_number", name="threads_repo_number_uq"),
)

# text[] has no SQLite equivalent, so labels are a child table (section 4.1).
thread_labels = Table(
    "thread_labels",
    metadata,
    _fk("threads.id"),
    Column("label", Text, nullable=False),
    UniqueConstraint("thread_id", "label", name="thread_labels_pk"),
)

documents = Table(
    "documents",
    metadata,
    Column("id", pk_type(), primary_key=True, autoincrement=True),
    _fk("threads.id"),
    # title|body|issue_comment|review|review_comment|commit_message
    Column("source_type", Text, nullable=False),
    Column("source_id", Text, nullable=False),
    Column("chunk_index", Integer, nullable=False, default=0),
    Column("author", Text),
    Column("author_assoc", Text),  # MEMBER|OWNER|COLLABORATOR|CONTRIBUTOR|NONE
    # Trust tier (section 6.2). Not a ranking prior -- a filter. Derived at derive time
    # from the stored payload plus repo_authority, so a rebuild reproduces it.
    Column("author_is_bot", Boolean, nullable=False, default=False),
    Column("trust", Text, nullable=False, default="reported"),
    Column("url", Text),
    Column("body_markdown", Text, nullable=False),  # verbatim, never rewritten
    Column("body_text", Text, nullable=False),  # normalized, feeds FTS
    Column("content_hash", Text, nullable=False),
    Column("chunking_version", Integer, nullable=False),
    Column("extractor_version", Integer, nullable=False),
    Column("embedding_model", Text),  # the vector itself is POSTGRES ONLY
    Column("github_created_at", UTCDateTime),
    Column("github_updated_at", UTCDateTime),
    Column("commit_sha", Text),  # the tree this document refers to; the lens reads it
    Column("enclosing_symbol", Text),  # derived by the lens (section 1)
    Column("metadata", json_type(), nullable=False),  # {path, line, diff_hunk, ...}
    CheckConstraint("trust IN ('authoritative','reported','machine')", name="documents_trust_ck"),
    UniqueConstraint(
        "thread_id", "source_type", "source_id", "chunk_index", name="documents_identity_uq"
    ),
)
Index("documents_thread_idx", documents.c.thread_id)
Index(
    "documents_symbol_idx",
    documents.c.enclosing_symbol,
    sqlite_where=documents.c.enclosing_symbol.isnot(None),
    postgresql_where=documents.c.enclosing_symbol.isnot(None),
)

# Every object verbatim as fetched, before anything is derived from it. This is what
# makes re-chunking and re-extracting cost local CPU instead of another day of API
# budget (section 5.0).
raw_objects = Table(
    "raw_objects",
    metadata,
    Column("repo", Text, primary_key=True),
    # issue|pr|issue_comment|review|review_comment|pr_details
    Column("object_type", Text, primary_key=True),
    Column("object_id", Text, primary_key=True),
    Column("thread_number", Integer),
    Column("payload", json_type(), nullable=False),
    Column("fetched_at", UTCDateTime, nullable=False),
    Column("github_updated_at", UTCDateTime),
)
Index("raw_objects_thread_idx", raw_objects.c.repo, raw_objects.c.thread_number)

# Signal tables, kept separate from documents so an exact match can be weighted
# independently of prose. Nothing writes them until milestone 3.
# Section 13's cutoff, on the five tables a query filters and ranks *by* (see
# temporal-cutoff-map.md section 8). The earliest instant at which this thread is known to have
# carried this value: a `min` over the documents that produced it, and `merged_at` for the
# two things no document carries -- a merged pull request's changed-file list and its
# commits. Nullable, and null fails closed the way an undated document does, because
# "we do not know when this appeared" must not read as "it was always there".
#
# One row per (thread, value) is preserved, which is the whole reason this is a timestamp
# and not a `document_id`: `extract_signals` deduplicates across a thread's documents on
# purpose and section 6's overlap terms count rows, so splitting a row would move
# production ranking. This column is read *only* when `as_of` is set.
thread_files = Table(
    "thread_files",
    metadata,
    _fk("threads.id"),
    Column("path", Text, nullable=False),
    Column("change_type", Text),
    # WHERE the path came from, because the three sources differ in kind and a caller that
    # cannot tell them apart reads presence as evidence: `changed` is the per-PR pass's
    # definitive diff, `comment` is a path GitHub itself attached an inline review comment
    # to, and `mentioned` is a string somebody typed in prose -- which may be a bare
    # basename, may be a file this thread never touched, and on
    # `huggingface/transformers#39847` was both (huggingface/relore#17). `change_type`
    # cannot carry this: it is null for the first two of those as well as the third.
    Column("source", Text),
    Column("first_seen_at", UTCDateTime),
)
thread_symbols = Table(
    "thread_symbols",
    metadata,
    _fk("threads.id"),
    Column("symbol", Text, nullable=False),
    Column("path", Text),
    Column("symbol_type", Text),
    Column("first_seen_at", UTCDateTime),
)
thread_errors = Table(
    "thread_errors",
    metadata,
    _fk("threads.id"),
    Column("exception_type", Text),
    Column("message_norm", Text),
    Column("first_seen_at", UTCDateTime),
)
thread_tests = Table(
    "thread_tests",
    metadata,
    _fk("threads.id"),
    Column("test_id", Text, nullable=False),
    Column("test_function", Text),
    Column("first_seen_at", UTCDateTime),
)
thread_commits = Table(
    "thread_commits",
    metadata,
    _fk("threads.id"),
    Column("sha", Text, nullable=False),
    Column("message", Text),
    Column("first_seen_at", UTCDateTime),
)
# The relationship edge (section 5.3, section 13.3): "this pull request claims to close
# #N". Two columns for the target, and that is the whole design decision:
#
# * ``target_number`` is what the *source* thread says, so the edge is derivable from that
#   thread alone -- which means the poll path fills it and it stays current. The plan had
#   a not-null foreign key on both ends, which made an edge unstorable until its target
#   was indexed and therefore made a corpus-wide pass a prerequisite for the *query*.
# * ``target_thread_id`` is the resolution, filled when the target is in the index. It is
#   nullable because the claim is a fact about the source and the resolution is not: an
#   agent asking "is someone already fixing #48630?" is answered by the number, and only
#   ranking (section 6's ``w_rel``) needs the join.
#
# The plan's ``metadata jsonb`` is not here: there is nothing to put in it, and a column
# nothing writes is the mistake section 5.3 declines for keywords.
thread_links = Table(
    "thread_links",
    metadata,
    _fk("threads.id"),
    Column("relationship", Text, nullable=False),  # closes | ...
    Column("target_number", Integer, nullable=False),
    Column(
        "target_thread_id", pk_type(), ForeignKey("threads.id", ondelete="CASCADE"), nullable=True
    ),
)
for _t in (
    thread_files,
    thread_symbols,
    thread_errors,
    thread_tests,
    thread_commits,
    thread_links,
):
    Index(f"{_t.name}_thread_idx", _t.c.thread_id)
# The reverse direction is the one an agent asks for: "which threads claim to close this
# number" (`relore inflight`).
Index("thread_links_target_idx", thread_links.c.target_number)

# The rename chain (section 1), derived from git history rather than any API payload.
path_aliases = Table(
    "path_aliases",
    metadata,
    Column("repo", Text, nullable=False),
    Column("old_path", Text, nullable=False),
    Column("new_path", Text, nullable=False),
    Column("commit_sha", Text),
    Column("renamed_at", UTCDateTime),
    UniqueConstraint("repo", "old_path", "new_path", name="path_aliases_uq"),
)

precedents = Table(
    "precedents",
    metadata,
    Column("id", pk_type(), primary_key=True, autoincrement=True),
    Column("repo", Text, nullable=False),
    Column("kind", Text, nullable=False),  # bug_fix | feature | refactor | deprecation
    Column("issue_number", Integer),
    Column("pr_number", Integer),
    Column("title", Text),
    Column("problem_summary", Text, nullable=False),
    Column("outcome_summary", Text),
    Column("review_rounds", Integer),
    # 'linked' (GitHub said so) | 'inferred' (we matched). Surfaced to callers.
    Column("confidence", Text, nullable=False),
    Column("metadata", json_type(), nullable=False),
    UniqueConstraint("repo", "kind", "issue_number", "pr_number", name="precedents_uq"),
)

# What a precedent consisted of: one narrow table rather than four parallel arrays, and
# indexed rather than array-scanned -- `relore precedent --file ...` filters on it.
precedent_signals = Table(
    "precedent_signals",
    metadata,
    Column(
        "precedent_id",
        pk_type(),
        ForeignKey("precedents.id", ondelete="CASCADE"),
        nullable=False,
    ),
    Column("kind", Text, nullable=False),  # file | symbol | error_token | test_id
    Column("value", Text, nullable=False),
    UniqueConstraint("precedent_id", "kind", "value", name="precedent_signals_pk"),
)
Index("precedent_signals_lookup", precedent_signals.c.kind, precedent_signals.c.value)

sync_state = Table(
    "sync_state",
    metadata,
    Column("repo", Text, primary_key=True),
    # threads|issue_comments|pr_comments|files|sweep|authority|sample:issue|sample:pr
    # Free text, not a CHECK: `sample:*` (section 5.5) must be a *separate* mark from
    # `threads`, which the poll and the backfill share.
    Column("pass", Text, primary_key=True),
    Column("high_water", UTCDateTime),
    Column("cursor", Text),
    Column("last_run_at", UTCDateTime),
    Column("last_ok_at", UTCDateTime),
)

# What a sampled index does *not* contain (section 10). `sync_state.high_water` records
# where a pass has reached, which on a sampled corpus reads as "covered" for years it
# never walked -- so the floor is recorded here instead, per thread type, because the
# issue and pull-request windows are chosen separately and cost differently. No row means
# no declared floor: a full backfill, or a repository nobody has sampled.
# Section 14.1, settled 2026-09-09: **keep versions.** `index_thread` is otherwise
# last-write-wins, and people materially rewrite issue bodies -- the motivating
# deployment's per-run triage issues are refreshed in place, so their current body is not a
# record of what was dispatched.
#
# This is the one place the plan deliberately writes rows nothing reads yet, and section
# 5.3 explains why that is normally refused (keywords "would write rows nothing reads").
# The asymmetry is the whole argument: a keyword can be recomputed from `raw_objects` at
# any time, and a superseded comment body **cannot be reconstructed at all** once GitHub
# has served the new one and section 5.1 has replaced the row. Cheap now, impossible later.
#
# Append-only: nothing updates or deletes a row here except a thread's own cascade.
documents_history = Table(
    "documents_history",
    metadata,
    Column("id", pk_type(), primary_key=True, autoincrement=True),
    _fk("threads.id"),
    Column("source_type", Text, nullable=False),
    Column("source_id", Text, nullable=False),
    Column("chunk_index", Integer, nullable=False, server_default="0"),
    # The text as it stood, already redacted: this is copied from `documents`, never
    # re-read from `raw_objects`, so section 11.3's promise that a rebuild cannot un-redact
    # holds for the history too. A secret kept out of `body_text` cannot arrive here.
    Column("body_markdown", Text, nullable=False),
    Column("body_text", Text, nullable=False),
    Column("content_hash", Text, nullable=False),
    Column("author", Text),
    Column("github_updated_at", UTCDateTime),
    #: When *we* replaced it, which is not when the author edited it. GitHub does not tell
    #: us the latter for a body that was rewritten between two polls.
    Column("superseded_at", UTCDateTime, nullable=False),
    Column("reason", Text, nullable=False),  # edited | deleted
    CheckConstraint("reason IN ('edited','deleted')", name="documents_history_reason_ck"),
    Index(
        "documents_history_doc_idx",
        "thread_id",
        "source_type",
        "source_id",
        "chunk_index",
    ),
)

repo_sample = Table(
    "repo_sample",
    metadata,
    Column("repo", Text, primary_key=True),
    Column("thread_type", Text, primary_key=True),
    Column("indexed_from", UTCDateTime, nullable=False),
    Column("selector", Text, nullable=False),  # re-runnable description of the filter
    Column("sampled_at", UTCDateTime, nullable=False),
    CheckConstraint("thread_type IN ('issue','pr')", name="repo_sample_type_ck"),
)

# Resolved write access per author (section 6.2). MEMBER is not authority and cannot be
# treated as either yes or no, so it is resolved once per author and stored.
repo_authority = Table(
    "repo_authority",
    metadata,
    Column("repo", Text, primary_key=True),
    Column("login", Text, primary_key=True),
    Column("permission", Text),  # admin|maintain|write|triage|read|none
    Column("source", Text, nullable=False),  # permission_api | merged_by | association
    Column("checked_at", UTCDateTime, nullable=False),
)
