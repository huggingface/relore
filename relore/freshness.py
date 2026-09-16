"""Freshness wording shared by the client and daemon; no server dependencies."""

COLLECTOR_NOTE = (
    "collector telemetry (not response freshness; use each thread's indexed_at): "
    "high-water is a source cursor; last-ok is collector completion"
)
COMMENT_PASS_NOTE = (
    "bulk comment collector; per-thread refreshes run via [threads] and do not update this row"
)
THREAD_STAMP_NOTE = "body and comments last indexed; later GitHub changes unknown"


def pass_note(name: str) -> str:
    return COMMENT_PASS_NOTE if name in {"issue_comments", "pr_comments"} else ""
