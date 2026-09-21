# Changelog

Notable changes to `relore`, using [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versions follow the milestone/patch policy in [AGENTS.md](AGENTS.md#one-version-both-ends).
The client and daemon must run the same version.

## [Unreleased]

### Fixed

- Expansion no longer replaces an explicit `--symbol` / `--file` / `--error` / `--test`
  filter with a derived term from the query text. Same-kind values are ORed, so the
  derived evidence stays on the text and filters-only legs instead of widening or
  dropping the caller's scope (#75). A kind the caller already scoped still counts toward
  ranking, so a skipped derived term keeps its signal-overlap weight.

## [0.3.17] - 2026-09-18

First PyPI release. Earlier versions were distributed from Git and deployed as
container images.

### Added

- Lightweight `relore` client for searching issue and pull-request history, reading
  threads, finding work in progress, and tracing code decisions with `why`.
- `relored` service for ingestion, indexing, and ranked retrieval, with Postgres
  support and SQLite for development.
- Code navigation over local checkouts and daemon-managed clones, including
  definitions, references, symbols, grep, and duplicate implementations.
- Author trust filters, source attribution, and an untrusted-content envelope for
  retrieved discussions.
- PyPI release workflow: release branches run checks and validate distributions;
  version tags publish through GitHub OIDC trusted publishing.
- Package metadata, `make build-release`, and a release guide.

### Improved

- Moved the README quick start into the CLI guide; the README links to deployment
  and usage instructions.
