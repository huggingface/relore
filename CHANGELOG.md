# Changelog

Notable changes to `relore`, using [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versions follow the milestone/patch policy in [AGENTS.md](AGENTS.md#one-version-both-ends).
The client and daemon must run the same version.

## [Unreleased]

### Improved

- `relore refs` parses only the files whose bytes contain the symbol, the prefilter
  `copies` and `symbol` have had since #45. On `huggingface/transformers` that is
  10.7 s → 0.6 s for a rare name, with byte-identical output (#83).
- `relore map` memoises its parse in a `.relore/` directory at the repository root:
  18.5 s → 1.0 s. Every file is still read and hashed on every call and only the parse
  is skipped, so an entry is fresh when its key equals the sha of the bytes on disk.
  Nothing consults a timestamp, and nothing asks git whether a file changed — git's
  "clean" is a stat comparison that `--assume-unchanged` is enough to defeat. No
  reference positions are stored. The directory ignores itself; `RELORE_NO_CACHE`
  switches the whole thing off.

### Added

- `benchmarks/probes/code_lens.py`, a client-side benchmark for the offline verbs:
  speed against `git grep` and `grep -rn` across rare-to-ubiquitous symbols, plus a
  strict-subset accuracy check and an assertion that cached and uncached runs are
  byte-identical.

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
