# relore — GitHub project memory

**Every codebase has lore. Ask it, then check it against the code.**

<img src="docs/relore.png" alt="logo" width="200">

`relore` indexes a repository's complete issue and pull-request history and keeps a working
clone beside it, both served over one HTTP API. Three questions, in the order an agent hits
them:

**1. Is somebody already doing this?** `inflight <issue>` — every open claimant, its state
and its author, in one hop. The most expensive thing an agent does is patch something that
is already in review.

**2. Why is it like this?** Why a fallback cannot be removed, which approach was tried and
rejected, what a maintainer made the last five contributors change — none of it is in the
codebase. `thread`, `why PATH:LINE` and `search` return it at comment level with who said
it attached, so `--trust authoritative` keeps only what someone with write access settled
and machine authors are excluded by default. `search --symbol GemmaRotaryEmbedding` asks
what has been *said* about a function — a question `grep` cannot answer.

**3. Is that still true?** `grep`, `symbol` and `copies` run against that clone at HEAD,
server-side, no checkout on your side. A three-year-old review is a claim about code that
has moved since.

![Relore connects project history with current code to answer an agent's three questions.](docs/relore-diag.png)

Threads tell you what people decided; the code verbs tell you whether it was true. A
[contributor claim] confirmed by `grep` is stronger evidence than either alone — the
argument for this over a search box.

Six agents have run it cold on the same `transformers` bug, one per release, on `--help`
alone and with no memory of the runs before. They filed 39 issues against the tool; most
are fixed, each run checking the last one's from the outside.
[#8](https://github.com/huggingface/relore/issues/8) is the record.

## Get started

- [Deploy the service](docs/operations.md).
- [Use relore](docs/cli.md).

## What you get

| | |
| --- | --- |
| **Comment-level results** | the matching document with its author, trust tier, age and URL — not a thread number to go re-read |
| **Exact signals** | errors, files, symbols, test ids, shas, extracted at ingest and repeatable as filters |
| **Trust tiers** | maintainer / contributor / bot, as a filter rather than a weight |
| **Schema to join on** | "which threads touched this file", and a bug and its merged fix as one record — open pull requests carry file lists too, and a `--file` page names the ones it could not test |
| **The code, server-side** | `why PATH:LINE`, `grep`, `copies`, `symbol`, and `defs`/`refs` with `--repo`, against a working clone the daemon keeps |
| **Throughput** | every query is a Postgres query and makes no GitHub request; ten agents in parallel cost the same as one |

The trade is freshness: each thread records its last indexing visit; later GitHub changes
are unknown. Collector poll times do not guarantee freshness for every thread.
[Why not GitHub search](docs/why-not-github-search.md) has the measurements.

Running the daemon needs a Postgres and a GitHub token with `issues:read` +
`pull_requests:read` — never write. `deploy/` has a Helm chart and the scripts that are its
interface; setup is in [`docs/operations.md`](docs/operations.md).

## Docs

- [`docs/how-search-works.md`](docs/how-search-works.md) — what kind of index this is
  (lexical and ranked, no embeddings), and what it is better and worse at than `grep`.
- [`docs/cli.md`](docs/cli.md) — worked queries, and the empty results that are not faults.
- [`docs/why-not-github-search.md`](docs/why-not-github-search.md) — the comparison.
- [`docs/architecture.md`](docs/architecture.md) — what is indexed, what is deliberately
  not, the symbol lens, and language plugins.
- [`docs/agents.md`](docs/agents.md) — wiring it into a harness (not to be confused with
  `AGENTS.md` below, which is the contributor contract for this repository).
- [`docs/ranking.md`](docs/ranking.md) — the scoring model and what the benchmark says.
- [`docs/operations.md`](docs/operations.md) — backfill, poll, clones, tokens, serving.
- [`docs/security.md`](docs/security.md) — the properties and how each is enforced.
- [`docs/release.md`](docs/release.md) — release branches, tags, and PyPI publishing.
- [`CHANGELOG.md`](CHANGELOG.md) — notable changes by release.
- [`docs/prior-art.md`](docs/prior-art.md) — related work, and what was borrowed.
- [`AGENTS.md`](AGENTS.md) — the operating contract: the invariants with tests behind them.

## License

[Apache-2.0](LICENSE).
