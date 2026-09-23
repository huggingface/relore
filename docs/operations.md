# Operations

Running the daemon needs a Postgres and a GitHub token with `issues:read` +
`pull_requests:read` — never write. `deploy/` has a Helm chart and the scripts that are its
interface.

## Install

```bash
GH=git+https://github.com/huggingface/relore   # no PyPI release yet; install from main
pip install $GH                        # client only
pip install "relore[postgres] @ $GH"   # the daemon
```

The client must be the same version as the daemon it talks to; if it is not, the daemon
refuses the request and says which end is behind. See [One version, both
ends](#one-version-both-ends).

## Index

```bash
export RELORE_DATABASE_URL=postgresql://localhost/relore   # or sqlite:///relore.db, dev only
export GITHUB_TOKEN=...                                    # issues:read + pull_requests:read
relored migrate
relored backfill --repo owner/name           # full history; resumable, ~a day
relored poll     --repo owner/name --interval 5m
relored sweep    --repo owner/name           # weekly: reconcile deletions
```

Run **`relored authority`** after backfilling an org-owned repository, or `--trust
authoritative` returns nothing: maintainers whose write access comes through a team report
as `MEMBER` and stay in `reported` until it does. Measured before that pass existed: zero
of 415 documents were `authoritative`.

`relored sample --repo … --since YYYY-MM-DD --kind issue|pr [--merged]` indexes a bounded
window instead of a history, fetching **each thread whole** — §10's evaluation set needs a
corpus, not a day of API budget. It is deliberately not `backfill --since`: that would walk
`/issues/comments?since=`, which filters individual comments and would index a thread that
moved inside the window without the older comments that explain it (§5.5). A sampled index
declares its floor and `relore status` prints it, because a recall number is only comparable
against a baseline restricted to the same window.

**`backfill --only files` is incremental.** It stages the merged PRs whose detail is not
staged yet, so a daily run costs what merged that day rather than the whole history — and
the poller stages no detail, so without it a PR first seen by polling carries no diff
stats, no `merged_by` and no reviews. `--refresh-details` re-stages everything, for when
the extraction changed rather than the corpus.

**It also walks open PRs, and walks them again every run.** A merged PR's diff is final, so
it is fetched once and skipped forever; an open one is still moving, so it is re-fetched.
That costs what is currently in flight — a small, self-limiting set — and it is what lets
`search --file` answer "is somebody already touching this path", which is most of what that
flag is asked. Closed-unmerged PRs are in neither set: they did not ship and are not in
flight. `--merged-only` restores the older behaviour for a run that wants precedent alone.

`fetch` and `derive` are separate on purpose: raw payloads are staged, so improving an
extractor and re-deriving costs minutes of local CPU instead of another day of API budget.
`poll` does both for the threads that moved, so the split is invisible in steady state.

## The working clone (issue #7)

`symbol`, `grep`, `copies` and `defs`/`refs --repo` read a clone per repository, checked
out at HEAD and complete rather than filtered, so `why`'s blame never fetches mid-query.
Create it **where `serve` runs** — that is the process answering them:

```bash
export RELORE_CLONE_ROOT=/var/lib/ghlore/clones   # default; a pod needs a volume for it
export RELORE_CLONE_REFRESH=1h                    # serve re-runs the fetch on this timer
relored clone --repo owner/name                   # create; re-running it is the refresh
```

Without one those verbs answer 503 with a sentence naming this command, and nothing else
degrades — the index does not depend on a checkout.

**`RELORE_CLONE_REFRESH` is off by default**, and the refresh lives inside `serve` rather
than in a CronJob: the clones volume is ReadWriteOnce and that process holds it, and a
laptop has nothing to schedule with. It refreshes what is already cloned and never creates
one — a repository added with `relored clone` joins on the next tick. Unset, HEAD ages
until somebody re-runs the command.

## Serve

One token per consumer, each scoped to repositories — `secret:repos[:scopes]`, where
`repos` is a comma list or `*` and `scopes` adds `label` for the UI:

```bash
export RELORE_API_TOKENS="tok_agent:owner/name tok_ui:*:label"
export RELORE_LABELS_PATH=./labels.jsonl     # enables the UI's relevance labelling
relored serve --port 8080 --host 0.0.0.0
```

Or no tokens at all, when a private network is the perimeter: `serve --trust-network` binds
wide with no tokens, which is how the deployment runs. Without either, **serve refuses to
bind anything but loopback** — the default fails closed — and it refuses a SQLite index
outright unless you pass `--allow-sqlite`.

## Point a client at it

`relore` defaults to the deployment, `https://ghlore.huggingface.tech`, which is reachable
over the private network only and needs no token — the network is the perimeter there, and
responses are read-only over public GitHub history. Set `RELORE_API` to use your own daemon
instead.

```bash
export RELORE_API=http://localhost:8080
export RELORE_TOKEN=tok_agent                # only if that daemon requires one
```

`--json` for machine-readable output, `--compact` to trim snippets. **No results is exit 0
with an empty result** — never nonzero, so an agent cannot mistake "nothing in the index"
for "the tool is broken".

## One version, both ends

The client and the daemon must be the **same** version, and they enforce it. Every request
to `/api/v1` declares its client version in `x-relore-client`; a daemon that reads anything
else answers `426 Upgrade Required` and one sentence saying which end is behind. Every
response carries `x-relore-version`, so a client whose daemon is too old to enforce that
catches the same mismatch from its side.

The reason is the failure mode, not tidiness: an old client asking a new daemon gets an
answer where every field it knows about is present and correct, and whatever the newer
version would have added is simply absent. Well-formed, plausible, silently incomplete — the shape of every defect in §13.3.
A refusal is legible; a short answer is not.

The price is that `relore/__init__.py`'s `__version__` is a contract rather than a label:

* it is the **only** place the version is written (`pyproject.toml` reads it from there);
* **a pull request leaves it alone**, however client-visible the change — the bump belongs
  to the release, not to the change that prompted it; and
* **bump and deploy are one operation.** A bump merged to `main` and not shipped breaks
  every client installed after the merge, and they will be told the deployment is behind.

`relore --version`, `relored --version`, `relore status` and the daemon's page all print
it, so "which version is this" never needs a guess. What the mismatch looks like from the
CLI, and which end to fix, is in [`cli.md`](cli.md).

## Benchmark

```bash
# §10's benchmark: mine candidates, fold the UI's labels in, score the baselines
relored mine  --repo owner/name --out benchmarks/owner-name.jsonl
relored judge --set benchmarks/owner-name.jsonl --labels labels.jsonl
relored bench --repo owner/name --set benchmarks/owner-name.jsonl \
  --system index --system index+expand --system github
```

`--system grep --clone <checkout>` adds the second baseline; it reads a working tree, so
point it at a detached worktree of `origin/main`, not a branch someone is working on.
Results and caveats: [`ranking.md`](ranking.md).

### A temporal run (§13)

An evaluation that can retrieve the comment written *after* the bug was fixed measures
nothing. `--before` scores the index as it stood at an instant; `--exclude-thread`
withholds what a cutoff cannot express — a thread older than the cutoff that is still the
answer:

```bash
relored bench --repo owner/name --set benchmarks/owner-name.jsonl \
  --system index --before 2026-03-01T00:00:00Z --exclude-thread 41302
```

The timestamp must carry a zone; a naive one is refused at the flag. `github` and `grep`
answer from outside this index and cannot be held to a cutoff, so asking for either
alongside one is refused — score them in a separate, untimed run.

The report prints how many hits the signal-table post-filter dropped, and what it does not
fix: the five `thread_*` tables carry no timestamp, so §6's overlap terms scored rows
written after the cutoff. **Filtering is corrected; ordering is not** until those tables
carry a `first_seen_at`. That belongs beside any number such a run produced.

### The corpus a baseline reads

```bash
relored export-corpus --repo owner/name --before 2026-03-01T00:00:00Z \
  --exclude-thread 41302 --out corpus/task-41302.jsonl
```

JSONL, header first, one line per document. Local and read-only. It selects with the same
predicate `search` applies, so a baseline built from this file reads the documents the
index read rather than a near-miss of them. The header carries the cutoff, the withheld
threads, a sha256 of the body (two exports of one corpus hash the same) and the limits the
corpus still has. Omitting `--before` exports the whole index — a valid control, recorded
as one.
