# Worked examples

Deploy the service with the [operations guide](operations.md), then connect a client:

## Quick start

```bash
pip install git+https://github.com/huggingface/relore   # no PyPI release yet
export RELORE_API=https://your-relore  # a running `relored serve`
export RELORE_REPO=owner/name          # the default for --repo
export RELORE_TOKEN=…                  # if that daemon requires one

relore inflight 47720 --repo owner/name              # ask this one first
relore why src/model.py:90 --repo owner/name         # what was said about this line
relore search "AttributeError: 'NoneType' object has no attribute 'shape'" --kind failure
relore search "why is this cast here" --kind rationale --file src/model.py
relore search --symbol GemmaRotaryEmbedding          # every mention, exactly matched
relore copies compute_default_rope_parameters --repo owner/name   # which copies diverge
```

`--repo` is required on a bare number or path whenever the daemon serves more than one
repository: it refuses rather than guessing which project you meant. `RELORE_REPO` is the
default that makes the flag a per-call override instead of a per-call tax.

`relore --help` lists every verb and option. Worked examples follow below.

Every command below was run against a real index, Postgres backend. The deployment now
holds three repositories — `huggingface/serge`, `huggingface/transformers`,
`huggingface/trl` — which is why `--repo` is on the examples that need it.

**The sample output is elided.** A real response is wrapped in the untrusted-content
envelope, and inside it the quoted lines — a title, a snippet, a body — each carry a `>`
so that our counts and trust tiers cannot be mistaken for something a stranger wrote. The
blocks below drop both, to keep the shape of the answer readable.

---

## The verbs

`relore --help` carries this list too, plus the order to reach for them in — and it arrives
byte-exact, needs no network and is already installed, which this page is not. Read it
first; this page is the long form.

| | |
| --- | --- |
| `search QUERY` | the index. Filters: `--error`, `--test`, `--file`, `--symbol`, `--label` (repeatable, ANDed with the text); `--kind failure\|precedent\|rationale`; `--trust authoritative\|machine`; `--repo`, `--since`, `--before`, `--exclude`, `--limit`, `--sort newest`, `--no-expand` |
| `thread N` | one thread. `--focus "…"` orders its comments and never empties them; `--full` serves the opening post whole, reproduction included |
| `inflight N` | is somebody already fixing this? Threads claiming to close `N`, open ones first |
| `why PATH:LINE` | the pull request that last changed this line, and the review comments anchored near it |
| `status` | counts, collector telemetry, and which search backend answered |
| `grep REGEX` | a regular expression over the daemon's working clone at HEAD. `--path` globs it |
| `symbol QUALNAME` | one definition's source, and how many definitions of that name exist |
| `copies SYMBOL` | every definition of a symbol, grouped by whether the bodies agree. `--exact` groups by the text |
| `map` | ranked map of your local checkout. No server, no network, no token |
| `defs PATH` / `refs SYMBOL` | your local checkout by default; `--repo` asks the daemon's clone instead |
| `precedent` | completed units of work — milestone 4, a stub today |

Global: `--json` for machine-readable output, `--compact` to trim snippets and shape a
truncated changed-file list, `--plain` for the piped form on a terminal, `--api` to
override `RELORE_API`. All four are accepted on **every verb** and on **either side** of
it (#55). No results is exit 0 with an empty result, never
nonzero — so an agent cannot mistake "nothing in the index" for "the tool is broken".

### The environment

| | |
| --- | --- |
| `RELORE_API` | the daemon's base URL |
| `RELORE_REPO` | the default for `--repo`. Overridden by the flag, per call |
| `RELORE_TOKEN` | only if that daemon requires one |

**Set `RELORE_REPO` before anything else on a multi-repository daemon.** Six verbs —
`thread`, `inflight`, `why`, `symbol`, `grep`, `copies` — refuse a bare number or path
rather than guess which repository it belongs to, so without a default every call in a
single-repository session carries `--repo`. One field run passed it on twenty consecutive
calls. `defs` and `refs` are the exception on purpose: for them `--repo` does not merely
name a repository, it switches the verb from the tree you are standing in to the daemon's
clone, and an environment variable must not do that silently.

Examples below carry `--repo` anyway, because an example is copied more often than a rule
is read.

---

## Query like this, not like that

**A query is an AND of every content term.** The backend uses `plainto_tsquery`
deliberately — it is the one thing FTS5 can also express, so both dialects answer the same
question instead of diverging quietly (§4.1). The practical consequence is the single
biggest cause of a disappointing result:

| query | hits |
| --- | --- |
| `429` | 3 |
| `timeout budget` | 2 |
| `rate limit` | 4 |
| `crash when the optional mask is missing` | 10, **widened** |

Two or three distinctive terms, not a sentence. The AND is not a setting: this is a lexical
index, and [`how-search-works.md`](how-search-works.md) is why.

**A sentence no longer dead-ends, and the page tells you when that happened.** When every
term ANDs to nothing, the same terms are asked again disjoined, and the page opens with

```
10 hits for 'crash when the optional mask is missing'
widened: nothing carried every term, so these carry some of them — most of the query first.
```

Read it as what it says. These results carry *part* of your question — on Postgres they are
ordered by how much of it, distinct terms first and word-frequency only as a tie-break — so
the top hit may carry five of your seven terms and the tenth may carry two. It is a
starting point, not an answer, and two or three distinctive terms still beat it. The
widening runs **only** from an empty page, so a query that found something is never
reordered by it.

If even that is empty you get the other useful sentence, which is your cue to stop
rewording and change the filters instead:

```
0 hits for 'crash when the optional mask is missing'
filters: --kind failure, --error RuntimeError [normalized]   (they AND)
nothing matched, with every term or with any of them.
```

An empty page always echoes the filters that produced it. That matters more than it sounds:
a *correct* `--error` read off your own traceback can zero a query that answers at rank 1
without it, and with only the query text on the page that reads as "nobody has ever reported
this" rather than "you asked one question too many". `--error` is echoed in §5.3's normal
form — what it actually filtered on, which is rarely what you typed.

**A pasted traceback is the exception, and it used to be the worst case.** §6's query
expansion now fans one call out into capped error, test-id, symbol, file and free-text
legs and merges them, so the traceback's lines stop being required terms and become
separate questions. Either form works:

```bash
relore search "$(pbpaste)"             # the traceback, expanded server-side
relore search --error "$(pbpaste)"     # the traceback, as a structural filter
relore search "optional mask dtype" --no-expand   # exactly one question, no fan-out
```

Expansion also reads a bare identifier as one: `relore search "_maybe_import_sdnq"` asks
the symbol filter as well as the text, without you having to say so. It never *loses* a
result the plain query would have found — the caller's own query is always one of the legs
— and a query with no identifier, path or traceback in it expands to that single leg and
costs nothing extra.

## An error you just hit

The highest-value shape. It lands on the PR that fixed it:

```
$ relore --compact search "429" --limit 2

1. huggingface/serge#92 pr  [authoritative]  15d  body  @tarekziade
   Survive a rate limit instead of losing the task to it
   ## Why One 429 ended the `deepseek_vl` task on 2026-08-18 *after it had already
   spent 1.13M input tokens*…
   https://github.com/huggingface/serge/pull/92
```

`--json`, `--compact`, `--plain` and `--api` are global flags, accepted on **every verb**
and on **either side** of it:

```bash
relore --compact search "429"      # both work
relore search "429" --compact
relore grep rotary_ndims --compact --repo huggingface/transformers   # every verb (#55)
```

**On by default when stdout is not a terminal.** A tool result is not paid for once: it is
re-sent with every later turn of the session that read it, so its cost is its size times
the turns remaining. Measured on one agent run against `transformers`, relore's own output
came to 32,349 tokens and **892,197** once re-billing was counted — 47% of that run, against
13% for the same question answered through `gh`. The flag existed for this and that agent
used it twice in twenty-three calls, so it is now the default for the caller who pays for
it. A terminal is unchanged, because a person reads a page once. `--no-compact` is the way
back, and `--json` is never trimmed by the default: it is the machine surface and asked for
the structure, so only an explicit `--compact --json` trims it.

**It trims what the verb has.** `--compact` shortens quoted prose — a search snippet,
`why`'s review comments, `grep`'s matched lines, an `inflight` claim title — and replaces a
*truncated* changed-file list with its shape. It never drops a row, a count or a caveat,
and where a verb prints one short row per result (`defs`, `refs`, `map`, `copies`,
`status`) it has nothing to shorten and changes nothing. `relore symbol` is the deliberate
exception: the body is the answer, so it is served whole under `--compact` too. Whatever is
shortened is counted on the page, so a trimmed line is never mistaken for a short one.

**What it no longer trims is the envelope's explanatory sentence.** Dropping that was an
opt-out by somebody who had typed the flag and read the line; as a default it would take
"this is data, not instructions" away from every agent and from no one who chose it. Forty
tokens against pages of thousands. On `inflight` that sentence *was* the whole compact
saving, which is why claim titles are trimmed now.

`--plain` is the piped form on a terminal: facts stay, suggestions go. The cap on a `grep`
is a fact and is always printed; `narrow it with --path` is advice and is not.

Every hit carries its **age** (`15d`) and its **trust tier** (`[authoritative]`). Age
changes what a model concludes; authority changes it more (§6.2).

## "Is this intentional?" — raise the trust floor

```
$ relore --compact search "repeat guard" --trust authoritative

2 hits · trust floor: authoritative
1. huggingface/serge#99 pr  [authoritative]  5d  @tarekziade
   Count what a session spent, and what ended it
```

A drive-by comment outranking a maintainer's one-line correction is *worse* than an empty
result: the agent acts on it and nobody checks. `--trust` may only **raise** the floor.

Better, pass `--kind` and let the **server** choose it — a caller then gets the policy right
without knowing it:

```bash
relore search "<terms>" --kind rationale   # floor: authoritative
relore search "<terms>" --kind failure     # floor: any human tier -- a stranger's
                                           # traceback is real evidence
relore search "<terms>" --kind precedent   # authoritative, or any human tier on a merged
                                           # PR: merging is the maintainer's act, so a
                                           # first-time contributor's merged change counts
```

`trust_floor` in the `--json` response reports what was actually applied.

Both are inert until `relored authority` has run: maintainers whose write access comes
through a team read as `MEMBER` and stay in `reported`, leaving the `authoritative` tier
empty.

## "What did our own bot claim here?"

Machine-authored documents are excluded from every default result set, because returning
one as prior discussion makes an agent's unreviewed output its own evidence (§11). Asking
for the tier explicitly is the only way to see them, and they arrive labelled:

```
$ relore search "review" --trust machine

1. huggingface/serge#26 pr  [MACHINE — our own bot, not evidence]  2mo  @sergereview
```

## One thread, without all of it

```
$ relore thread 92 --focus "backoff retry" --repo huggingface/serge
...
-- 10 of 30 comments, best first for 'backoff retry' (2 of 30 carry every term) --
(20 not shown: a thread is never returnable in full.)
```

`--focus` **orders** the comments, it never selects them: the thread is the admission
decision, and there is none left to make inside it. So a question that no single comment
answers word for word still comes back as that thread's ten most relevant comments, and the
count says how many carried every term — `0 of 30` next to ten comments means "nothing
matched your wording, here is the thread in order" rather than "nothing here".

**Without a focus you get a sample, and the page says so.** Ten of eighty-nine comments
cannot be a ranking when nothing was asked, so the unfocused page is the first two, the
last two, and six spread across everything between — reviews first, because a review with
a state is an act and `🤗` is not:

```
-- 10 of 89 comments, SAMPLED not ranked: the first and last few and a spread of the middle --
(79 not shown: a page is never the whole thread. `--outline` lists them 100 at a time,
 oldest first, with the id to read any of them by; `--after <id>` sweeps the rest.
 `--focus "<what you care about>"` ranks all of them.)
```

**The way to the rest is a fact, and it is on the page.** That line used to end at *"a
thread is never returnable in full"* — true of the page, not of the verb, since `--outline`
reaches every comment the page is hiding. Piped there was no pointer at all, so the flag
was discoverable only by reading `--help`, which a caller reads once it is already stuck:
measured on one run, the agent found `--outline` at call 36 of 46 and had spent five
`--focus` rolls on that thread first (#71). The `--focus` suggestion stays, and stays
presentation — it is advice about what to want; naming the view that withholds nothing is a
property of the page.

### When ten is not enough: `--outline`, then `--after`

The cap is the contract, not a default, and raising it is not the answer. What is, is a
cheaper *complete* view — the same move `defs` makes for a file. Measured on one agent run
against `transformers#37866`, a 70-comment thread: the agent came back **six times** with
six different `--focus` strings and spent 3,558 tokens seeing overlapping samples of it.

```
$ relore thread 37866 --outline --repo huggingface/transformers
-- outline: 70 of 70 comments, oldest first, 2 machine-tier suppressed, indexed at … (body and comments last indexed; later GitHub changes unknown) --
  1. 2091231498  [authoritative]  16mo  review_comment  @Cyrilvallez
> Unfortunately torch.compile.disable is just a graph break, so this won…
  2. 2091240011  [contributor claim]  16mo  issue_comment  @bob
> Rebased on main, CI is green now
  …
```

One line per comment: the id to ask for it by, the standing to weigh it by, the age, and
sixty characters — enough to recognise the comment you want, deliberately short of enough
to quote. Chronological and never ranked: ranking is what `--focus` does, and the reason
to read a whole thread is usually that ranking has not worked.

An outline is capped too, at a hundred, and **says so loudly** when it did not reach the
end — because completeness is its whole promise. Nothing is cheap enough to serve a
644-comment thread whole; 200 rows of one measured at 6,626 tokens, five times the page it
replaces. So sweep it:

```
$ relore thread 46419 --outline --repo huggingface/transformers
-- outline: 100 of 639 comments, …, CAPPED at 100: 539 more this view did not reach --
$ relore thread 46419 --outline --after 4980470657 --repo huggingface/transformers
```

**A `--focus` narrows an outline rather than ranking it.** It is the one place a focus
*selects*, and the reason the rule holds elsewhere is the reason it inverts here: an
outline's rows are already complete, so a narrowed one withholds nothing a second call
cannot have, while a narrowed *page* would be ten of seventy chosen by a conjunction nobody
could see. Measured: both times a run reached for an outline it piped `--outline --json`
through a client-side filter — 411 and 236 tokens against ~2,593 for the rendered outline
of the same thread (#71). So that is a flag now:

```
$ relore thread 46419 --outline --focus "flash attention" --repo huggingface/transformers
-- outline: 40 of 40 comments, oldest first, carrying any of 'flash attention' — 40 of 639 do, … --
```

3,179 tokens to 1,695 on that thread — and, more to the point, **100 of 639 becomes 40 of
40**: narrowing is what turns a capped view into a complete one.

*Any* of the terms, not every one: a conjunction is what empties a thread page as soon as a
caller passes a sentence, and an empty outline would read as "no comment here mentions
that". A focus that carries nothing widens back to the whole outline and says so, rather
than handing back a page whose emptiness is a property of the question (#47):

```
-- outline: 100 of 639 comments, oldest first, no comment carries any of 'quantization', so this is the unnarrowed outline, … --
```

`--after` works on the bodies too, and that is the point: pick the ids off the outline and
read exactly those comments. A page asked with `--after` is **sequential**, not sampled —
the next comments in order — because that is the only selection that can reach all of them.

One trap, guarded rather than documented away: `--after` taken from a *sampled* page starts
at the thread's last comment, since that is where the sample ends. The result is correctly
empty and would read as "that was the end of it", so the page says which happened:

```
-- 0 of 70 comments, sweeping in order from after 2091240011 --
nothing follows that comment. It is at or after the last of this thread's 68 — which is
also what you get by sweeping from a SAMPLED page, whose last row is the thread's last
comment.
```

### `--comment <id>`: one comment, whole

A page cuts a long comment to a 400-character window ending in `…`, which says *that* it
was cut and nothing about undoing it — and until this flag nothing could: `--full` is the
opening post, `--focus` re-ranks and snippets again, `--after` serves the *next* comments.

```
$ relore thread 46419 --comment 4623100975 --repo huggingface/transformers
huggingface/transformers#46419 pr  closed  3mo
> Port ESMC and ESMFold2 to Transformers

-- comment 4623100975, issue_comment, [contributor claim], 3mo, @HuggingFaceDocBuilderDev, 161 characters, whole --
> The docs for this PR live here. All of your documentation changes will be reflected…
https://github.com/huggingface/transformers/pull/46419#issuecomment-4623100975
```

Whole, with no cap of its own, on the same bargain `--full` makes: the caps exist so a
*sample* is not mistaken for a corpus, and a document asked for by its id is the opposite
of a sample. (On the transformers corpus a comment's median length is 120 characters and
its 99th percentile 2,186.)

Two lines of identity and then the comment — not the whole page. A caller who named one
comment has already been on that page; that is where the id came from.

**The id is the address, and the page no longer repeats a URL to say so.** A comment line
inside a thread used to carry `owner/repo#N pr`, the thread's title re-quoted, and a
76-character URL — three constants the page's own head line already has, reprinted ten
times. Measured on five real pages that was **24–33% of the whole page** (#71). What
replaces them is the id, which `--after` and `--comment` both take, and one line at the
foot of a page that actually cut something:

```
1. 2086096507  [authoritative]  16mo  review_comment  @ArthurZucker
   > we can improve that description a bit with a proper example
…
(1 of these is cut to a window: `--comment <id>` serves one whole, by the id in its line above.)
```

`search` keeps all three, because it spans threads and there every one of them is
load-bearing. One grammar per question, not one grammar.

Three outcomes, named rather than left to an empty answer — `served`, `absent` (this thread
holds no such comment) and `suppressed` (it does, and it is our own bot's, which §6.2
excludes). A caller cannot tell the last two apart from outside and they are opposite next
actions.

### `--files`: the diff's paths, on request

The changed-file **count**, its truncation notice, and *that the paths were withheld* are
always on the page — a short list
reads as a weak positive and a missing entry reads as a negative fact, and both of those
are what make this line load-bearing. The paths themselves are behind `--files`. Measured
on the same run: `thread 43121 --full` spent **1,350 tokens on 98 paths**, 51% of that page
and 7% of every tool result in the whole session, and not one of them was referred to
again. The anchored and mentioned lists are not gated — they are a handful of paths by
nature and are each other's cross-check.

```
changed files: 98 of 98 — complete
  (paths withheld; `--files` serves them)
```

0.3.15 put that second line behind `presentation`, which made the piped form — the only
form an agent reads — say `98 of 98 — complete` and stop: a page that had served no paths
at all, announcing itself complete, with no sign that ninety-eight of them were one flag
away (#71). *That* the paths are withheld is a caveat, and every count, cap and caveat is
in both forms.

That wording is the fix for a real misreading: a bare `-- 10 of 89 comments --` was read as
the ten *best*, so the agent concluded the thread held nothing further — while the review
that answered its question sat at position 51 of 97. **If you have a question, pass
`--focus`;** the sample is for orientation, not for deciding.

**A filtered thread is not a quiet one.** Machine-authored comments are excluded from the
default view — correct, they are not evidence — and that exclusion is now announced rather
than left to be inferred from a short list. `-- 0 of 0 comments --` on a thread the index
holds a bot comment for asserted the thread was untouched, which is the one thing it did
not mean:

```
-- 0 of 1 comments, 1 machine-tier suppressed (`relore search --trust machine` asks what the bots claimed) --
```

`comments_total` counts every comment the thread has; `comments_machine_suppressed` is how
many of them this view withheld. The difference is what the cap and the sampling compose
with, so a suppressed comment never shows up as one the page ran out of room for.

**Each page records when its body and comments were last indexed**, as `indexed_at`
in `--json`, with `indexed_at_note` explaining its scope:

```
-- 2 of 2 comments, indexed at 2026-09-09T09:02:49Z (body and comments last indexed; later GitHub changes unknown) --
```

This records an indexing visit, not a completeness guarantee or an atomic GitHub snapshot.
`status` reports collector telemetry: high-water is a source cursor and last-ok is collector
completion, neither the freshness of a given thread. Bulk `[issue_comments]` and
`[pr_comments]` rows do not track per-thread refreshes via `[threads]`.

**A long comment comes back as one hit, marked.** GitHub comments are chunked into several
indexed documents, and two chunks of one comment used to arrive as two hits with the same
URL, author and tier — which reads as two people agreeing. Now they collapse into one slot,
and a hit that is a piece of something longer says which piece:

```
1. huggingface/transformers#39847 pr  [authoritative]  12mo  issue_comment  @zucchini-nlp  (passage 4 of 7 in this comment)
```

A 200-comment thread has no full form; `search` is capped at 10 hits and 400 characters of
snippet. The caps are the contract, not a default (§6). The *body* is the exception, and
only on request: `--full` serves the opening post whole, because an issue template spends
its first several hundred characters on environment boilerplate and the reproduction
starts after it.

```bash
relore thread 48630 --full --repo huggingface/transformers   # the opening post whole
```

### "Did this pull request touch that file?"

Read the right list. A thread's paths come from three places and they are three keys, never
one:

```
changed files: 100 of 323 collected. TRUNCATED: a path that is absent here may still have been touched
  the collected page is the first 100 in path order, cut after src/transformers/models/gemma3/modular_gemma3.py — a path sorting after that one is absent whether or not the thread touched it
  src/transformers/modeling_rope_utils.py, …
  + 2 more the diff must contain, from the files inline review comments are anchored to (not part of the collected page)
mentioned in the discussion: 5 — named by somebody, NOT the diff. A bare filename here is not evidence the thread changed it
  config.json, modeling_rope_utils.py, …
```

- **changed** is the diff, collected 100 rows at a time by the per-PR pass. **Absence
  proves nothing** while it says TRUNCATED — and the cut is a *prefix*, not a sample:
  GitHub serves a diff in path order, so `100 of 323` means every path sorting after the
  one named on the second line is missing whatever the thread touched. On `#39847` that
  boundary fell at `gemma3`, which is why a reader looking for `gpt_neox` found nothing.
- **anchored** is also the diff — GitHub will not anchor an inline review comment anywhere
  else — and is the one source that can name a path the 100-row cap dropped.
- **mentioned** is prose: a bare basename, a traceback's path, a file somebody merely
  brought up. `config.json` above is named in `huggingface/transformers#39847`'s discussion
  and is **not** in its 323-file diff. Presence here is not evidence of a change.

Merged into one array — which is what it used to be — that last line answered a membership
question with a wrong yes, and the same file appeared twice in two different spellings.

Under `--compact` a **truncated** changed-file list is served as its shape instead of its
paths, because that list is the one nothing may be concluded from and on `#39847` it was
half the compact page (#56):

```
changed files: 100 of 323 collected. TRUNCATED: a path that is absent here may still have been touched
  the collected page is the first 100 in path order, cut after src/transformers/models/gemma3/modular_gemma3.py — a path sorting after that one is absent whether or not the thread touched it
  92 under src/transformers/ across 33 directories, 5 under examples/modular-transformers/, 3 under docs/source/
  (paths omitted under --compact; `--json` serves them, and so does the default render)
```

*"A wide mechanical refactor across model directories"* is what the list was read for, and
it is a sentence. The counts and the caveats are unchanged — `--compact` never trims those
— and a **complete** list keeps its paths in every mode, since a complete list is the one
that can answer a membership question.

**`--repo` is required on a bare number when more than one repository is in scope.** The
deployment indexes three, so a bare `relore thread 47720` there answers

```
400: pass repo=: more than one repository is in scope ['huggingface/serge',
'huggingface/transformers', 'huggingface/trl'] — or set RELORE_REPO to default it
for the whole session
```

That is deliberate — a number alone is ambiguous and guessing would silently answer about
the wrong project — and the message lists what to choose from. It has nothing to do with
authentication: an open daemon resolves the anonymous caller to every indexed repository,
so the same rule applies. `search` needs no `--repo` because it spans the whole scope by
design.

`RELORE_REPO` is the standing answer: export it once and the flag is a per-call override
rather than a per-call tax.

## "Is somebody already fixing this?"

Ask it **before** diagnosing. It is one hop through the `Fixes|Closes|Resolves #N` edge,
and it prevents an agent's most expensive mistake — writing a patch for something already
in review.

```
$ relore inflight 48630 --repo huggingface/transformers

1 thread claims to close huggingface/transformers#48630

1. huggingface/transformers#48672 pr  open  8h  closes  @somebody
   fix: respect partial_rotary_factor in GPTNeoXJapaneseRotaryEmbedding
   https://github.com/huggingface/transformers/pull/48672
```

Every row carries `state`, `draft` and whether it merged, because those imply opposite next
actions: an approved pull request means stop, a stale draft means supersede it, a merged one
means the fix has shipped and the issue may simply need closing.

The **empty** answer is the one to read carefully. `nothing in the index claims to close …`
is an answer; the same sentence followed by *"this repository has no relationship rows at
all"* is not — it means the index was derived before the edge existed, and `relored derive`
fills it. There is no trust floor here: the deployment's own bot having an open fix is
precisely the duplicate you must not create.

## The offline verbs

`map`, `defs` and `refs` need **no server, no index and no network** — they read the
checkout you are standing in, uncommitted edits included, which is the whole point (§1).
`defs` and `refs` take `--repo` to ask the daemon's clone of HEAD instead, which is a
different question: the branch you are mid-edit on is the one thing the index has never
seen. `map` is local-only. `RELORE_REPO` deliberately does not switch these two — that is
a decision about *which tree*, and only the flag makes it.

```
$ relore defs relore/ingest/authority.py
43-56        class     AuthorityResult
59-117       function  resolve_authority
120-156      function  _resolve_one

$ relore refs resolve_authority
./relore/daemon.py:293  name
./relore/daemon.py:298  call
./relore/ingest/authority.py:88  definition
./relore/ingest/backfill.py:117  call
-- 9 references in 95 files (5 call, 1 definition, 3 name)

$ relore map relore/ingest --limit 2
74 definitions in 12 files, top 2 by name matches per definition
  (a count of the written *name*: same-named definitions share it, so the definition
   count is how much this row overstates)
  parse_timestamp     function  16 matches / 1 definition   relore/ingest/timestamps.py:19
  iso_utc             function  11 matches / 1 definition   relore/ingest/timestamps.py:29
```

`refs` reports **every occurrence and what it is** — a call, a definition, an attribute
read, a bare mention — with the counts per kind. Calls alone missed 95% of the sites in a
real sweep, including `self.foo` assigned rather than called, and a short list with no
denominator reads as the whole truth. `attribute` and `name` rows are matched on the bare
name, so they are the last mention of the chain (`a.b.foo` matches `foo`): reading them is
part of the answer, not a promise that a type was resolved.

`map` counts a written *name*, so definitions sharing one share the count — which is why
both numbers are printed and the ranking divides by the second. Dunders are excluded: they
are the same name everywhere, and a top 40 of `__init__` is the same list for every Python
project.

---

## "Why is this line like this?" (#9)

```
$ relore why src/transformers/models/gpt_neox_japanese/modeling_gpt_neox_japanese.py:90 \
    --repo huggingface/transformers
```

`git blame` gives the commit; this gives the argument. The daemon blames the line in its
working clone, resolves the commit to a pull request through `thread_commits` — falling
back to the `(#1234)` squash-merge subject, which it labels as a guess — and returns the
thread with the review comments **anchored on or near that line**. That last part is what
blame cannot give and what a clone cannot be asked for.

Anchors are matched within a window of lines rather than exactly: GitHub stores an anchor
as a name and we store a line number, so an exact match would drop every comment on a file
edited since.

**The line's revisions, not only its last one (#57).** Blame names the commit that touched
the line last, which on a reformatted line is a tidy-up standing in front of the pull
request that decided anything. The page lists the chain — `git log -L` over the *enclosing
definition*, so a line that moved inside its function is still tracked — with each revision
resolved to its pull request. Measured on `transformers`: blame of `generation/utils.py`
names #43121, a later refactor; the behaviour was argued in #37866.

**And where the line's history cannot reach, the word's can.** A revision chain follows a
*range*, so it tracks a line that moved and loses a block that was **rewritten** — which is
the common case, and was the measured one: on `generation/utils.py:2301` the whole
eight-commit chain postdates #37866. Four separate agent runs left this verb at that point
and ran `git log -S` by hand. The page now runs it for you, as a second list that never
merges with the first:

```
-- 7 commit(s) changed `fullgraph` in this file, newest first. This follows the word,
   not the line, so it reaches a block that was rewritten rather than moved --
   a2e76b908b1f  2025-08-19  #40137
> 🚨🚨 Switch default compilation to fullgraph=False (#40137)
   163138a911c1  2025-05-22  #37866
> 🚨🚨[core] Completely rewrite the masking logic for all attentions (#37866)
   …
```

**The word is chosen, and the page names it**, because the choice is a judgement and both
obvious rules for it are wrong. *Rarest word on the line* picks
`is_flash_attention_requested`, whose pickaxe returns exactly the commit blame already named
— the name arrived with the refactor. *Deepest history* picks `compile` and `cache`, which
are simply common. The rule is both, in order: the most distinctive word that actually
reaches further back than the line already does. A candidate whose history fills the page is
unbounded and skipped; one that stops no earlier than the revision chain has added nothing
and is skipped. `--presentation` (a TTY) also prints what it tried and what each reached.

**An empty origin list is a real answer, and it now says which answer** (#71). It printed
nothing at all, which is "the revision chain is the whole story" given by omission — and it
was not even always that answer. Three causes, three sentences:

```
(no origin chain: no word on this line reaches further back than the revisions above,
 so they are the whole story — not a gap. 4 candidate(s) tried.)
(no origin chain: this line carries no code, so there is no behaviour on it to follow back.)
(no origin chain: this file could not be read from the working clone… Absence here is not evidence.)
```

**A line with no code on it does not get pickaxed at all, and that was the slow path.** The
candidates come from the line *and the comment attached above it* — right for a line of
code, that is where the reason lives — and on line 1 of a source file the block is the
licence header. Measured against production, `why src/transformers/masking_utils.py:1` and
`modeling_llama.py:1` each spent five `git log -S` passes on `Copyright`, `HuggingFace`,
`rights`, `reserved` and `team`, every one a full walk of that file's history because a
term matching one commit never fills the page early: **3.98s and 6.35s**, against 1.77s for
a line whose first candidate was accepted and ~0.6s of baseline. Two of the web UI's three
`why` sample buttons are exactly that line, which is where it got noticed.

What this still cannot follow is a **renamed** symbol, which is milestone 4's rename chains.

**When the line window is empty, it widens rather than stopping (#63, #64).** In order,
labelled, so a widened answer is never read as an exact one: comments on the line →
comments on this file elsewhere in the same pull request → the pull request's **review
bodies**. That last group has no line anchor at all, so no window could have reached it, and
it is where an approval states its conditions — `transformers`#37866's approving review
("make sure full graph training is not broken… or at least fa2 training") is the reason the
line under it exists.

Two empty answers that are not the same, and the page says which:

- **no pull request carries that commit** — it predates the index, or reached the branch
  outside a pull request. Both chains above are still printed on that page: blame's own
  commit being unresolvable is exactly when a revision or a pickaxe hit that *is* indexed
  is the only way in;
- **a pull request, and nothing said at any level** — the page names the next command
  (`relore thread <n> --full`) rather than ending on a count of zero.

`why` needs the working clone below, because blame does. A `--depth 200` clone can read a
file and cannot attribute a line, which is the hole every cost-minimising agent falls into.

---

## The same questions, asked of the server (#7)

`defs` and `refs` take `--repo`, and `symbol`, `grep` and `copies` only exist there: they
read the daemon's **working clone**, one per indexed repository checked out at HEAD. That
is the half of a diagnosis that used to end in a throwaway clone and a throwaway script.

The clone carries its blobs. A filtered one is smaller and cannot answer `why`: blame walks
a file's history, so it would fetch from the remote in the middle of a query -- or fail,
where the remote will not serve an old object.

```
$ relore copies compute_default_rope_parameters --repo huggingface/transformers
186 definitions of compute_default_rope_parameters in 2 shapes
(grouped by what the body does: type annotations, docstrings and comments are
 normalized away first — `--exact` groups by the text instead)

-- shape 1: 185 copies (the majority shape)
   src/transformers/models/llama/modeling_llama.py:88
   …
-- shape 2: 1 copy  <- the only one of its shape
   src/transformers/models/gpt_neox_japanese/modeling_gpt_neox_japanese.py:90

$ relore grep 'partial_rotary_factor' --repo huggingface/transformers \
    --path 'src/transformers/models/**/modeling_*.py'
$ relore symbol GPTNeoXJapaneseRotaryEmbedding.forward --repo huggingface/transformers
```

`copies` groups rather than listing, because in a repository that duplicates model code on
purpose the question is never *where is it* but *which copies diverge* — the shape of
`huggingface/transformers#48630`.

**What "the same body" means here matters, and it is not the text.** Grouping on the text
was the first implementation and it made the verb useless on precisely this corpus: those
186 definitions came back as **172 shapes**, every group one model's generated file paired
with its own `modular_*.py`, and a "majority shape" of two. The entire split was one token —

```python
def compute_default_rope_parameters(config: GPTNeoXConfig, device=None, **kwargs)
def compute_default_rope_parameters(config: LlamaConfig,  device=None, **kwargs)
```

— so every model was its own shape before any difference in what it *computes* was
considered, and the one model that had diverged was invisible among 171 that had not. So
type annotations, docstrings, comments and formatting are normalized away before the
grouping, along with indentation: a function lifted into a class groups with its original.
`--exact` keeps the old rule for whoever wants the text, and `--json` carries both hashes
per copy plus `bodies`, the number of distinct texts inside each shape.

**A shape of one is called out.** On a symbol with 186 copies the singleton is almost
always the row the question was about.

`symbol` prints how many definitions of that name exist. Serving the first of 186 as *the*
body is a wrong answer a caller cannot see.

`grep` names both denominators — `-- 3 hits in 2 of 4 files searched`. It used to print
`3 in 4 files`, where 4 was the count of files *read*; an agent sizing blast radius reads
that as "appears in four files", and on an audit a file count that over-reports is a wrong
answer in the direction that looks like diligence. The cap discloses itself on its own
line, with the remedy in it.

**HEAD, not the tree as it was.** These verbs answer "what does this code look like now",
which is what an audit asks. The lens that reads a comment's tree at the time it was
written is a different depth and a separate decision.

**A repository with no clone answers 503 with a sentence**, and the history verbs are
unaffected: the conversation index never depends on a checkout. `relored clone --repo
OWNER/NAME`, run where `relored serve` runs, creates one.

---

## "What did the project know, before it knew the answer?"

`--since` narrows to recent discussion. `--before` is the other question: the index as it
stood at a past instant, with everything written after it withheld.

```
$ ghlore search "rope scaling" --before 2026-03-01T00:00:00Z
```

It is a predicate in the query, not an instruction — the server drops the rows, so nothing
after the cutoff is read and then disregarded. A document whose creation date the index does
not know is **excluded** rather than admitted: an unknown date cannot be shown to predate
the cutoff. `--since` and `--before` compose into a window, and a window with nothing in it
(`--since` at or after `--before`) is refused rather than answered with an empty page.

`thread`, `why` and `inflight` take the same cutoff. A thread opened after it is **404**,
not an empty page — "this did not exist yet" and "this exists and said nothing" are
different answers. `inflight` in particular: asked for an issue number it normally names
the pull request that closed it, which under a cutoff is the answer.

`--exclude N` withholds one thread by number, repeatably. It is the part a cutoff cannot
express: the thread a question was drawn from is *older* than the cutoff and is still the
answer.

```
$ ghlore search "rope scaling" --before 2026-03-01T00:00:00Z --exclude 41302 --exclude 41577
```

### Pinning a whole daemon

A cutoff a caller has to pass is a cutoff a caller can omit, which is no use for an
evaluation where the caller is an agent with a shell. So the pin belongs to the process:

```
$ ghlored serve --as-of 2026-03-01T00:00:00Z --exclude-thread 41302
```

Every read is then bounded, a request may only narrow it further, and `ghlore status`
reports the pin. This is an evaluation mode — a pinned daemon serving real callers is an
index quietly missing its last six months, which answers every question plausibly.

One thing it does *not* freeze yet: the signal tables (`--file`, `--symbol`, `--error`,
`--test`) are per *thread* and carry no date, so a path first named after the cutoff still
matches that thread's older documents. The documents returned are genuinely pre-cutoff; the
selection is not.

## Empty results that are not faults

**No results is always exit 0 with an empty result.** So when something comes back empty,
it is one of these before it is a bug:

**1. The query was a sentence.** See the table above.

**2. A structural filter was involved.** `--file`, `--symbol`, `--error` and `--test` read
the signal tables that build-plan §5.3's extraction pass fills, and they *AND* with the text
query — so one of them narrows an otherwise-good search:

```
$ relore search "429"                              → 3 hits
$ relore search "429" --file reviewbot/llm.py      → 0 hits   # nothing said both
```

A `--file` page also says what it could **not** decide:

```
$ relore search "tensor size mismatch" --file modeling_gpt_neox_japanese.py
10 hits for 'tensor size mismatch'
3 more threads matched but have no collected changed-file list, so --file could not test them.
```

Those three are not non-matches. A pull request only has a changed-file list once the
per-PR pass has reached it, and until then it is absent from the page without having been
tested — so an empty or short `--file` result is a statement about the index as much as
about the corpus. `files_untested` carries the same number in `--json`.

It counts **pull requests only**: an issue has no diff and never will, so its absence from
a `--file` page is the right answer rather than a gap.

Three things make one match nothing on an index that does hold the answer. `--file` takes
the path **as the repository spells it** — but any trailing part of it will do, matched at a
`/` boundary, so `modeling_gpt.py` and `models/gpt/modeling_gpt.py` both find
`src/transformers/models/gpt/modeling_gpt.py`. An absolute path from your own checkout
still will not: strip the leading directories the repository does not have. `--test` takes
a runner id
(`tests/test_x.py::test_y`, with or without its `[params]`), not a bare function name; a
bare name is a `--symbol`. And `--error` may be given a whole pasted traceback, which is
normalized into the form the index stores — but a *paraphrase* of an error is text, so pass
it as the query instead.

**3. A trust floor removed everything.** `--trust authoritative` — or `--kind rationale` /
`--kind precedent`, which imply it — on an index where `relored authority` never ran returns
nothing at all. Check `trust_floor` in the `--json` response.

**4. One verb is parsed but not built.** It says so rather than returning an empty result
that reads like an answer, and `--help` marks it:

```
$ relore precedent --kind bug_fix
relore precedent: not implemented yet (milestone 4, the build plan section 13)
```

`why` was the other one in this block for two releases after it shipped, which is the drift
`tests/unit/test_prose_surfaces.py` now catches: no page may call a shipped verb
unimplemented, and the list of shipped verbs comes from the parser rather than from a copy
of it.

A daemon that is *down* is a different message from an index that is empty, and the client
tells you which — that distinction is deliberate (§12).

## Checking the index rather than the query

```
$ relore status
version   0.3.17
backend   postgresql / ts_rank_cd  capabilities: fulltext, weighted
schema    applied [1, 2, 3, 4, 5, 6, 7], pending []
index     55168 threads, 517276 documents, 332 raw objects
collector telemetry (not response freshness; use each thread's indexed_at): high-water is a source cursor; last-ok is collector completion
  huggingface/serge [threads] high-water 2026-09-03T06:55:06+00:00 last-ok …
  huggingface/transformers [threads] high-water 2026-09-12T18:41:02+00:00 last-ok …
  huggingface/trl [threads] high-water 2026-09-12T18:44:15+00:00 last-ok …
quota     1/60 per minute, 1/5000 today
```

**Three rows, and that is the precondition rather than a footnote about it.** This block is
the reader's mental model of `status`; when it showed one repository it also showed the one
case in which a bare `relore thread 47720` never fails. Every row here is a repository
`--repo` has to choose between, and `RELORE_REPO` is how you stop choosing.

`capabilities` is how you tell which engine answered. A SQLite index scores with `bm25` and
has no trigram or vector tier, so a result set from one says nothing about the other —
which is why `relored serve` refuses a SQLite URL without `--allow-sqlite`, and why §10's
benchmark refuses to mix backends.

`version` is the first line because the client and the daemon must be the *same* version.

## "your client is older than this relore daemon"

Not an outage and not something to work around:

```
$ relore search "429"
relore: client 0.2.4 is older than this relore daemon (0.3.0), so it would read an
out-of-date answer as a complete one. pip install --upgrade 'relore @ git+…'
```

The two ship together, so every request declares its version and a daemon refuses any
other — an old client would otherwise get a well-formed answer missing whatever it does
not know to ask for, which is the one failure nothing downstream can detect. Do what the
message says. If it instead says the daemon is **behind**, the client is fine and the
*deployment* is the stale thing: redeploy it rather than downgrading.

## Parsing the output

With no MCP server, stdout is the API, so what it prints is a contract rather than a
rendering.

**There is one TTY branch, and it may only add advice** (#13). On a terminal the page also
carries the backend tag and the flags worth trying next — `--full`, `--focus`, `relored
derive`. Through a pipe, none of that is printed; `--plain` forces the piped form on a
terminal.

**Every fact is in both forms.** Counts, caps, truncation warnings, the sample-versus-ranked
line, the trust tiers, the envelope and the `> ` marking are not presentation and do not
move. This section used to say no branch could ever exist, because a quiet mode for pipes
makes the version nobody looks at the version everybody consumes — and that reason is
exactly why the split is advice-only: what a person sees is what a pipe sees plus
suggestions, so the two cannot disagree about what is true. The renderer stays server-side
and shared with the web UI for the same reason.

**Prefer `--json`.** It carries everything the text does and nothing a caller has to
un-format: the comments array, `body_chars`/`body_truncated`, the three file lists,
`files_total`/`files_collected`, `comments_total` with `comments_machine_suppressed`,
`indexed_at`, `selection`, per-hit `score` with its
`breakdown`, and each hit's `document_id`, `source_id`, `chunk_index` and `passages`. The
code verbs the same way: `grep` carries `files_searched` and `files_with_hits` separately,
and `copies` carries `exact`, each group's `shape_hash` and `bodies`, and each copy's
`body_hash`, `shape_hash` and `normalized`.
All four globals — `--json`, `--compact`, `--plain`, `--api` — are accepted on every verb
and on either side of it.

If you do read the text, these hold within a version — and the version is enforced on every
call, so "within a version" is something you can rely on rather than hope for:

- one blank-line-separated block per hit, `N. repo#number type  [tier]  age  source_type
  @author` first, then the quoted title and snippet, then the URL;
- every line of retrieved prose is prefixed `> `, and no line of ours ever is, so an
  unmarked line is always `relore` speaking;
- the whole page sits inside `<<<RELORE-UNTRUSTED>>>` … `<<<RELORE-UNTRUSTED-END>>>`;
- counts and caveats are prose on their own line (`-- 10 of 89 comments … --`,
  `changed files: …`, `(body truncated: …)`), and a caveat is never dropped for brevity —
  `--compact` replaces a truncated changed-file list with its shape and says it did, but
  the count, the denominator and the TRUNCATED caveat above it are identical in both
  forms;
- no score is printed. The order is the ranking, and the number's scale is a property of
  the backend — it is in `--json` for whoever is tuning weights;
- `state_reason`, `closed_by`, `merged`, `review_decision` and `requested_reviewers` (#22)
  render as their own lines under the header — `closed as duplicate by @login`, `closed by
  its author`, `merged`, `review: approved by @login` — and as `closed (duplicate) by
  @login` in an `inflight` row's state column. The review line is printed even when the
  thread has no comments, which is the only thing that tells `-- 0 of 0 comments --` apart
  from *approved without typing*;
- the comment head line names every reason a comment is not on the page — the cap, the
  sampling, the focus, and the trust tier — and ends with what the thread is current to. A
  count printed after a filter, with the filter unmentioned, is the failure this whole
  section exists to prevent;
- `(body truncated: …)` fires only when something worth reading was withheld. Normalizing
  whitespace is not truncation, and a remainder shorter than the sentence announcing it is
  served instead of announced;
- fields are never reordered within a minor version, and the version is enforced on every
  call, so a parser pinned to one version cannot be surprised by a cosmetic change.
