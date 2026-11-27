# Ranking

**Built.** Retrieval is full text, filters, the trust floor, §6's **query expansion** and
§6's **weighted score**, gated on the evaluation set below rather than fitted by eye — a weight
chosen by hand is one nobody can argue with later. Every hit carries its score broken down
by term, so the term that is wrong is visible in the web UI.

Lexical and structural: exact evidence outranks topical similarity. What that means for the kind of
question you can ask — and the two tools it is not — is in
[`how-search-works.md`](how-search-works.md).

```text
score = w1*error overlap + w2*test-id + w3*symbol + w4*file + w5*full-text rank
      + w6*relationship, then adjusted for recency and author association
```

Weights are per-deployment, because what discriminates in one repository does not in
another. Decay is query-kind-aware — an 18-month half-life for errors, four years for
precedent, **none** for rationale, where the oldest thread is often the answer.

Two things ride alongside the score rather than in it:

- **`--sort newest`** reorders a page by date without re-selecting it. Selection stays on
  relevance, because letting recency choose a thread's representative document is a
  measured bug: it picks the sign-off (§10.6, §10.7).
- **Decay is written and gated off** (`ranking.DECAY_ENABLED`). Its release condition — a
  full-history backfill — was met on 2026-09-09, but §10.5's numbers were measured on a frozen
  six-month set that refuses new labels, so turning it on honestly needs new ground truth
  first. The gate now records a measurement not yet taken.

## Trust is a filter, not a weight

A maintainer's "we cannot do that because…" is a judgement, a passer-by's is a claim, and a
bot's is our own output coming back. Human tiers are a ladder — `reported`, then
`authoritative` — and `--trust authoritative` raises the floor to the second.

`machine` is deliberately **not** on that ladder. `--trust machine` is a separate, explicit
question (*what did our own bot claim here?*), not a lowered floor. The **bot exclusion** is
live because it is a security property rather than a ranking choice: the corpus already
contains the deployment's own agent's comments, and returning one as prior discussion makes
that agent's unreviewed output its own evidence.

The **per-query-kind floors** are live too — a `precedent` query requires authority,
because merging is the maintainer's act — together with the `relored authority` pass they
depend on. Until each author's repository permission is resolved, a rationale floor of
"maintainers only" filters out the very comments it exists to find, because on an org-owned
repository the real maintainers read as an unresolved `MEMBER`.

## What the benchmark says

Two sets, both model-judged, both scored against **both** baselines — GitHub search and the
agent's own `grep`. Postgres `ts_rank_cd`, window 2026-03-01, and every baseline restricted
to that same window (`grep` reads a working tree, which has no window at all). Recall@10:

| corpus / slice | n | unranked | **+ expansion** | GitHub | `grep` |
| --- | --- | --- | --- | --- | --- |
| transformers `failure` | 48 | 0.938 | **1.000** | 0.875 | 0.000 |
| transformers `precedent` | 40 | 0.375 | **0.475** | 0.325 | 0.200 |
| transformers `rationale`, pooled | 51 | 0.765 | **0.980** | 0.294 | 0.314 |
| transformers `rationale`, leak-free | 7 | 0.286 | **0.857** | 0.857 | 0.143 |
| bot-reviews `rationale`, leak-free | 9 | 0.222 | **1.000** | 0.667 | 0.667 |

**Six things to know before quoting any of it**, because the interesting parts are the
caveats:

1. **The pooled `rationale` figure is mostly a leak.** A *human* reviewer's finding is
   itself a document in the answer's own thread, so any lexical system reaches the answer
   without retrieving anything. Only a *bot's* finding is clean, because §6.2 excludes
   machine documents from reads — hence the separate leak-free rows, which are the ones that
   mean something. The bot-reviews corpus (diffusers and mlinter, where bot review comments
   are 4.7% and 54% of the traffic against transformers' 0.9%) exists purely to make that
   slice wider than seven examples.
2. **Unranked, `relore` lost that slice on both corpora** — 0.286 and 0.222 — and the cause
   was *under-retrieval*, not ranking: 6 of 9 queries returned nothing, because the query is
   drawn from the bot's finding and that finding is the only document carrying all its
   terms, which §6.2 refuses to read. Search ANDs content terms, so it asked
   for the one document the index will not return.
3. **Query expansion fixed it** (§10.4): fanning one call into error, test, symbol, file and
   free-text legs — plus the caller's own filters asked without their text — and merging
   them. Every slice improved and none regressed. The two leg families do disjoint work,
   which is measured: the filters leg alone fixes `failure`, the derived legs alone fix
   `precedent`, `rationale` needs both.
4. **`grep`'s zero on `failure` is the sharpest result in the set, and it is structural.**
   Those answers are mostly *issues*, and grep's only route from a file to a thread is "the
   threads that touched this path" — a PR-shaped relation. Traced over every candidate
   rather than the top ten, the answer is in grep's list for 13 of 48 examples and never
   above rank 26.
5. **The weighted ranking closed the one gap expansion left** (§10.5). Expansion raised
   `precedent`'s recall while its MRR stayed flat at 0.191 — it *found* the precedent thread
   and could not order it, because a textless leg scored every row 0. The weights moved that
   MRR to **0.286** at Recall@10 0.525. The mechanism is not the weight values: scoring each
   leg against the *whole* question rather than against its own single filter is the entire
   delta, and keeping the weights while dropping that reproduces the pre-weights numbers
   exactly. A weight can only discriminate over evidence the filter did not already require.
6. **Then a single real query found what 139 examples could not** (§10.6). Asked with a file filter,
   nine of ten slots came back `LGTM`, `Thx`, `Yep` — because the term that scores text a
   query is *not* filtering on was a conjunction, so almost nothing scored above zero and
   the page fell through to a recency tie-break that picks each thread's last comment.
   Filtering ANDs; scoring ORs. **Recall@k is blind to this**: a page of ten slots where
   nine are noise scores identically to ten useful ones, as long as the answer is on both. A
   recall number is necessary and not sufficient.
7. **And the same trap, one rung further out** (0.3.13). A prose query whose terms AND to
   nothing is now asked again with them disjoined rather than returning an empty page. Ranked
   the obvious way — `ts_rank_cd`, the ranking the weights were fitted with — that page was
   wrong in §10.6's exact shape: density counts *occurrences*, so against 41,857 documents of
   the `transformers` sample the top hit carried two of seven terms and the one carrying six
   came sixth. A widened page is ordered by **coverage** instead, `matched / asked` over the
   caller's terms, with normalized density divided by `len(terms) + 1` underneath it so the
   order is exactly lexicographic. Neither frozen set can see this either: the widening is
   reachable only from an empty page and both sets report `empty 0`, so the numbers above are
   unchanged by it — which is the third time on this list that the benchmark was necessary
   and not sufficient.

8. **Coverage and semantic legs, for relore#79** (§10.8, 0.3.20). `qwen` never matched
   `qwen3`, and the widened score above still has no IDF — `transformers` + `got` scored like
   `qwen3` + `prefill`. A **coverage** leg ranks threads by `ln(N/df)` over the six rarest
   query terms they hold anywhere, plain words prefix-matched, fused by thread-level RRF and
   filled breadth first: frozen `precedent` Recall@10 0.500 → **0.575**, nothing regressed.
   A **semantic** leg (`bge-small-en-v1.5`), gated to prose because on identifier queries
   it only cost rank, leaves every frozen row unchanged except `failure` MRR 0.917 → 0.927,
   and moves prose queries built from the sets' own text: `precedent` 0.575 / 0.388 →
   **0.650 / 0.437** (Recall@10 / MRR). Larger models were not better in aggregate.

Two limits the table does not show. `failure` is close to circular — its ground truth is
"the threads carrying this error" and `--error` matches exactly that — so no weight set can
be fitted on it, and it is the one slice that lost a little MRR (0.938 → 0.927, at unchanged
ceiling recall) when the weights landed. That was left uncompensated on purpose. And the
weight *values* are not yet falsifiable on this corpus at all, for the reason in point 5 —
`thread_links` and a live `w_rel` are milestone 4's first chance to test one.

How to run it: [`operations.md`](operations.md).
