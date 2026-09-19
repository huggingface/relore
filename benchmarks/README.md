# benchmarks

Five things are measured in this repository, and they answer different questions. Reaching
for the wrong one is how a session spends an evening proving something nobody asked.

| | what it answers | needs | in `make check`? |
|---|---|---|---|
| `huggingface-*.jsonl` + `relored mine\|judge\|bench` | **retrieval**: given a query, is the answering thread on the page? | an indexed database | no |
| `fieldrun/` | **flow**: what does an agent actually do with that page? | a database, a model, network | no |
| `probes/page_cost.py` | **page cost**: what does one rendered page cost, and how much of it is itself repeated? | an indexed database | no |
| `probes/api_latency.py` | **latency**: how long does the deployment take, verb by verb — and what do the web UI's own buttons cost? | a deployment | no |
| `probes/code_lens.py` | **the local verbs**: what do `map` and `refs` cost against `grep`, and is the answer still a strict subset of it? | a checkout | no |

None of them run in CI. Three need a database, two need the network, and one costs money.
They are measurements you take deliberately, before and after a change to what a verb puts
on a page.

## Section 10: retrieval

The frozen evaluation sets and the runner behind build plan §10 — `relore` scored against
GitHub search and against the agent's own `grep`, on `failure`, `rationale` and `precedent`
slices. `benchmarks/huggingface-transformers.jsonl` and `benchmarks/huggingface-bot-reviews.jsonl`
are **frozen**: `judge` refuses to add a label to either, and growing the ground truth means
a new file. See `relored bench --help`.

Two rules that live in the runner rather than in whoever reads the table: never compare
across backends, and restrict every baseline to the corpus window.

## `fieldrun/`: flow

Drives a tool-calling model on Hugging Face Inference Providers through real questions, with
the `relore` CLI in one arm and the `gh` CLI in the other, and records every call. Derives
`time_to_evidence`, `tool_share_of_prompt`, `tool_chars_median`, `repeat_share`, `rerolls`
and `cited_unseen`. See `fieldrun/README.md` — especially the part about not quoting totals.

## `probes/page_cost.py`: what a page costs

The static half of `fieldrun`'s `repeat_share`, and the cheaper one: no model, no network,
one second. It renders real threads out of an index and reports bytes, approximate tokens,
and **how much of each page the page has already said**.

```bash
python benchmarks/probes/page_cost.py --url "$RELORE_DATABASE_URL" \
    --repo huggingface/transformers --biggest 5
```

This is the measure that found relore#71: a thread page reprinted `owner/repo#N pr`, the
thread's title and a 76-character URL under every comment — three constants the head line
already carried — and on five production threads that was **24–33% of the whole page**. No
token total showed it; the repetition did. On `#46766`, before and after:

```
pre-fix :  6347 B  repeat=0.193  const=558 B  ['huggingface/transformers#46766', …]
post-fix:  4775 B  repeat=0.000  const=128 B  ['[authoritative]']
```

Both of its thresholds were wrong on the first draft and that same page corrected them —
the reasoning is in `echo()`, because a probe that silently under-reports is worse than none.

## `probes/code_lens.py`: what the offline verbs cost, and whether they still tell the truth

The only one here that needs no database and no deployment — just a checkout.

```bash
python benchmarks/probes/code_lens.py --root ~/src/transformers
```

It times `relore refs` and `relore map` against `git grep` and the naive `grep -rn` an agent
reaches for by reflex, across four symbols chosen to span rare → ubiquitous, because the
prefilter behind issue #83 is exactly as selective as the name is rare and a table with only
a rare name in it overstates the change by twenty times.

Speed is the easy half and on its own it is misleading, since **either speedup can be made
arbitrarily fast by returning less**. So every run also checks that `relore`'s locations are
a strict subset of `git grep`'s, and accounts for every line grep has that the lens does not:
a file outside the walk, a longer identifier containing the substring, or prose. Prose is
decided by `tokenize` — the stdlib's own lexer, deliberately a different implementation from
the tree-sitter grammar under test. A per-line regex could not see that a docstring's fourth
line is inside a docstring, and reported 867 of them as losses. On `transformers` @ `d9890f6`:

```
  use_kernels                           45 of 66     subset  (12 wider identifier, 9 prose, 0 unexplained)
  forward                             8627 of 15355  subset  (4445 wider identifier, 2283 prose, 0 unexplained)
  config                             77732 of 121697 subset  (35210 wider identifier, 8755 prose, 0 unexplained; 7 outside the walk)
```

Anything left unexplained, or any location the lens reports that grep cannot see, is printed
in full and exits non-zero. It also asserts that cached and `RELORE_NO_CACHE` runs are
byte-identical, which is issue #83's acceptance criterion in executable form.

## `probes/api_latency.py`: what the deployment costs

```bash
python benchmarks/probes/api_latency.py --base https://relore.example.org \
    --repo huggingface/transformers --thread 46419
```

Times each verb against a live deployment, and separates the **connection baseline** from
what the verb costs — against a VPN-internal deployment, DNS + TCP + TLS is most of a fast
call and attributing it to the verb makes every verb look slow.

It also times the web UI's **own sample buttons**, read out of `relore/api/ui.py` rather
than retyped, because that strip is a latency surface and it is where a person reported the
product as slow. Measured against production before relore#71:

```
connection baseline (GET /healthz, best of 3): 0.11s
ui sample  why   src/transformers/masking_utils.py:1          4.03s
ui sample  why   …/modeling_rope_utils.py:180                 1.40s
ui sample  why   …/models/llama/modeling_llama.py:1           5.22s
ui sample  search  (all four)                            0.16-0.25s
verb       thread / --outline / --full                   0.14-0.17s
```

Two of the three `why` samples point at line 1 of a source file, where the origin pass was
pickaxing the licence header: five full `git log -S` walks for an answer that was empty.
