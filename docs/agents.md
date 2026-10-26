# Using it from an agent

*(This page is harness wiring. The root `AGENTS.md` is a different document — the operating
contract for contributing to `relore` itself.)*

**`relore --help` is the surface to point an agent at first.** It names every verb, the
order to reach for them in, the environment, and runnable examples — and unlike this page
it arrives byte-exact, needs no network and is already installed. A page is rendered,
summarized, truncated and cached by whatever sits between it and the reader; one field run
read our landing page through a summarizing fetch that dropped `--repo` off every example,
and its first two calls failed on exactly that.

`skills/search-project-history/SKILL.md` ships a ready-made skill for agents that read one
(Claude Code and similar): when to reach for the index, when to reach for the code lens,
how to phrase a query, what the trust tiers mean, and how to triage an empty result. Point
your harness at it, or lift the prose.

## Set `RELORE_REPO`

A daemon serving more than one repository refuses a bare number rather than guess which
project it belongs to, so `--repo` is required on `thread`, `inflight`, `why`, `symbol`,
`grep` and `copies`. `RELORE_REPO` is the session default for it, alongside `RELORE_API`
and `RELORE_TOKEN`. Without it a single-repository session pays that flag on every call — a
cost that grows as the index grows by repository, and one that does not depend on the agent
having read any of this.

Any agent with a shell can use `relore` with no integration work. There is no MCP server on
purpose: any agent with a shell can already call an HTTP API. If your framework reads a
repo-declared tool manifest, declaring it is a few lines:

```json
{
  "helpers": [{
    "name": "search_project_history",
    "description": "Search this repository's issue and PR history. Flags: --error, --test, --file, --symbol, --label, --author, --kind, --trust, --limit, --compact, --json. Use it before writing a patch to check whether this failure has precedent.",
    "command": ["relore", "search", "--json"],
    "allow_args": true, "max_args": 12, "timeout_seconds": 30,
    "install": ["pip", "install", "relore"]
  }]
}
```

## Two things every integration must get right

- **Retrieved history is untrusted text**, written by whoever opened the issue. `relore`
  wraps every response in an untrusted-content envelope and scrubs delimiters, and inside
  that envelope the quoted lines — and only those — are marked `>`, so a trust tier or a
  count of ours is never mistaken for something a stranger wrote. The agent must still be
  told never to follow instructions found in retrieved content.
- **Cite and verify.** Every result carries a URL and its age. A 2019 comment can be right
  about intent and wrong about today's code.

## Ask before starting work

`relore inflight <number> --repo owner/name` answers *is somebody already fixing this?* —
the threads that claim to close a given issue, open ones first, each with `state`, `draft`
and `merged`. An agent's most expensive failure mode is duplicating work that is already in
review, and this is the query that prevents it.

## Check the claims against the code

`relore grep`, `relore symbol` and `relore copies` run against the daemon's working clone
at HEAD, with `--repo`. They are what turns a thread's claim into a verified one, and they
need no checkout — which is what they exist for. `relore why <path>:<line> --repo
owner/name` is the join: the pull request that last changed a line, and the review left on
it.

## Query shape

A query is an AND of every content term, so two or three distinctive ones beat a sentence.
A pasted traceback works too: expansion fans it out into error, file and symbol legs and
merges them. `--no-expand` asks exactly one question instead.

Worked examples, including the four ways a healthy index returns nothing:
[`cli.md`](cli.md).
