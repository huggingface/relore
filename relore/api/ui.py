"""The web UI (section 8) -- built early, on purpose.

One self-contained page served by ``relored``, calling the *same* API the CLI calls. No
build step, no framework, no second ranking implementation: if the UI grows its own query
logic it stops being representative, and the thing you tuned is no longer the thing agents
use.

It is the instrument the rest of the work is calibrated with. After a backfill the only
question is "are these results any good?" -- no metric answers it, and a person reading ten
results answers it in a minute. Section 6's weights are a guess, and replacing them
honestly means running real queries and fixing the term that is wrong.

Three affordances exist for people and not for agents:

1. **Score breakdown per hit.** An agent has no use for *why* something ranked; the person
   tuning the weights has nothing else.
2. **View as the model sees it.** The ``rendered`` field from the API -- the exact string
   the CLI prints, envelope and delimiter scrubbing included -- so a formatting or
   injection bug is caught by a person reading it rather than by an agent meeting it
   mid-task. It is the server's own rendering, not a copy: :mod:`relore.render` has one
   implementation and two callers.
3. **Copy as CLI invocation.** Debugging a bad agent answer starts with reproducing it.

Plus the index-health strip, which beats a dashboard panel here because the question is
almost always "is the index current?" and the answer belongs next to the results.

Labelling writes section 10's evaluation set as a byproduct of use rather than as a chore
nobody schedules. It needs a token with the ``label`` scope and a configured path, and it
lands in a JSONL file that is never indexed -- see :mod:`relore.api.server`.
"""

from __future__ import annotations

import json
from html import escape

from relore import __version__, guidance

#: The tab icon: the header mark without the waves, which are mush at 16px. A data URI
#: rather than a route, so the page stays one self-contained response -- and assembled
#: here rather than inline because one line of percent-encoded SVG is six times the line
#: limit. Fixed ink: a favicon has no page to take `currentColor` from.
_FAVICON = (
    "data:image/svg+xml,"
    "%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='-3 0 46 38'"
    " fill='none' stroke-linecap='round'%3E"
    "%3Cpath d='M14.7 3.3V31.4' stroke='%232f3437' stroke-width='4'/%3E"
    "%3Cpath d='M14.7 18.5Q23.4 18.5 28.4 12.4' stroke='%232f3437' stroke-width='4'/%3E"
    "%3Cg stroke='%232f3437' stroke-width='4' fill='%23fbfbfa'%3E"
    "%3Ccircle cx='14.7' cy='3.3' r='3'/%3E"
    "%3Ccircle cx='14.7' cy='18.5' r='3'/%3E"
    "%3Ccircle cx='14.7' cy='31.4' r='3'/%3E%3C/g%3E"
    "%3Ccircle cx='30.6' cy='10.6' r='2.8' stroke='%23c8633a' stroke-width='3.4'"
    " fill='%23fbfbfa'/%3E%3C/svg%3E"
)

_PAGE = """
<!doctype html>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>relore</title>
<link rel="icon" href="/*FAVICON*/">
<style>
  :root {
    color-scheme: light dark;
    --bg: #fbfbfa; --fg: #1a1a1a; --dim: #6b6b6b; --line: #e0dfdc;
    --card: #ffffff; --accent: #1c5d99; --warn: #8a4b00; --mach: #8a1c1c;
    font: 14px/1.5 ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif;
  }
  @media (prefers-color-scheme: dark) {
    :root { --bg:#16181c; --fg:#e6e6e6; --dim:#9a9a9a; --line:#2c3038;
            --card:#1d2026; --accent:#7fb2e5; --warn:#e0a55c; --mach:#e08080; }
  }
  * { box-sizing: border-box; }
  body { margin: 0; background: var(--bg); color: var(--fg); }
  main { max-width: 60rem; margin: 0 auto; padding: 1.5rem 1rem 4rem; }
  h1 { font-size: 1.1rem; margin: 0 0 .25rem; letter-spacing: .02em;
       display: flex; align-items: center; gap: .5rem; }
  h1 .mark { flex: none; }
  h1 small { color: var(--dim); font-weight: 400; }
  form { display: flex; flex-wrap: wrap; gap: .5rem; margin: 1rem 0 .5rem; }
  input, select, button, textarea {
    font: inherit; color: inherit; background: var(--card);
    border: 1px solid var(--line); border-radius: 4px; padding: .4rem .55rem;
  }
  input[name=q] { flex: 1 1 22rem; }
  button { cursor: pointer; }
  button.primary { background: var(--accent); border-color: var(--accent); color: #fff; }
  .strip {
    display: flex; flex-wrap: wrap; gap: 0 1.25rem; padding: .5rem .75rem;
    border: 1px solid var(--line); border-radius: 4px; background: var(--card);
    color: var(--dim); font-size: .82rem;
  }
  .strip b { color: var(--fg); font-weight: 600; }
  .hit {
    border: 1px solid var(--line); border-radius: 4px; background: var(--card);
    padding: .7rem .85rem; margin: .6rem 0;
  }
  .hit h2 { font-size: .95rem; margin: 0 0 .2rem; }
  .hit h2 a { color: var(--accent); text-decoration: none; }
  .meta { color: var(--dim); font-size: .8rem; display: flex; flex-wrap: wrap; gap: .6rem; }
  .snippet { margin: .45rem 0 .3rem; white-space: pre-wrap; overflow-wrap: anywhere; }
  .tier { font-weight: 600; }
  .tier.authoritative { color: var(--accent); }
  .tier.reported { color: var(--warn); }
  .tier.machine { color: var(--mach); }
  .actions { display: flex; flex-wrap: wrap; gap: .35rem; margin-top: .4rem; }
  .actions button { font-size: .78rem; padding: .2rem .45rem; }
  .actions button.done { border-color: var(--accent); color: var(--accent); }
  details { margin-top: .4rem; }
  summary { cursor: pointer; color: var(--dim); font-size: .8rem; }
  table.terms { border-collapse: collapse; font-size: .8rem; margin-top: .3rem; }
  table.terms td { padding: .1rem .6rem .1rem 0; }
  pre {
    white-space: pre-wrap; overflow-wrap: anywhere; background: var(--card);
    border: 1px solid var(--line); border-radius: 4px; padding: .75rem;
    font: .8rem/1.45 ui-monospace, SFMono-Regular, Menlo, monospace;
  }
  .note { color: var(--dim); font-size: .82rem; margin: .75rem 0; }
  .needs-token { border: 1px solid var(--warn); border-radius: 4px; background: var(--card);
                 padding: .6rem .75rem; margin: .75rem 0; font-size: .85rem; color: var(--fg); }
  .needs-token b { color: var(--warn); }
  #token.wanted { border-color: var(--warn); }
  h2.minor { font-size: .82rem; text-transform: uppercase; letter-spacing: .06em;
             color: var(--dim); margin: 2rem 0 .5rem; font-weight: 600; }
  /* The strip used to run its numbers together -- "3064 threads41857 documents".
     A flex row for the totals, and the per-pass lines stacked underneath, because
     one line per pass is the only shape that stays readable past two repositories. */
  .strip .passes { flex-basis: 100%; display: flex; flex-direction: column; gap: .15rem;
                   margin-top: .35rem; }
  .strip .pass { font-variant-numeric: tabular-nums; padding-left: .75rem; }
  .strip .repo { display: flex; flex-direction: column; gap: .15rem; margin-top: .35rem; }
  .strip .repo-name { font-weight: 600; }
  .warn-text { color: var(--warn); }
  .dim-text { color: var(--dim); }
  /* A pass that is working right now. The page polls health, so this appears and goes
     without anybody reloading. */
  .working { color: var(--accent); }

  details.guide { margin: 1rem 0; border: 1px solid var(--line); border-radius: 4px;
                  background: var(--card); }
  details.guide > summary { padding: .55rem .75rem; cursor: pointer; font-size: .85rem;
                            color: var(--accent); }
  details.guide .body { padding: 0 .95rem .85rem; font-size: .85rem; }
  details.guide h3 { font-size: .82rem; margin: 1rem 0 .35rem; text-transform: uppercase;
                     letter-spacing: .05em; color: var(--dim); }
  details.guide pre { background: var(--bg); border: 1px solid var(--line); border-radius: 3px;
                      padding: .55rem .7rem; overflow-x: auto; font-size: .78rem; margin: .3rem 0; }
  details.guide p { margin: .4rem 0; color: var(--dim); }
  .samples { display: flex; flex-wrap: wrap; gap: .4rem; margin: .3rem 0 .2rem; }
  .samples button { font-size: .78rem; text-align: left; }
  .error { color: var(--mach); }
</style>

<main>
  <h1>
    <!-- The mark from docs/relore.png, redrawn rather than embedded: the PNG is 1448px
         and 717KB of mostly white margin, and its ink is a fixed dark that disappears
         on this page's dark theme. As geometry it is under 1KB, sharp on any display,
         and `currentColor` makes it follow the theme the way the wordmark does. Below
         about 30px the node rings close up, so the size is not a free choice. -->
    <svg class="mark" viewBox="0 0 40 44" width="27" height="30" aria-hidden="true"
         fill="none" stroke-linecap="round">
      <path d="M0 30.2 Q7.4 27.2 14.7 31.4 T36 26.6" stroke="#beb2a7" stroke-width="1.7"/>
      <path d="M0 34.6 Q7.4 31.6 14.7 35.8 T36 31" stroke="#c8633a" stroke-width="1.7"/>
      <path d="M0 39 Q7.4 36 14.7 40.2 T36 35.4" stroke="currentColor" stroke-width="1.7"/>
      <path d="M14.7 3.3 V31.4" stroke="currentColor" stroke-width="2.6"/>
      <path d="M14.7 18.5 Q23.4 18.5 28.4 12.4" stroke="currentColor" stroke-width="2.6"/>
      <g stroke="currentColor" stroke-width="2.7" fill="var(--bg)">
        <circle cx="14.7" cy="3.3" r="2.65"/>
        <circle cx="14.7" cy="18.5" r="2.65"/>
        <circle cx="14.7" cy="31.4" r="2.65"/>
      </g>
      <circle cx="30.6" cy="10.6" r="2.35" stroke="#c8633a" stroke-width="2.1" fill="var(--bg)"/>
    </svg>
    <span>relore <small id="backend">…</small></span>
  </h1>

  <!-- Shown only once the daemon has actually refused us. A page that opens by
       demanding a token teaches nothing; a page that opens with a raw 401 JSON blob
       teaches less. This appears when it is true, and says what to do about it. -->
  <div class="needs-token" id="needs-token" hidden>
    <b>This index requires a token.</b>
    Paste one below to search. Ask whoever runs this daemon — it is one entry of
    <code>RELORE_API_TOKENS</code>, the part before the first <code>:</code>.
  </div>

  <form id="search">
    <input name="q" placeholder="error text, a symbol, or a question" autofocus>
    <select name="kind">
      <option value="">any kind</option>
      <option value="failure">failure</option>
      <option value="precedent">precedent</option>
      <option value="rationale">rationale</option>
    </select>
    <select name="trust">
      <option value="">human tiers</option>
      <option value="authoritative">authoritative only</option>
      <option value="machine">machine (our own bots)</option>
    </select>
    <input name="file" placeholder="path filter" size="16">
    <!-- Options are added by the health call, which is the only thing that knows what is
         indexed. It can only narrow the token's scope, so the list is what you may already
         see and "all repositories" is not a wildcard, it is "do not filter". -->
    <select name="repo">
      <option value="">all repositories</option>
    </select>
    <input name="labels" placeholder="labels, comma-separated" size="18">
    <select name="sort">
      <option value="relevance">best match</option>
      <option value="newest">newest first</option>
    </select>
    <button class="primary">search</button>
    <button type="button" id="toggle-raw">view as the model sees it</button>
  </form>
  <div class="note" id="token-field">
    <label>token
      <input id="token" size="24" placeholder="bearer token, if required"></label>
  </div>

  <!-- The answer sits directly under the form, ABOVE the guide. It used to be below
       it, and the guide is long enough that clicking "try one" scrolled the results
       off the bottom of the page: the search ran, the hits rendered, and the visitor
       saw an unchanged page of instructions. A result the caller cannot see is the
       same failure as no result. -->
  <div id="message" class="note"></div>
  <pre id="raw" hidden></pre>
  <div id="hits"></div>

  <!-- Everything below is for a first-time visitor. The tool assumed you already
       knew what to type, which is a poor first impression for something whose whole
       argument is that the knowledge exists but nobody can reach it. -->
  <details class="guide" id="guide">
    <summary>How to use this — examples, the CLI, and wiring it into an agent</summary>
    <div class="body">

      <h3>Try one</h3>
      <p>These run against this index. The three <em>kinds</em> are not cosmetic: each
         carries its own trust floor, because a report and a judgement are different
         claims.</p>
      <div class="samples" id="samples"></div>

      <!-- `why` is the one verb a search box cannot express: its question is a *place*
           in the tree, not words. Printing the examples would have taught it the way
           the samples above were once printed -- and a query you have to retype is one
           nobody tries -- so these run. -->
      <h3>Why is this line the way it is?</h3>
      <p><code>relore why PATH:LINE</code> blames the line, then answers from the index:
         the pull request that carried the commit, and the review comments anchored near
         it. Blame reads a clone of <b>HEAD</b>, so line numbers are today's.</p>
      <form id="why-form">
        <input name="at" placeholder="src/transformers/masking_utils.py:1" size="46">
        <select name="repo"><option value="">pick a repository</option></select>
        <button class="primary">why</button>
      </form>
      <div class="samples" id="why-samples"></div>
      <pre id="why-out" hidden></pre>

      <div id="token-guide">
      <h3>Tokens</h3>
      <p>Every endpoint that returns content is scoped to a bearer token, so a daemon
         with tokens configured will refuse both the search and the health strip until
         you paste one. A token is one entry of <code>RELORE_API_TOKENS</code> — the
         part before the first <code>:</code>. It is remembered in this browser only.</p>
      </div>

      <h3>The CLI</h3>
      <p>The base install is a read-only HTTP client — no database driver, no parser —
         so it is safe to drop into a constrained agent sandbox. It installs from
         <code>main</code>, not from PyPI: there is no release yet, so
         <code>pip install relore</code> would fetch whatever else owns that name.
         The client and the daemon must be the <em>same version</em> — this one refuses a
         request from any other, rather than answering it with a contract the caller does
         not have. If yours is refused, upgrade it; if it says the daemon is behind, this
         deployment is the thing to redeploy.</p>
      <pre id="cli-setup">pip install git+https://github.com/huggingface/relore
export RELORE_API=https://relore.example.org   # this page's own origin
export RELORE_REPO=&lt;owner/name&gt;                # the default for --repo
export RELORE_TOKEN=&lt;your token&gt;              # only if this daemon requires one</pre>
      <p><code>RELORE_REPO</code> is worth setting first. This daemon indexes more than
         one repository, and the verbs that answer about a number or a path refuse a bare
         one rather than guess between them — so without a default, every call in a
         single-repository session carries <code>--repo</code>.</p>
      <p>Opened in a browser that block names <em>this</em> daemon and the version it
         wants, so it can be pasted without editing.</p>
      <pre>relore search "AttributeError: 'NoneType' object has no attribute 'shape'" --kind failure
relore search "why is this cast here" --kind rationale --file src/transformers/masking_utils.py
relore inflight 48630 --repo huggingface/transformers
relore thread 47720 --focus "cropping" --repo huggingface/transformers
relore status</pre>
      <p><code>--repo</code> is required on a bare number whenever more than one
         repository is in scope — a number alone would be ambiguous, and guessing would
         silently answer about the wrong project. The error lists what to choose from.</p>
      <p><code>--compact</code> trims snippets for a tight context budget, and serves a
         <em>truncated</em> changed-file list as its shape — <code>92 under
         src/transformers/ across 33 directories</code> — rather than as 100 paths whose
         absence proves nothing. A client may ask for less; never for more. No count and
         no caveat is ever trimmed, and <code>--json</code> carries every path either
         way.</p>
      <p>The four globals — <code>--json</code>, <code>--compact</code>,
         <code>--plain</code>, <code>--api</code> — are accepted on <b>every verb</b> and
         on either side of it. What <code>--compact</code> shortens is whatever the verb
         has to shorten: a snippet, a review comment, a matched line, a truncated file
         list. Where a verb prints one short row per result it changes nothing, and
         <code>relore symbol</code> keeps its body whole either way — the body is the
         answer.</p>

      <h3>The code lens</h3>
      <p>The same questions, asked of the tree instead of the conversation. Four read this
         daemon's working clone, checked out at <b>HEAD</b> — so they answer about the
         project as it is, which is what a caller cannot get from the index:</p>
      <pre>relore copies compute_default_rope_parameters --repo huggingface/transformers
relore symbol compute_default_rope_parameters --repo huggingface/transformers
relore grep 'partial_rotary_factor' --repo huggingface/transformers --path 'src/**/modeling_*.py'
relore why src/transformers/masking_utils.py:1 --repo huggingface/transformers</pre>
      <p><code>copies</code> is the one to reach for first on a repository that duplicates
         model code on purpose: it groups every definition by whether the bodies agree, so
         the outlier is the answer rather than something to spot in a list of 186.</p>
      <p>Three more run against <em>your own checkout</em> and need no daemon at all —
         they read the working tree you are editing, including the branch this index has
         never seen:</p>
      <pre>relore map                      # ranked repo map of the local checkout
relore defs src/transformers/masking_utils.py
relore refs compute_default_rope_parameters</pre>
      <p><code>defs</code> and <code>refs</code> take <code>--repo</code> to ask the
         daemon's clone instead. <code>refs</code> reports each occurrence by kind — call,
         definition, attribute, name — because a reference index that returns only call
         sites answers "find every affected site" with a fraction of them and no way to
         tell. A repository with no clone answers the server-side verbs with a sentence
         saying so; nothing else degrades.</p>

      <h3>Give it to Claude Code or Codex</h3>
      <p>There is no MCP server, on purpose: any agent with a shell can already call
         this. Put the variables in the agent's environment and one paragraph in
         the file it reads at startup — <code>CLAUDE.md</code>, <code>AGENTS.md</code>,
         or whatever your harness uses. The same paragraph is in
         <code>relore --help</code>, which is the copy that cannot be lost to a fetch.</p>
      <pre id="agent-snippet">/*GUIDANCE*/</pre>
      <p>Retrieved text is wrapped in an untrusted-content envelope before it reaches a
         model. It is data, never instructions — and the envelope is applied by this
         server, not by the client, because an unknown client cannot be assumed to add
         it.</p>

      <h3>Source</h3>
      <p><a href="https://github.com/huggingface/relore">github.com/huggingface/relore</a>
         — the build plan and the evidence base it argues from are held with the
         deployment that commissioned them.</p>
    </div>
  </details>

  <!-- The index-health strip lives at the bottom, not under the title. It answers
       "is the index current?", which is a question you ask *about* a result set --
       so it belongs after one, not in front of the search box every visitor meets
       first. -->
  <h2 class="minor">index</h2>
  <div class="strip" id="health">…</div>
</main>

<script>
const $ = (s) => document.querySelector(s);

// These four sit at the top because `const` is hoisted but not initialized, and the
// sample-button strip below renders `esc(...)` at load time. Declared after their
// first use that is a ReferenceError, and it aborts the whole script -- so the search
// form loses its submit handler and the browser falls back to a native GET of
// `/?q=...`. The page still renders, the access log still says 200, and nothing
// searches. Shipped that way on 2026-09-09 and found by a person opening the page,
// which is exactly what section 8 says the UI is for.
// Whether this daemon wants a token, answered by the daemon rather than guessed from a
// 401. `page()` rewrites the literal below. When it is false the token field, the token
// section of the guide and the `RELORE_TOKEN` line of the setup snippet are all removed:
// telling somebody to paste a credential that is not read is worse than saying nothing,
// and it is the kind of instruction people follow anyway and then debug.
const AUTH_REQUIRED = /*AUTH*/true;
const esc = (s) => String(s ?? "").replace(/[&<>"']/g,
  (c) => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
const split = (s) => String(s || "").split(",").map((x) => x.trim()).filter(Boolean);
const quote = (s) => /[\\s"']/.test(s) ? "'" + String(s).replace(/'/g, "'\\\\''") + "'" : s;
const short = (iso) => iso ? String(iso).slice(0, 16).replace("T", " ") : "never";
// "8s ago" answers the question a timestamp only helps you compute: is this thing moving?
const ago = (iso) => {
  if (!iso) return "never";
  const seconds = Math.max(0, (Date.now() - Date.parse(iso)) / 1000);
  for (const [unit, size] of [["d", 86400], ["h", 3600], ["m", 60]]) {
    if (seconds >= size) return `${Math.floor(seconds / size)}${unit} ago`;
  }
  return `${Math.floor(seconds)}s ago`;
};

const token = $("#token");
token.value = localStorage.getItem("relore-token") || "";
// The token input sits outside the search form on purpose -- it is not a query field --
// so Enter in it triggers no implicit submit, and a refusal focuses it (see the 401
// handler). Without the keydown below, the page answers a pasted token by doing nothing,
// which reads as ignoring it. Committing one has to do the work explicitly: remember it,
// re-check the daemon so the banner and the health strip stop saying it is missing, and
// re-run the query that was refused.
let applied = token.value;
function applyToken() {
  localStorage.setItem("relore-token", token.value);
  applied = token.value;
  health();
  const f = $("#search");
  if (f.q.value.trim()) f.requestSubmit();
}
// Blur commits a token only when it actually changed; Enter always retries, because
// pressing it again is how someone asks for another attempt at the same value.
token.addEventListener("change", () => { if (token.value !== applied) applyToken(); });
token.addEventListener("keydown", (event) => {
  if (event.key !== "Enter") return;
  event.preventDefault();
  applyToken();
});

// A 401 is not an error to display, it is a question to ask. Everything else is
// shown as whatever the server said, because those are real failures worth reading.
const AUTH_HINT = "this index requires a token — paste one below";
// A 426 here can only be a page the browser kept from a previous version of the daemon:
// the page and the daemon that served it always agree. Saying "reload" is the whole fix,
// and the daemon's own sentence (which talks about `pip install`) is not.
const STALE_PAGE = "this page was served by an older relore — reload it (⌘/Ctrl-Shift-R)";
function readable(status, body) {
  if (status === 401 || status === 403) return AUTH_HINT;
  if (status === 426) return STALE_PAGE;
  try {
    const parsed = typeof body === "string" ? JSON.parse(body) : body;
    return String(parsed.detail || parsed.message || body);
  } catch (_) { return String(body); }
}
function wantToken(yes) {
  $("#needs-token").hidden = !yes;
  $("#token").classList.toggle("wanted", yes);
}

// The page is a client of the same API as `relore`, so it declares its version like one.
// `page()` rewrites the literal to the daemon's own version, which means the two agree by
// construction -- until a browser reuses a page from before a deploy, which is exactly
// the case the handshake should catch rather than answer with a stale renderer.
const CLIENT_VERSION = "/*VERSION*/0.0.0";
const headers = () => {
  const h = {"content-type": "application/json", "x-relore-client": CLIENT_VERSION};
  if (token.value.trim()) h["authorization"] = "Bearer " + token.value.trim();
  return h;
};

let showRaw = false;
let last = null;

$("#toggle-raw").addEventListener("click", () => {
  showRaw = !showRaw;
  $("#toggle-raw").classList.toggle("done", showRaw);
  paint();
});

async function health() {
  try {
    const r = await fetch("/api/v1/status", {headers: headers()});
    if (!r.ok) throw new Error(readable(r.status, await r.text()));
    wantToken(false);
    const s = await r.json();
    $("#backend").textContent =
      s.backend.name + " / " + s.backend.ranking + " · v" + s.version;
    const n = (v) => Number(v).toLocaleString();
    // Grouped by repository, because the flat list interleaved them: with three repos
    // and five passes each, finding "is transformers current" meant reading fifteen
    // lines in an order nobody chose. And a pass that is *working right now* says so --
    // `last_run_at` moves per committed thread while `last_ok_at` moves only when the
    // pass finishes, so a run newer than the ok is a pass in flight, and a cursor says
    // how far it got. Stale is the same signal read later, which is why the age is
    // printed rather than a bare timestamp: "8s ago" is running, "2d ago" is stopped.
    const byRepo = new Map();
    for (const p of s.passes || []) {
      if (!byRepo.has(p.repo)) byRepo.set(p.repo, []);
      byRepo.get(p.repo).push(p);
    }
    const working = (p) =>
      p.last_run_at && (!p.last_ok_at || p.last_run_at > p.last_ok_at);
    const passes = [...byRepo.entries()].sort().map(([repo, rows]) => {
      const live = rows.filter(working);
      const lines = rows.sort((a, b) => a.pass.localeCompare(b.pass)).map((p) => {
        const at = p.cursor ? ` · at ${esc(p.cursor)}` : "";
        const state = working(p)
          ? `<b class="working">indexing</b> ${ago(p.last_run_at)}${at}`
          : `ok ${short(p.last_ok_at)}`;
        return `<span class="pass"><b>${esc(p.pass)}</b>` +
               ` high-water ${short(p.high_water)} · ${state}` +
               (p.note ? ` (${esc(p.note)})` : "") + `</span>`;
      }).join("");
      const badge = live.length
        ? `<b class="working">${live.length} indexing</b>`
        : `<span class="dim-text">idle</span>`;
      return `<div class="repo"><div class="repo-name">${esc(repo)} ${badge}</div>` +
             `${lines}</div>`;
    }).join("");
    const sampled = (s.samples || []).map((x) =>
      `<span class="pass warn-text">sample: ${esc(x.repo)} ${esc(x.thread_type)}s` +
      ` from ${short(x.indexed_from)} only — NOT full history</span>`).join("");
    $("#health").innerHTML =
      `<span><b>${n(s.threads)}</b> threads</span>` +
      `<span><b>${n(s.documents)}</b> documents</span>` +
      `<span><b>${n(s.raw_objects)}</b> staged</span>` +
      `<div class="passes">${esc(s.passes_note || "")}${passes}${sampled}</div>`;
    // The repo filter's options, deduped: a repo reports one pass per phase, so `passes`
    // names most of them several times. Rebuilt on every health call rather than once,
    // because the first call may have been refused and the second is the one that knows --
    // and the current choice is restored, so re-checking does not silently drop a filter
    // the user set and then search something else.
    const known = [...new Set((s.passes || []).map((p) => p.repo))].sort();
    const options = known.map((r) => `<option value="${esc(r)}">${esc(r)}</option>`).join("");
    for (const [chooser, empty] of [[$("#search").repo, "all repositories"],
                                    [$("#why-form").repo, "pick a repository"]]) {
      const chosen = chooser.value;
      chooser.innerHTML = `<option value="">${empty}</option>` + options;
      if (known.includes(chosen)) chooser.value = chosen;
    }
  } catch (e) {
    if (e.message === AUTH_HINT) wantToken(true);
    $("#health").innerHTML = `<span class="error">index health: ${esc(e.message)}</span>`;
  }
}

// Samples are clickable rather than printed: the fastest way to learn what this
// answers well is to see one land, and a query you have to retype is one nobody tries.
const SAMPLES = [
  {label: "a pasted traceback → the threads that explain it",
   q: "AttributeError: 'NoneType' object has no attribute 'shape'", kind: "failure"},
  {label: "why is the code like this? (maintainers only)",
   q: "why is the attention mask cast here", kind: "rationale",
   file: "src/transformers/masking_utils.py"},
  {label: "how is this done here? → prior work as precedent",
   q: "add a new model configuration", kind: "precedent"},
  {label: "what did our own bots claim?",
   q: "review", trust: "machine"},
];

// Real places in `transformers`, each answering a different shape of the question: what
// a file is for, and what one line of logic was for. Line 1 is the stable one -- it is
// the file's own introduction, and it cannot drift the way a line in the middle does
// when the clone moves to a newer HEAD.
//
// It was also, until relore#71, the slowest call this API makes. `why`'s origin pass
// pickaxes candidate words out of the line and the comment attached above it, and on line
// 1 that block is the licence header: five `git log -S` passes on `Copyright`,
// `HuggingFace`, `rights`, `reserved` and `team`, each a full walk of the file's history,
// for an answer that was empty and knowably so. Two of these three buttons are that line,
// which is why this strip was where the slowness got noticed. A line with no code on it
// now declines the pickaxe and says it did, so the sample is one blame rather than six
// subprocesses -- measured against production at 3.98s and 6.35s before.
const WHY_SAMPLES = [
  {label: "what is this file for?",
   at: "src/transformers/masking_utils.py:1", repo: "huggingface/transformers"},
  {label: "why is this line written this way?",
   at: "src/transformers/modeling_rope_utils.py:180", repo: "huggingface/transformers"},
  {label: "a line from 2023 — blame still reaches it",
   at: "src/transformers/models/llama/modeling_llama.py:1", repo: "huggingface/transformers"},
];

const whyForm = $("#why-form");
$("#why-samples").innerHTML = WHY_SAMPLES.map((s, i) =>
  `<button type="button" data-i="${i}">${esc(s.label)}</button>`).join("");
$("#why-samples").addEventListener("click", (event) => {
  const button = event.target.closest("button");
  if (!button) return;
  const s = WHY_SAMPLES[Number(button.dataset.i)];
  whyForm.at.value = s.at;
  // The repository is part of the example: the clone it blames is per repository, and a
  // path from one answers 404 against another.
  whyForm.repo.value = s.repo;
  whyForm.requestSubmit();
});

whyForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const out = $("#why-out");
  const at = String(whyForm.at.value || "").trim();
  // Split on the LAST colon: a path may contain one, a line number may not.
  const cut = at.lastIndexOf(":");
  const path = cut > 0 ? at.slice(0, cut) : at;
  const line = cut > 0 ? Number(at.slice(cut + 1)) : NaN;
  if (!path || !Number.isInteger(line) || line < 1) {
    out.hidden = false;
    out.textContent = "Give it PATH:LINE, e.g. src/transformers/masking_utils.py:1";
    return;
  }
  const query = new URLSearchParams({path, line: String(line), render: "true"});
  if (whyForm.repo.value) query.set("repo", whyForm.repo.value);
  out.hidden = false;
  out.textContent = `why ${at} …`;
  try {
    const r = await fetch(`/api/v1/why?${query}`, {headers: headers()});
    const body = await r.json();
    // 503 is the designed answer for a repository with no clone, and 404 for a line that
    // is not in the tree at HEAD. Both say what to do, so both are shown as themselves.
    out.textContent = r.ok ? body.rendered : readable(r.status, body);
    if (!r.ok && (r.status === 401 || r.status === 403)) wantToken(true);
  } catch (e) {
    out.textContent = String(e.message || e);
  }
});

const form = $("#search");
$("#samples").innerHTML = SAMPLES.map((s, i) =>
  `<button type="button" data-i="${i}">${esc(s.label)}</button>`).join("");
$("#samples").addEventListener("click", (event) => {
  const button = event.target.closest("button");
  if (!button) return;
  const s = SAMPLES[Number(button.dataset.i)];
  form.q.value = s.q;
  form.kind.value = s.kind || "";
  form.trust.value = s.trust || "";
  form.file.value = s.file || "";
  // Reset the two controls a sample does not set, for the same reason it overwrites the
  // three above: a sample has to *land*, and one filtered to the other repository or
  // ordered by date teaches the wrong lesson about what this answers well.
  form.repo.value = "";
  form.sort.value = "relevance";
  form.requestSubmit();
  form.scrollIntoView({behavior: "smooth", block: "start"});
});

// The setup lines name *this* daemon, so they can be pasted without editing. They are in
// the markup as static text first, with a placeholder origin, and rewritten here -- a
// block that exists only inside a template literal is invisible to `curl`, to an
// HTML-to-markdown fetch, and to most agent page fetchers, and the reader who could not
// see these two names went and set `RELORE_API_TOKENS` instead, which is the *server
// operator's* variable (huggingface/relore#30).
if (!AUTH_REQUIRED) {
  for (const id of ["token-field", "token-guide"]) {
    const el = $("#" + id);
    if (el) el.hidden = true;
  }
}
$("#cli-setup").textContent =
  `pip install git+https://github.com/huggingface/relore   # must be ${CLIENT_VERSION}` +
  `\nexport RELORE_API=${location.origin}` +
  (AUTH_REQUIRED ? `\nexport RELORE_TOKEN=<your token>   # only if this daemon requires one` : ``);

// Remember whether the guide is open. Open on a first visit -- somebody who has never
// seen this page has no way to guess what it answers -- and never again after that.
const guide = $("#guide");
guide.open = localStorage.getItem("relore-guide") !== "closed";
guide.addEventListener("toggle", () =>
  localStorage.setItem("relore-guide", guide.open ? "open" : "closed"));

$("#search").addEventListener("submit", async (event) => {
  event.preventDefault();
  const f = new FormData(event.target);
  const body = {
    query: f.get("q") || "",
    kind: f.get("kind") || null,
    trust: f.get("trust") || null,
    files: split(f.get("file")),
    // An empty select means "do not filter", not "every repository": the server
    // intersects this with the token's scope and can only ever narrow it.
    repos: f.get("repo") ? [f.get("repo")] : [],
    labels: split(f.get("labels")),
    sort: f.get("sort") || "relevance",
    // Always asked for: the toggle switches what is displayed, not what the server did,
    // so "view as the model sees it" cannot show a different query's rendering.
    render: true,
  };
  $("#message").textContent = "searching…";
  try {
    const r = await fetch("/api/v1/search", {
      method: "POST", headers: headers(), body: JSON.stringify(body),
    });
    const payload = await r.json();
    if (!r.ok) throw new Error(readable(r.status, payload));
    wantToken(false);
    last = {payload, request: body};
    paint();
  } catch (e) {
    last = null;
    if (e.message === AUTH_HINT) { wantToken(true); $("#token").focus(); }
    $("#hits").innerHTML = "";
    $("#message").innerHTML = `<span class="error">${esc(e.message)}</span>`;
  }
});

function paint() {
  $("#raw").hidden = !showRaw;
  if (!last) { $("#message").textContent = ""; return; }
  const {payload, request} = last;
  const floor = (payload.query.trust_floor || []).join(", ");
  const scope = payload.query.repos || [];
  $("#message").textContent =
    `${payload.count} hit${payload.count === 1 ? "" : "s"} · trust floor: ${floor}` +
    // Say when the order is not relevance: a date-sorted page of weak matches otherwise
    // reads as a broken ranking rather than as the question that was asked.
    (payload.query.sort === "newest" ? " · newest first" : "") +
    (scope.length === 1 ? ` · ${scope[0]} only` : "") +
    // The fail-closed case, named. Filtering to a repository this token cannot see leaves
    // no repository in scope, and "nothing matched" would blame the corpus for it.
    (scope.length === 0
      ? " · no repository in scope — that filter is outside this token's reach"
      : payload.count === 0 ? " · nothing matched" : "");
  $("#raw").textContent = payload.rendered || "";
  $("#hits").innerHTML = showRaw ? "" : payload.hits.map((h, i) => card(h, i, request)).join("");
}

function card(h, i, request) {
  const terms = Object.entries(h.breakdown || {})
    .map(([k, v]) => `<tr><td>${esc(k)}</td><td>${v}</td></tr>`).join("");
  const cli = cliFor(h, request);
  return `<div class="hit" data-i="${i}">
    <h2><a href="${esc(h.url || "#")}" target="_blank" rel="noreferrer noopener">
      ${esc(h.repo)}#${h.number}</a> — ${esc(h.title || "")}</h2>
    <div class="meta">
      <span class="tier ${esc(h.trust)}">${esc(h.trust)}</span>
      <span>${esc(h.age)}</span><span>${esc(h.source_type)}</span>
      <span>${h.author ? "@" + esc(h.author) : ""}</span>
      <span>score ${h.score ?? "—"}</span>
    </div>
    <div class="snippet">${esc(h.snippet || "")}</div>
    <div class="actions">
      <button data-verdict="relevant" data-i="${i}">relevant</button>
      <button data-verdict="not_relevant" data-i="${i}">not relevant</button>
      <button data-verdict="decisive" data-i="${i}">decisive</button>
      <button data-copy="${esc(cli)}">copy as CLI</button>
    </div>
    <details><summary>score breakdown</summary>
      <table class="terms">${terms || "<tr><td>no terms yet</td></tr>"}</table>
    </details>
  </div>`;
}

function cliFor(h, request) {
  const parts = ["relore search"];
  if (request.query) parts.push(quote(request.query));
  if (request.kind) parts.push("--kind " + request.kind);
  if (request.trust) parts.push("--trust " + request.trust);
  (request.files || []).forEach((f) => parts.push("--file " + quote(f)));
  (request.labels || []).forEach((l) => parts.push("--label " + quote(l)));
  return parts.join(" ");
}

document.addEventListener("click", async (event) => {
  const copy = event.target.closest("button[data-copy]");
  if (copy) {
    await navigator.clipboard.writeText(copy.dataset.copy);
    copy.classList.add("done");
    return;
  }
  const vote = event.target.closest("button[data-verdict]");
  if (!vote || !last) return;
  const hit = last.payload.hits[Number(vote.dataset.i)];
  const r = await fetch("/api/v1/label", {
    method: "POST", headers: headers(),
    body: JSON.stringify({
      query: last.request.query, kind: last.request.kind, repo: hit.repo,
      number: hit.number, source_type: hit.source_type, verdict: vote.dataset.verdict,
      // The filters travel with the verdict: the same text under a different `--file` is
      // a different question, and section 10's set has to be able to tell them apart.
      filters: {
        files: last.request.files || [], symbols: last.request.symbols || [],
        errors: last.request.errors || [], tests: last.request.tests || [],
      },
    }),
  });
  if (r.ok) {
    vote.closest(".actions").querySelectorAll("[data-verdict]")
      .forEach((b) => b.classList.remove("done"));
    vote.classList.add("done");
  } else {
    const body = await r.json().catch(() => ({}));
    $("#message").innerHTML =
      `<span class="error">labelling: ${esc(body.detail || r.status)}</span>`;
  }
});

health();
// The strip is a live view, not a snapshot: a pass that starts or finishes appears and
// clears on its own, which is the whole point of showing what is indexing. Only while the
// tab is visible -- `/api/v1/status` counts three large tables, and a background tab
// paying for that every half minute is a cost nobody asked for.
setInterval(() => {
  if (document.visibilityState === "visible") health();
}, 30000);
</script>
"""


def page(*, auth_required: bool = True) -> str:
    """The page, told whether this daemon wants a token, and stamped with its version.

    ``auth_required`` defaults to ``True`` so a caller that forgets to ask renders the page
    that *mentions* a credential rather than the one that hides it -- the harmless
    direction of a wrong guess.

    The version stamp has no such default to get wrong: the page is served by the daemon
    whose version it declares, so it passes the handshake (:mod:`relore.wire`) by
    construction, and a page that fails it is a page the browser cached across a deploy.
    """
    out = _PAGE
    if not auth_required:
        # The script hides every mention of a token it can reach, but the setup block is
        # now static text so that a reader with no script can find the two variable names
        # (huggingface/relore#30) -- and that reader has to be told the same thing.
        out = out.replace("/*AUTH*/true", "/*AUTH*/false").replace(
            "\nexport RELORE_TOKEN=&lt;your token&gt;"
            "              # only if this daemon requires one",
            "",
        )
    # The agent paragraph is rendered from :mod:`relore.guidance` rather than written here
    # (issue #40). It used to be a hardcoded block in this file, and it was the best
    # guidance in the project sitting on the surface least likely to arrive intact: a
    # summarizing fetch of this page is how one field run never saw it at all. Same text,
    # same module, now also under `relore --help` where it needs no network.
    out = out.replace("/*GUIDANCE*/", escape(guidance.agent_paragraph()))
    out = out.replace('"/*VERSION*/0.0.0"', json.dumps(__version__))
    out = out.replace("/*FAVICON*/", _FAVICON)
    return out.strip()
