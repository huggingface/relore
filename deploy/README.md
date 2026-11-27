# Deploying relore

The chart runs four things: a Postgres 16 StatefulSet with pgvector, `relored serve`
(the API and the section 8 web UI), one `relored poll` Deployment per indexed
repository, and a nightly `pg_dump` CronJob. `relored migrate` runs as a Helm hook
before any of them, on every install and upgrade.

**The deployment is still called `ghlore`.** The project was renamed to relore on
2026-09-14; the release, its namespace, its Postgres role and database, the
`ghlore-secrets` Secret, the `/var/lib/ghlore` volume paths and the
`ghlore.huggingface.tech` hostname were not. Each of those names a live object —
renaming them in a values file does not rename the object, it points the chart at one
that does not exist, and for the namespace it would stand up a second, empty install
beside the real one. What did move is everything a caller sees: the image
(`ghcr.io/huggingface/relore`), the commands (`relore`, `relored`), the environment
(`RELORE_*`) and the wire headers. The Secret's *keys* moved with the code —
`GHLORE_API_TOKENS` is now `RELORE_API_TOKENS`, which a token-authenticated deployment
must rename in its secret store; `trustNetwork: true` empties it in the pod spec and is
unaffected.

**The production values are not in this repository.** `deploy/helm/values.yaml` is
the chart's *shape* and safe defaults; the environment lives in the operator's own
repository and is passed with `-f`. `deploy.sh` refuses to run without it, because
a values file that silently falls back to chart defaults produces a healthy pod
with settings quietly unset.

```bash
deploy/scripts/deploy.sh --plan -f ../env/prod.yaml          # what it manages
deploy/scripts/deploy.sh --dry-run -f ../env/prod.yaml       # full manifest
deploy/scripts/deploy.sh --context <ctx> -n ghlore -f ../env/prod.yaml
deploy/scripts/logs.sh -c poll --repo huggingface/serge --since 2h
```

## What has to exist first

| Thing | Why |
| --- | --- |
| A Secret named by `existingSecret` | `GITHUB_TOKEN` (`issues:read` + `pull_requests:read`, **never** write — section 11), `RELORE_API_TOKENS`, `POSTGRES_PASSWORD`. Produce it with the environment's secret operator via the `externalSecret` block, or by hand. |
| `poll.repos` | The repositories to index — and section 11's allowlist, which is re-checked per thread. An allowlist, never a denylist. |
| `botAccounts` | Section 6.2's bot-account list. Leaving it empty in a deployment that indexes its own agent's comments reintroduces what section 11 refused. |

## After the first deploy

1. **Backfill, once per repository.** The poll loop keeps an index current; it does
   not create one. `relored backfill --repo <repo>` is a restartable day for a
   large repository, `relored sample --repo <repo> --since <date>` a bounded window.
2. **Resolve authority.** `relored authority --repo <repo>`. Without it every
   org-team maintainer reads as an unresolved `MEMBER` and the `authoritative`
   tier is empty — measured on `huggingface/serge`: 0 of 415 documents before, 288
   after. Rationale queries answer from that tier, so until this runs they return
   nothing.
3. **Check `relored status`.** It prints the backend and its capabilities, the
   per-pass high-water marks, the sample window if the corpus is one, and the
   count of superseded documents kept (section 14.1).

## Changing the bot list or the allowlist

Both live in the ConfigMap. `trust` and `author_is_bot` are `DERIVED_COLUMNS`
(section 5.1), so an edit does not move existing documents by itself — re-derive
after deploying:

```bash
kubectl -n ghlore exec deploy/ghlore -- relored derive --repo <repo>
```

That writes only the documents whose tier actually changed.

## Turning on the semantic leg

Off by default. `embed.model: BAAI/bge-small-en-v1.5` (the model the image carries)
puts `RELORE_EMBED_MODEL` on the ConfigMap, which serve and every poll pod read:

1. Raise `poll.resources.limits.memory` to `3Gi`; a poll pod holds ~1.7 GiB while
   it embeds. Serve needs ~400 MiB more and fits its default.
2. Deploy, then backfill once per repository. The poll loop embeds what each pass
   rewrites; it does not embed the history:

   ```bash
   deploy/scripts/pass.sh embed --repo <repo> --context <ctx> -f
   ```

   It is idempotent and resumable: a document is pending while its
   `embedding_model` differs from the configured one, which is also what a model
   change or an edited comment looks like.
3. `relore status` prints `semantic  <model> · N of M documents embedded`. The leg
   reads only rows embedded by the configured model, so until N reaches M it
   answers from part of the corpus — that line is how you tell.
