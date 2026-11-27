#!/usr/bin/env bash
# Run one long `relored` pass over a repository, as a Kubernetes Job.
#
# **Not `kubectl exec`.** Build plan section 3 sizes a backfill of a large repository
# at about a restartable *day*: ~3,600 REST requests and a GraphQL pass that is
# point-limited rather than request-limited, so 6-12 hours. A full `derive` is
# cheaper -- local CPU and one transaction per thread, no network -- but 48,000
# threads is still tens of minutes. An exec dies with the terminal that started it;
# a Job survives, restarts where it stopped, and can be watched from anywhere.
#
# Resuming is safe by construction. Every pass keeps its own cursor (section 5.2
# rule 4), the checkpoint advances per *committed* thread and stops at the first
# failure (rule 5), and `index_thread` makes re-processing a no-op (section 5.1) --
# so a killed Job re-covers the remainder rather than starting over or skipping.
# `derive` is re-runnable for the same reason and writes nothing for a thread that
# has not changed.
set -euo pipefail

# The verb, from the wrapper that called us (backfill.sh) or from argv[1]. Read
# before the option loop, so `--help` is answered by the usage text below rather
# than by a complaint about a missing verb.
verb="${RELORED_PASS:-}"
if [[ -z "$verb" && "${1:-}" != -* ]]; then
  verb="${1:-}"
  shift || true
fi

namespace="ghlore"
release="ghlore"
context=""
repo=""
follow=0
extra=()

usage() {
  cat <<'EOF'
Usage: deploy/scripts/pass.sh VERB --repo OWNER/NAME [options]

VERB is one of backfill | derive | fetch | authority | sweep | embed -- the passes
long enough to outlive a terminal. A full `derive` is what a change to extraction, the
trust derivation or the relationship edges needs before it reaches the index.

Options:
      --repo OWNER/NAME   required; must be in the chart's allowlist
  -n, --namespace NAME    default: ghlore
  -r, --release NAME      default: ghlore
      --context NAME      kubectl context
  -f, --follow            stream the logs after creating the Job
      --                  everything after this is passed to `relored VERB`
  -h, --help              this

Examples:
  deploy/scripts/pass.sh backfill --repo huggingface/transformers --follow
  deploy/scripts/pass.sh derive   --repo huggingface/transformers --follow
  deploy/scripts/pass.sh backfill --repo huggingface/x -- --no-graphql
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo) repo="$2"; shift 2 ;;
    -n|--namespace) namespace="$2"; shift 2 ;;
    -r|--release) release="$2"; shift 2 ;;
    --context) context="$2"; shift 2 ;;
    -f|--follow) follow=1; shift ;;
    --) shift; extra=("$@"); break ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

case "$verb" in
  backfill|derive|fetch|authority|sweep|embed) ;;
  "") echo "error: which pass? one of backfill|derive|fetch|authority|sweep|embed" >&2; usage >&2; exit 2 ;;
  *) echo "error: $verb is not a pass this script runs" >&2; usage >&2; exit 2 ;;
esac
[[ -n "$repo" ]] || { echo "error: --repo is required" >&2; exit 2; }

kube=(kubectl --namespace "$namespace")
[[ -n "$context" ]] && kube+=(--context "$context")

slug="$(echo "$repo" | tr '/.' '--' | tr '[:upper:]' '[:lower:]')"
job="${release}-${verb}-${slug}"

# Take the image and the wiring from the running deployment rather than from values:
# a pass must be the same build as the daemon that will serve what it writes.
image="$("${kube[@]}" get deploy "$release" -o jsonpath='{.spec.template.spec.containers[0].image}')"
[[ -n "$image" ]] || { echo "error: could not read the image from deploy/$release" >&2; exit 2; }
db_url="$("${kube[@]}" get deploy "$release" \
  -o jsonpath='{.spec.template.spec.containers[0].env[?(@.name=="RELORE_DATABASE_URL")].value}')"
secret="$("${kube[@]}" get deploy "$release" \
  -o jsonpath='{.spec.template.spec.containers[0].envFrom[1].secretRef.name}')"
config="$("${kube[@]}" get deploy "$release" \
  -o jsonpath='{.spec.template.spec.containers[0].envFrom[0].configMapRef.name}')"

echo "pass    $verb"
echo "repo    $repo"
echo "job     $job"
echo "image   $image"
echo

"${kube[@]}" delete job "$job" --ignore-not-found >/dev/null

# `-v` is not optional in practice. `relored` defaults to WARNING and every
# progress message it has is INFO, so without this a day-long backfill stages
# hundreds of thousands of objects and prints nothing at all -- the only way to
# tell a healthy run from a wedged one is to query the database.
args="[\"-v\", \"$verb\", \"--repo\", \"$repo\""
for a in ${extra[@]+"${extra[@]}"}; do args="$args, \"$a\""; done
args="$args]"

"${kube[@]}" apply -f - <<YAML
apiVersion: batch/v1
kind: Job
metadata:
  name: $job
  namespace: $namespace
  labels:
    # ghlore, matching the release's own app label -- see deploy.sh. logs.sh selects
    # on it, so a Job labelled relore would run and be unreadable.
    # (No backticks in this heredoc: it is unquoted, so they command-substitute.)
    app: ghlore
    component: pass
spec:
  # Generous: the pass is resumable, so a retry costs a re-walk of the remainder
  # rather than the whole history. Section 3 calls it "one restartable day".
  backoffLimit: 20
  # Keep the record for a day after it finishes, then clean up on its own.
  ttlSecondsAfterFinished: 86400
  template:
    metadata:
      labels:
        app: ghlore
        component: pass
    spec:
      restartPolicy: OnFailure
      containers:
        - name: pass
          image: "$image"
          args: $args
          envFrom:
            - configMapRef:
                name: $config
            - secretRef:
                name: $secret
          env:
            - name: POSTGRES_PASSWORD
              valueFrom:
                secretKeyRef:
                  name: $secret
                  key: POSTGRES_PASSWORD
            - name: RELORE_DATABASE_URL
              value: "$db_url"
          resources:
            requests: {cpu: 200m, memory: 512Mi}
            limits: {memory: 3Gi}
YAML

echo
echo "watch it with:"
echo "  kubectl ${context:+--context $context} -n $namespace logs -f job/$job"
[[ $follow -eq 1 ]] && exec "${kube[@]}" logs -f "job/$job"
