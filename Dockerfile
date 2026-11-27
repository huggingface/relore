# relore — the daemon image. `relored` and `relore`, nothing else.
#
# It installs `relore[postgres,python,embed]`: the daemon's half of the code lens landed, so
# without a parser `symbol`, `copies` and `defs`/`refs --repo` answer `MissingParser`.
# `git` is what `relored clone` and `why`'s blame shell out to.
#
# `pg_isready` comes from postgresql-client and is used by the migrate hook to
# wait for the database rather than fail the release on a first-install race.
FROM python:3.10-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update \
 && apt-get install -y --no-install-recommends git postgresql-client \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md ./
COPY relore ./relore
RUN pip install --no-cache-dir '.[postgres,python,embed]'

# Section 6.1's model, baked in: pods load it from here and never reach the Hub. Inert
# unless the chart sets `embed.model` (RELORE_EMBED_MODEL) to this same name.
ENV FASTEMBED_CACHE_PATH=/opt/fastembed
RUN python -c "from fastembed import TextEmbedding; TextEmbedding('BAAI/bge-small-en-v1.5')" \
 && chmod -R a+rX /opt/fastembed
# After the download: nothing at runtime may reach the Hub, so a missing model fails at
# startup instead of fetching from wherever the pod's egress allows.
ENV HF_HUB_OFFLINE=1

# Not root. The daemon reads a token and writes one JSONL file; it has no reason
# to be able to do anything else.
RUN useradd --create-home --uid 10001 relore \
 && mkdir -p /var/lib/ghlore \
 && chown -R relore:relore /var/lib/ghlore
# `/var/lib/ghlore` and not `/var/lib/relore`: the chart mounts the labels and clones PVCs
# there, and those volumes predate the 2026-09-14 rename. The path is the volume's, not
# the project's.
USER relore

# `relored` is the entry point, so a workload's args read as the verb they are:
# ["serve", "--host", "0.0.0.0"] rather than a full command line.
ENTRYPOINT ["relored"]
CMD ["--help"]
