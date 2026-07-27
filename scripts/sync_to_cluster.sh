#!/bin/bash
# Push code and the processed panel to a SLURM cluster. Pull results back.
#
# Needs a multiplexed SSH master to be open. If your cluster enforces MFA, that first
# connection needs a real terminal, so open it yourself once (`ssh <host>`); with
# ControlMaster/ControlPersist configured in ~/.ssh/config it then persists and
# everything here reuses it without another prompt.
#
#   ./scripts/sync_to_cluster.sh push      code + data/processed  (~541MB the first time)
#   ./scripts/sync_to_cluster.sh pull      outputs/ back here
#   ./scripts/sync_to_cluster.sh status    queue + how many work units have landed
#
# The raw swap parquet in ~/Projects/defi-rv is NOT synced: it is 4.9GB and the
# cluster only needs the aggregated panel that data/aggregate.py and data/swaps.py
# already produced from it.

set -euo pipefail

REMOTE=${REMOTE_HOST:?set REMOTE_HOST to an ssh alias in your ~/.ssh/config}
PROJECT=${REMOTE_PROJECT:?set REMOTE_PROJECT to the project path on the cluster}
LOCAL="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# macOS ships openrsync (protocol 29), which has no --info=progress2 and no
# --exclude-from niceties. Prefer a real rsync if Homebrew put one on the PATH,
# otherwise fall back to flags openrsync actually understands.
RSYNC_FLAGS="-az"
if rsync --version 2>/dev/null | head -1 | grep -qv openrsync; then
  RSYNC_FLAGS="-az --info=progress2"
fi

if ! ssh -o BatchMode=yes -o ConnectTimeout=10 "$REMOTE" true 2>/dev/null; then
  echo "cannot reach $REMOTE. The SSH master is closed or expired (>8h)." >&2
  echo "Open the master yourself in a terminal:  ssh $REMOTE" >&2
  exit 1
fi

case "${1:-}" in
  push)
    ssh "$REMOTE" "mkdir -p $PROJECT/slurm_logs $PROJECT/outputs $PROJECT/data/processed"
    echo "--- code ---"
    rsync $RSYNC_FLAGS \
      --exclude '.git' --exclude '__pycache__' --exclude '*.pyc' \
      --exclude '/data' --exclude '/outputs' --exclude '/rl-code/output' \
      --exclude 'paper' --exclude '.DS_Store' \
      "$LOCAL/" "$REMOTE:$PROJECT/"
    echo "--- panel (only what changed) ---"
    rsync $RSYNC_FLAGS "$LOCAL/data/processed/" "$REMOTE:$PROJECT/data/processed/"
    echo "--- env check ---"
    ssh "$REMOTE" "test -x ${REMOTE_CONDA:?}/envs/deeprl-uniswap/bin/python3 \
      && echo 'env deeprl-uniswap: present' \
      || echo 'env deeprl-uniswap: MISSING -> conda env create -f $PROJECT/environment.yml'"
    ;;
  pull)
    mkdir -p "$LOCAL/outputs"
    rsync $RSYNC_FLAGS "$REMOTE:$PROJECT/outputs/" "$LOCAL/outputs/"
    ;;
  status)
    ssh "$REMOTE" "squeue -u \$USER -o '%.10i %.9P %.20j %.2t %.10M %.6D %R' | head -20; \
      echo '--- work units landed ---'; \
      for d in $PROJECT/outputs/*/; do \
        [ -d \"\$d\" ] && echo \"\$(basename \$d): \$(ls \$d/*.json 2>/dev/null | wc -l) units\"; \
      done"
    ;;
  *)
    echo "usage: $0 {push|pull|status}" >&2
    exit 2
    ;;
esac
