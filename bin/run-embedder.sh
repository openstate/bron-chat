#!/bin/bash
# Run the embedding pipeline once. Intended for cron or manual invocation.
# Add to crontab: 0 3 * * * /home/projects/bron-chat/bin/run-embedder.sh >> /var/log/bron-embedding.log 2>&1
# Extra arguments are passed to run.py, e.g.:
#   bin/run-embedder.sh --estimate --sample-size 500
#   bin/run-embedder.sh --since 2026-01-01T00:00:00

set -euo pipefail

cd "$(dirname "$0")/.."

docker compose -f docker-compose.prod.yml run --rm embedder python run.py "$@"
