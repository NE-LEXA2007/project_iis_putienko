#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p reports/runs
for size in 384 768; do
    CHUNK_SIZE="$size" CHUNK_OVERLAP=64 .venv/bin/python -u ingest.py > "reports/runs/retrieval-$size.log" 2>&1
    CHUNK_SIZE="$size" CHUNK_OVERLAP=64 .venv/bin/python -c 'from ingest import evaluate_retrieval; evaluate_retrieval(k=1)' >> "reports/runs/retrieval-$size.log" 2>&1
done
# Восстанавливаем индекс под выбранное значение, даже если лучший вариант другой.
# После анализа измените константу config.CHUNK_SIZE и пересоберите индекс сами.
.venv/bin/python ingest.py
