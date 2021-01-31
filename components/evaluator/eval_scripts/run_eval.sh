#!/usr/bin/env bash
set -euxo pipefail

# Modify this path variable, if needed!
E2E_METRICS_FOLDER=$HOME/projects/E2EChallenge/materials/e2e-metrics

REF_FNAME=$1
PRED_FNAME=$2

echo 'Running evaluation script (dev)'
python $E2E_METRICS_FOLDER/measure_scores.py $REF_FNAME $PRED_FNAME
