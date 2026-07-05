#!/usr/bin/env bash

set -Eeuo pipefail

DATA_ROOT="${DATA_ROOT:-datasets}"
LIMIT="${LIMIT:-100}"
PYTHON="${PYTHON:-python}"
RUNNER="${RUNNER:-scripts/run_graphsum.py}"
OUTPUT_DIR="${OUTPUT_DIR:-runs}"
LOG_DIR="${LOG_DIR:-${OUTPUT_DIR}/logs}"

if [[ ! -f "${RUNNER}" ]]; then
    echo "Error: runner script not found: ${RUNNER}" >&2
    exit 1
fi

mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"

COMMON_ARGS=(
    --data-root "${DATA_ROOT}"
    --limit "${LIMIT}"
    --llm openai_compatible
)

run_experiment() {
    local name="$1"
    shift

    local log_file="${LOG_DIR}/${name}.log"

    echo
    echo "============================================================"
    echo "Running: ${name}"
    echo "Log:     ${log_file}"
    echo "============================================================"

    "${PYTHON}" "${RUNNER}" \
        "$@" \
        "${COMMON_ARGS[@]}" \
        2>&1 | tee "${log_file}"

    echo "Completed: ${name}"
}

echo "Data root:    ${DATA_ROOT}"
echo "Sample limit: ${LIMIT}"
echo "Multi-News runs are disabled."

# ---------------------------------------------------------------------------
# 1. Main E2b runs
# ---------------------------------------------------------------------------

run_experiment "vn_mds_e2b_real" \
    --dataset vn_mds \
    --salience e2b \
    --output "${OUTPUT_DIR}/vn_mds_e2b_real.csv"

run_experiment "vims_e2b_real" \
    --dataset vims \
    --salience e2b \
    --output "${OUTPUT_DIR}/vims_e2b_real.csv"

# ---------------------------------------------------------------------------
# 2. Pure-LLM baselines
# ---------------------------------------------------------------------------

run_experiment "vn_mds_pure_llm_real" \
    --dataset vn_mds \
    --pure-llm \
    --output "${OUTPUT_DIR}/vn_mds_pure_llm_real.csv"

run_experiment "vims_pure_llm_real" \
    --dataset vims \
    --pure-llm \
    --output "${OUTPUT_DIR}/vims_pure_llm_real.csv"

# ---------------------------------------------------------------------------
# 3. Graph-weight grid on VN-MDS (testing)
# ---------------------------------------------------------------------------

run_experiment "vn_mds_e2b_grid_real" \
    --dataset vn_mds \
    --salience e2b \
    --grid \
    --output "${OUTPUT_DIR}/vn_mds_e2b_grid_real.csv"

echo
echo "============================================================"
echo "All configured experiments completed successfully."
echo "Outputs: ${OUTPUT_DIR}"
echo "Logs:    ${LOG_DIR}"
echo "============================================================"