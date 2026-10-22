#!/usr/bin/env bash
# Run a strict same-episode HM3Dv2 comparison:
#   arm A: policy=1, greedy maximum of the fused frontier value
#   arm B: policy=4, WTRP over the same fused frontier values

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
RUNNER="${ROOT}/scripts/run_hm3dv2_greedy1000.sh"
PAIR_SESSION="${PAIR_SESSION:-infonav_value_wtrp_pair100}"
PAIR_STAMP="${PAIR_STAMP:-$(date +%Y%m%d_%H%M%S)}"
PAIR_ROOT="${1:-${ROOT}/videos/test_hm3dv2_value_wtrp_pair100_${PAIR_STAMP}_val}"
VALUE_PROTOCOL="${PAIR_ROOT}/value_greedy/run_protocol.env"
if [[ -f "${VALUE_PROTOCOL}" ]]; then
    EXPERIMENT_GIT_COMMIT="$(sed -n 's/^GIT_COMMIT=//p' "${VALUE_PROTOCOL}" | head -n 1)"
    EXPERIMENT_GIT_DIRTY_TRACKED="$(sed -n 's/^GIT_DIRTY_TRACKED=//p' "${VALUE_PROTOCOL}" | head -n 1)"
else
    EXPERIMENT_GIT_COMMIT="$(git -C "${ROOT}" rev-parse HEAD)"
    if [[ -n "$(git -C "${ROOT}" status --porcelain --untracked-files=no)" ]]; then
        EXPERIMENT_GIT_DIRTY_TRACKED=true
    else
        EXPERIMENT_GIT_DIRTY_TRACKED=false
    fi
fi

common_env=(
    TOTAL=100
    HABITAT_SEED=100
    SUCCESS_DISTANCE=1.0
    USE_MULTI_SOURCE=true
    FUSION_ALPHA=0.8
    USE_IG_FUSION=true
    USE_ADAPTIVE_ALPHA=true
    ALPHA_BASE=0.8
    TAU_CONF=0.8
    TOP_K_VALUE=8
    WTRP_TEMPERATURE=0.3
    WTRP_HYSTERESIS_RATIO=0.8
    WTRP_MAX_BRUTE_FORCE=8
    ENABLE_FEISHU_REPORT=false
    VLM_VALIDATION_ENABLED=false
    EXPERIMENT_GIT_COMMIT="${EXPERIMENT_GIT_COMMIT}"
    EXPERIMENT_GIT_DIRTY_TRACKED="${EXPERIMENT_GIT_DIRTY_TRACKED}"
    FORCE_REGENERATE_HYPOTHESES=0
)

state_value() {
    local file="$1" key="$2"
    if [[ ! -f "${file}" ]]; then
        return 0
    fi
    sed -n "s/^${key}=//p" "${file}" | head -n 1
}

metrics_count() {
    local metrics_file="$1"
    if [[ ! -f "${metrics_file}" ]]; then
        printf '0\n'
        return
    fi
    wc -l <"${metrics_file}" | tr -d '[:space:]'
}

wait_for_arm() {
    local state_file="$1" session="$2" metrics_file="$3" status completed
    while true; do
        completed="$(metrics_count "${metrics_file}")"
        if [[ "${completed}" =~ ^[0-9]+$ ]] && (( completed >= 100 )); then
            printf '[pair] %s completed according to episode metrics (%s/100)\n' \
                "${session}" "${completed}"
            return 0
        fi
        status="$(state_value "${state_file}" STATUS)"
        case "${status}" in
            completed)
                printf '[pair] %s completed\n' "${session}"
                return 0
                ;;
            stopped)
                printf '[pair:error] %s stopped: %s\n' \
                    "${session}" "$(state_value "${state_file}" REASON)" >&2
                return 1
                ;;
        esac
        if [[ -n "${status}" ]] && ! tmux has-session -t "${session}" 2>/dev/null; then
            printf '[pair:error] %s disappeared with status=%s\n' "${session}" "${status}" >&2
            return 1
        fi
        sleep 15
    done
}

run_arm() {
    local name="$1" policy="$2"
    local run_tag="hm3dv2_${name}100_${PAIR_STAMP}"
    local arm_session="infonav_${name}100_${PAIR_STAMP}"
    local protocol_id="hm3dv2_${name}_hsvm_ig_gdino_success1m_pair100_v1"
    local output_dir="${PAIR_ROOT}/${name}"
    local state_file="${ROOT}/.run_state/${run_tag}.env"
    local metrics_file="${output_dir}/episode_metrics.jsonl"

    local status completed
    status="$(state_value "${state_file}" STATUS)"
    completed="$(metrics_count "${metrics_file}")"
    if [[ "${status}" == "completed" ]] ||
       { [[ "${completed}" =~ ^[0-9]+$ ]] && (( completed >= 100 )); }; then
        printf '[pair] %s was already completed; continuing\n' "${name}"
        return 0
    fi
    if tmux has-session -t "${arm_session}" 2>/dev/null; then
        printf '[pair] reusing active %s arm session %s\n' "${name}" "${arm_session}"
        wait_for_arm "${state_file}" "${arm_session}" "${metrics_file}"
        return
    fi
    if [[ "${status}" == "running" || "${status}" == "starting" ]]; then
        printf '[pair:error] %s state is %s but its tmux session is absent\n' \
            "${name}" "${status}" >&2
        return 1
    fi
    printf '[pair] starting %s (policy=%s) -> %s\n' "${name}" "${policy}" "${output_dir}"
    env "${common_env[@]}" \
        RUN_TAG="${run_tag}" SESSION="${arm_session}" \
        PROTOCOL_ID="${protocol_id}" EXPLORATION_POLICY="${policy}" \
        "${RUNNER}" "${output_dir}"
    wait_for_arm "${state_file}" "${arm_session}" "${metrics_file}"
}

if [[ "${PAIR_INNER:-0}" != "1" ]]; then
    if tmux has-session -t "${PAIR_SESSION}" 2>/dev/null; then
        printf '[pair:error] manager session already exists: %s\n' "${PAIR_SESSION}" >&2
        exit 2
    fi
    mkdir -p "${PAIR_ROOT}"
    tmux new-session -d -s "${PAIR_SESSION}" \
        "env PAIR_INNER=1 PAIR_SESSION='${PAIR_SESSION}' PAIR_STAMP='${PAIR_STAMP}' '${ROOT}/scripts/run_hm3dv2_value_wtrp_pair100.sh' '${PAIR_ROOT}' >'${PAIR_ROOT}/pair_controller.log' 2>&1"
    printf '[pair:ready] tmux=%s root=%s\n' "${PAIR_SESSION}" "${PAIR_ROOT}"
    printf '[pair:note] arms run sequentially and reuse the existing persistent VLM services\n'
    exit 0
fi

run_arm value_greedy 1
run_arm wtrp 4
"${ROOT}/scripts/compare_hm3dv2_value_wtrp.py" \
    "${PAIR_ROOT}/value_greedy/episode_metrics.jsonl" \
    "${PAIR_ROOT}/wtrp/episode_metrics.jsonl" \
    >"${PAIR_ROOT}/paired_comparison.txt"
printf '[pair:done] comparison=%s\n' "${PAIR_ROOT}/paired_comparison.txt"
