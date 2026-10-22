#!/usr/bin/env bash
# Resumable and GPU-safe HM3Dv2 evaluation runner.

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PYTHON="${INFONAV_PYTHON:-$(command -v python || command -v python3)}"
CC_CONNECT_DATA_DIR="${CC_CONNECT_DATA_DIR:-}"
STATE_DIR="${ROOT}/.run_state"
RUN_TAG="${RUN_TAG:-hm3dv2_greedy1000}"
STATE_FILE="${STATE_DIR}/${RUN_TAG}.env"
CRON_ID_FILE="${STATE_DIR}/${RUN_TAG}_cron_id"
SESSION="${SESSION:-infonav_greedy1000}"
TOTAL="${TOTAL:-1000}"
PROTOCOL_ID="${PROTOCOL_ID:-hm3dv2_distance_greedy_hsvm_ig_gdino_success1m_v1}"
SUCCESS_DISTANCE="${SUCCESS_DISTANCE:-1.0}"
EXPLORATION_POLICY="${EXPLORATION_POLICY:-0}"
USE_MULTI_SOURCE="${USE_MULTI_SOURCE:-true}"
FUSION_ALPHA="${FUSION_ALPHA:-0.8}"
USE_IG_FUSION="${USE_IG_FUSION:-true}"
USE_ADAPTIVE_ALPHA="${USE_ADAPTIVE_ALPHA:-false}"
ALPHA_BASE="${ALPHA_BASE:-0.8}"
TAU_CONF="${TAU_CONF:-0.8}"
TOP_K_VALUE="${TOP_K_VALUE:-8}"
WTRP_TEMPERATURE="${WTRP_TEMPERATURE:-0.3}"
WTRP_HYSTERESIS_RATIO="${WTRP_HYSTERESIS_RATIO:-0.8}"
WTRP_MAX_BRUTE_FORCE="${WTRP_MAX_BRUTE_FORCE:-8}"
ENABLE_FEISHU_REPORT="${ENABLE_FEISHU_REPORT:-false}"
HABITAT_SEED="${HABITAT_SEED:-100}"
VLM_VALIDATION_ENABLED="${VLM_VALIDATION_ENABLED:-false}"
EXPERIMENT_GIT_COMMIT="${EXPERIMENT_GIT_COMMIT:-$(git -C "${ROOT}" rev-parse HEAD)}"
if [[ -z "${EXPERIMENT_GIT_DIRTY_TRACKED+x}" ]]; then
    if [[ -n "$(git -C "${ROOT}" status --porcelain --untracked-files=no)" ]]; then
        EXPERIMENT_GIT_DIRTY_TRACKED=true
    else
        EXPERIMENT_GIT_DIRTY_TRACKED=false
    fi
fi
BATCH_SIZE="${BATCH_SIZE:-25}"
STALL_TIMEOUT="${STALL_TIMEOUT:-2700}"
GPU_MEMORY_LIMIT="${GPU_MEMORY_LIMIT:-11000}"
GPU_TEMPERATURE_LIMIT="${GPU_TEMPERATURE_LIMIT:-82}"
GPU_POWER_LIMIT_MAX="${GPU_POWER_LIMIT_MAX:-220}"
GPU_IDLE_MEMORY_LIMIT="${GPU_IDLE_MEMORY_LIMIT:-1500}"
GPU_IDLE_STABLE_SECONDS="${GPU_IDLE_STABLE_SECONDS:-60}"
GPU_IDLE_TIMEOUT="${GPU_IDLE_TIMEOUT:-300}"
CONTROL_RATE_HZ="${CONTROL_RATE_HZ:-4}"
MODEL_LOAD_COOLDOWN="${MODEL_LOAD_COOLDOWN:-30}"
VLM_REQUEST_TIMEOUT="${VLM_REQUEST_TIMEOUT:-120}"
VLM_REQUEST_ATTEMPTS="${VLM_REQUEST_ATTEMPTS:-3}"
QWEN_VL_MODEL="${QWEN_VL_MODEL:-qwen3-vl-plus}"
DASHSCOPE_REGION="${DASHSCOPE_REGION:-beijing}"
DEEPSEEK_MODEL="${DEEPSEEK_MODEL:-deepseek-v4-flash}"
QWEN_HYPOTHESIS_MODEL="${QWEN_HYPOTHESIS_MODEL:-qwen3-vl-flash}"
DEEPSEEK_HYPOTHESIS_MODEL="${DEEPSEEK_HYPOTHESIS_MODEL:-deepseek-chat}"
FORCE_REGENERATE_HYPOTHESES="${FORCE_REGENERATE_HYPOTHESES:-0}"

# ~/.bashrc keeps API exports below its non-interactive early return. Import
# only the named runtime variables through an interactive shell, without
# printing values or embedding them in process command lines.
load_interactive_export() {
    local name="$1" value
    if [[ -n "${!name:-}" ]]; then
        return 0
    fi
    value="$(bash -ic "printf '%s' \"\${${name}:-}\"" 2>/dev/null)"
    if [[ -n "${value}" ]]; then
        printf -v "${name}" '%s' "${value}"
        export "${name}"
    fi
}
for runtime_name in DASHSCOPE_API_KEY DASHSCOPE_BASE_URL DASHSCOPE_WORKSPACE_ID \
                    DEEPSEEK_API_KEY DEEPSEEK_BASE_URL; do
    load_interactive_export "${runtime_name}"
done

PREFLIGHT_ONLY=0
START_VLM_ONLY=0
if [[ "${1:-}" == "--preflight-only" ]]; then
    PREFLIGHT_ONLY=1
    shift
elif [[ "${1:-}" == "--start-vlm-only" ]]; then
    START_VLM_ONLY=1
    shift
fi
if (( START_VLM_ONLY == 1 )); then
    OUTPUT_DIR="${1:-${STATE_DIR}/vlm_start}"
else
    OUTPUT_DIR="${1:-${ROOT}/videos/${RUN_TAG}_$(date +%Y%m%d_%H%M%S)_val}"
fi
LOG_DIR="${OUTPUT_DIR}/run_logs"
RUN_PROTOCOL_FILE="${OUTPUT_DIR}/run_protocol.env"
VLM_SESSION="${VLM_SESSION:-infonav_vlm_persistent}"
VLM_LOG_DIR="${STATE_DIR}/vlm_logs"
GPU_TRACE_LOG="${LOG_DIR}/gpu.csv"
KERNEL_GPU_LOG="${LOG_DIR}/kernel_gpu.log"
HYPOTHESIS_DIR="${ROOT}/env/env_hm3dv2"

# One BLIP2 instance scores every HSVM and IG channel in a combined request.
# GroundingDINO remains mandatory for non-COCO contextual hypotheses.
export QWEN_VL_MODEL DASHSCOPE_REGION DEEPSEEK_MODEL
export QWEN_HYPOTHESIS_MODEL DEEPSEEK_HYPOTHESIS_MODEL
export USE_PARALLEL_BLIP2=false
export VLM_REQUEST_TIMEOUT VLM_REQUEST_ATTEMPTS

declare -A PIDS=()

protocol_value() {
    local key="$1"
    sed -n "s/^${key}=//p" "${RUN_PROTOCOL_FILE}" | head -n 1
}

protocol_matches() {
    [[ "$(protocol_value PROTOCOL_ID)" == "${PROTOCOL_ID}" ]] &&
        [[ "$(protocol_value TOTAL_EPISODES)" == "${TOTAL}" ]] &&
        [[ "$(protocol_value SUCCESS_DISTANCE_M)" == "${SUCCESS_DISTANCE}" ]] &&
        [[ "$(protocol_value EXPLORATION_POLICY)" == "${EXPLORATION_POLICY}" ]] &&
        [[ "$(protocol_value USE_MULTI_SOURCE)" == "${USE_MULTI_SOURCE}" ]] &&
        [[ "$(protocol_value FUSION_ALPHA)" == "${FUSION_ALPHA}" ]] &&
        [[ "$(protocol_value USE_IG_FUSION)" == "${USE_IG_FUSION}" ]] &&
        [[ "$(protocol_value USE_ADAPTIVE_ALPHA)" == "${USE_ADAPTIVE_ALPHA}" ]] &&
        [[ "$(protocol_value ALPHA_BASE)" == "${ALPHA_BASE}" ]] &&
        [[ "$(protocol_value TAU_CONF)" == "${TAU_CONF}" ]] &&
        [[ "$(protocol_value TOP_K_VALUE)" == "${TOP_K_VALUE}" ]] &&
        [[ "$(protocol_value WTRP_TEMPERATURE)" == "${WTRP_TEMPERATURE}" ]] &&
        [[ "$(protocol_value WTRP_HYSTERESIS_RATIO)" == "${WTRP_HYSTERESIS_RATIO}" ]] &&
        [[ "$(protocol_value WTRP_MAX_BRUTE_FORCE)" == "${WTRP_MAX_BRUTE_FORCE}" ]] &&
        [[ "$(protocol_value HABITAT_SEED)" == "${HABITAT_SEED}" ]] &&
        [[ "$(protocol_value VLM_VALIDATION_ENABLED)" == "${VLM_VALIDATION_ENABLED}" ]]
}

write_run_protocol() {
    local tmp_file="${RUN_PROTOCOL_FILE}.tmp.$$"
    {
        printf 'PROTOCOL_ID=%s\n' "${PROTOCOL_ID}"
        printf 'DATASET=hm3dv2\n'
        printf 'SPLIT=val\n'
        printf 'TOTAL_EPISODES=%s\n' "${TOTAL}"
        printf 'HABITAT_SEED=%s\n' "${HABITAT_SEED}"
        printf 'SUCCESS_DISTANCE_M=%s\n' "${SUCCESS_DISTANCE}"
        printf 'EXPLORATION_POLICY=%s\n' "${EXPLORATION_POLICY}"
        printf 'USE_MULTI_SOURCE=%s\n' "${USE_MULTI_SOURCE}"
        printf 'FUSION_ALPHA=%s\n' "${FUSION_ALPHA}"
        printf 'USE_IG_FUSION=%s\n' "${USE_IG_FUSION}"
        printf 'USE_ADAPTIVE_ALPHA=%s\n' "${USE_ADAPTIVE_ALPHA}"
        printf 'ALPHA_BASE=%s\n' "${ALPHA_BASE}"
        printf 'TAU_CONF=%s\n' "${TAU_CONF}"
        printf 'TOP_K_VALUE=%s\n' "${TOP_K_VALUE}"
        printf 'WTRP_TEMPERATURE=%s\n' "${WTRP_TEMPERATURE}"
        printf 'WTRP_HYSTERESIS_RATIO=%s\n' "${WTRP_HYSTERESIS_RATIO}"
        printf 'WTRP_MAX_BRUTE_FORCE=%s\n' "${WTRP_MAX_BRUTE_FORCE}"
        printf 'VLM_VALIDATION_ENABLED=%s\n' "${VLM_VALIDATION_ENABLED}"
        printf 'VALUE_DEFINITION=normalized_multi_source_HSVM_plus_IG\n'
        printf 'GIT_COMMIT=%s\n' "${EXPERIMENT_GIT_COMMIT}"
        printf 'GIT_DIRTY_TRACKED=%s\n' "${EXPERIMENT_GIT_DIRTY_TRACKED}"
        printf 'DETECTOR_MODE=2\n'
        printf 'LOCAL_MODELS=blip2_itm,grounding_dino,mobile_sam,dfine\n'
        printf 'BLIP2_REPLICAS=1\n'
        printf 'CREATED_AT=%s\n' "$(date --iso-8601=seconds)"
    } >"${tmp_file}"
    mv "${tmp_file}" "${RUN_PROTOCOL_FILE}"
}

write_state() {
    local status="$1"
    local reason="${2:-none}"
    mkdir -p "${STATE_DIR}"
    local tmp_file="${STATE_FILE}.tmp.$$"
    {
        printf 'STATUS=%s\n' "${status}"
        printf 'REASON=%s\n' "${reason// /_}"
        printf 'OUTPUT_DIR=%s\n' "${OUTPUT_DIR}"
        printf 'TOTAL=%s\n' "${TOTAL}"
        printf 'HABITAT_SEED=%s\n' "${HABITAT_SEED}"
        printf 'PROTOCOL_ID=%s\n' "${PROTOCOL_ID}"
        printf 'SUCCESS_DISTANCE_M=%s\n' "${SUCCESS_DISTANCE}"
        printf 'POLICY=%s\n' "${EXPLORATION_POLICY}"
        printf 'USE_MULTI_SOURCE=%s\n' "${USE_MULTI_SOURCE}"
        printf 'FUSION_ALPHA=%s\n' "${FUSION_ALPHA}"
        printf 'USE_IG_FUSION=%s\n' "${USE_IG_FUSION}"
        printf 'USE_ADAPTIVE_ALPHA=%s\n' "${USE_ADAPTIVE_ALPHA}"
        printf 'ALPHA_BASE=%s\n' "${ALPHA_BASE}"
        printf 'TAU_CONF=%s\n' "${TAU_CONF}"
        printf 'TOP_K_VALUE=%s\n' "${TOP_K_VALUE}"
        printf 'WTRP_TEMPERATURE=%s\n' "${WTRP_TEMPERATURE}"
        printf 'WTRP_HYSTERESIS_RATIO=%s\n' "${WTRP_HYSTERESIS_RATIO}"
        printf 'WTRP_MAX_BRUTE_FORCE=%s\n' "${WTRP_MAX_BRUTE_FORCE}"
        printf 'VLM_VALIDATION_ENABLED=%s\n' "${VLM_VALIDATION_ENABLED}"
        printf 'BATCH_SIZE=%s\n' "${BATCH_SIZE}"
        printf 'CONTROL_RATE_HZ=%s\n' "${CONTROL_RATE_HZ}"
        printf 'GPU_POWER_LIMIT_MAX=%s\n' "${GPU_POWER_LIMIT_MAX}"
        printf 'VLM_SESSION=%s\n' "${VLM_SESSION}"
        printf 'VLM_REQUEST_TIMEOUT=%s\n' "${VLM_REQUEST_TIMEOUT}"
        printf 'LOCAL_MODELS=blip2_itm,grounding_dino,mobile_sam,dfine\n'
        printf 'BLIP2_REPLICAS=1\n'
        printf 'BLIP2_ENABLED=true\n'
        printf 'QWEN_VL_MODEL=%s\n' "${QWEN_VL_MODEL}"
        printf 'DASHSCOPE_REGION=%s\n' "${DASHSCOPE_REGION}"
        printf 'DEEPSEEK_MODEL=%s\n' "${DEEPSEEK_MODEL}"
        printf 'QWEN_HYPOTHESIS_MODEL=%s\n' "${QWEN_HYPOTHESIS_MODEL}"
        printf 'DEEPSEEK_HYPOTHESIS_MODEL=%s\n' "${DEEPSEEK_HYPOTHESIS_MODEL}"
        printf 'FORCE_REGENERATE_HYPOTHESES=%s\n' "${FORCE_REGENERATE_HYPOTHESES}"
        printf 'UPDATED_AT=%s\n' "$(date --iso-8601=seconds)"
    } >"${tmp_file}"
    mv "${tmp_file}" "${STATE_FILE}"
}

gpu_query() {
    timeout 5 nvidia-smi --id=0 \
        --query-gpu=memory.used,temperature.gpu,power.draw,power.limit,utilization.gpu,pstate,pcie.link.gen.current,pcie.link.width.current \
        --format=csv,noheader,nounits 2>/dev/null
}

kernel_gpu_fault() {
    [[ -s "${KERNEL_GPU_LOG}" ]] &&
        grep -E -q -i 'NVRM: Xid|fallen off the bus|PCIe Bus Error|Physical Layer.*RxErr' \
            "${KERNEL_GPU_LOG}"
}

gpu_health_reason() {
    local sample used temperature power_draw power_limit utilization _rest
    sample="$(gpu_query)" || {
        printf 'gpu_nvidia_smi_unavailable\n'
        return
    }
    IFS=',' read -r used temperature power_draw power_limit utilization _rest <<<"${sample}"
    used="${used//[[:space:]]/}"
    temperature="${temperature//[[:space:]]/}"
    power_limit="${power_limit//[[:space:]]/}"
    if kernel_gpu_fault; then
        printf 'gpu_pcie_or_xid_kernel_fault\n'
    elif ! [[ "${used}" =~ ^[0-9]+$ && "${temperature}" =~ ^[0-9]+$ &&
              "${power_limit}" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
        printf 'gpu_metrics_invalid\n'
    elif awk -v actual="${power_limit}" -v maximum="${GPU_POWER_LIMIT_MAX}" \
             'BEGIN { exit !(actual > maximum) }'; then
        printf 'gpu_power_limit_%sW_above_%sW\n' \
            "${power_limit}" "${GPU_POWER_LIMIT_MAX}"
    elif (( used >= GPU_MEMORY_LIMIT )); then
        printf 'gpu_memory_limit_%sMiB\n' "${used}"
    elif (( temperature >= GPU_TEMPERATURE_LIMIT )); then
        printf 'gpu_temperature_limit_%sC\n' "${temperature}"
    else
        printf 'healthy\n'
    fi
}

gpu_healthy() {
    [[ "$(gpu_health_reason)" == "healthy" ]]
}

gpu_compute_idle() {
    local apps
    apps="$(timeout 5 nvidia-smi --id=0 --query-compute-apps=pid \
        --format=csv,noheader,nounits 2>/dev/null)" || return 1
    [[ -z "${apps//[[:space:]]/}" ]]
}

wait_for_gpu_idle() {
    local waited=0 stable=0 sample used
    printf '[cooldown] Waiting for an idle GPU before the persistent VLM cold start\n' |
        tee -a "${LOG_DIR}/controller.log"
    while (( waited < GPU_IDLE_TIMEOUT )); do
        if ! gpu_healthy; then
            printf '[error] GPU is unhealthy during idle wait: %s\n' \
                "$(gpu_health_reason)" >&2
            return 1
        fi
        sample="$(gpu_query)"
        IFS=',' read -r used _ <<<"${sample}"
        used="${used//[[:space:]]/}"
        if gpu_compute_idle && [[ "${used}" =~ ^[0-9]+$ ]] &&
           (( used <= GPU_IDLE_MEMORY_LIMIT )); then
            stable=$((stable + 5))
            if (( stable >= GPU_IDLE_STABLE_SECONDS )); then
                printf '[cooldown] GPU idle for %ss; starting persistent VLM services\n' \
                    "${stable}" | tee -a "${LOG_DIR}/controller.log"
                return 0
            fi
        else
            stable=0
        fi
        sleep 5
        waited=$((waited + 5))
    done
    printf '[error] GPU did not remain idle for %ss within %ss\n' \
        "${GPU_IDLE_STABLE_SECONDS}" "${GPU_IDLE_TIMEOUT}" >&2
    return 1
}

port_open() {
    timeout 1 bash -c "exec 3<>/dev/tcp/127.0.0.1/$1" >/dev/null 2>&1
}

wait_for_port() {
    local port="$1"
    local name="$2"
    local max_wait="${3:-300}"
    local waited=0
    while (( waited < max_wait )); do
        if port_open "${port}"; then
            printf '[ready] %s on port %s\n' "${name}" "${port}"
            return 0
        fi
        if ! gpu_healthy; then
            printf '[error] GPU became unhealthy while loading %s\n' "${name}" >&2
            return 1
        fi
        sleep 2
        waited=$((waited + 2))
    done
    printf '[error] %s did not open port %s in %ss\n' "${name}" "${port}" "${max_wait}" >&2
    return 1
}

vlm_stack_ready() {
    tmux has-session -t "${VLM_SESSION}" 2>/dev/null &&
        port_open 12181 && port_open 12182 && port_open 12183 && port_open 12185 &&
        pgrep -f ' -m vlm\.detector\.grounding_dino( |$)' >/dev/null &&
        pgrep -f ' -m vlm\.itm\.blip2itm( |$)' >/dev/null &&
        pgrep -f ' -m vlm\.segmentor\.sam( |$)' >/dev/null &&
        pgrep -f ' -m vlm\.detector\.dfine( |$)' >/dev/null
}

vlm_stack_present() {
    tmux has-session -t "${VLM_SESSION}" 2>/dev/null ||
        port_open 12181 || port_open 12182 || port_open 12183 || port_open 12185 ||
        pgrep -f ' -m vlm\.(itm\.blip2itm|segmentor\.sam|detector\.(dfine|grounding_dino))( |$)' \
            >/dev/null
}

wait_for_persistent_service() {
    local name="$1" port="$2" process_pattern="$3" max_wait="$4" waited=0
    while (( waited < max_wait )); do
        if port_open "${port}" && pgrep -f "${process_pattern}" >/dev/null; then
            printf '[ready] persistent %s on port %s after %ss\n' \
                "${name}" "${port}" "${waited}" | tee -a "${LOG_DIR}/controller.log"
            return 0
        fi
        if ! tmux has-session -t "${VLM_SESSION}" 2>/dev/null ||
           ! gpu_healthy; then
            printf '[error] persistent %s failed during cold start: %s\n' \
                "${name}" "$(gpu_health_reason)" >&2
            return 1
        fi
        sleep 2
        waited=$((waited + 2))
    done
    printf '[error] persistent %s did not open port %s in %ss\n' \
        "${name}" "${port}" "${max_wait}" >&2
    return 1
}

start_vlm_window() {
    local mode="$1" name="$2" module="$3" port="$4" command
    printf -v command \
        "cd %q; exec env -u ALL_PROXY -u all_proxy -u HTTPS_PROXY -u https_proxy -u HTTP_PROXY -u http_proxy VLM_SERVER_THREADS=1 %q -m %q --port %q >>%q 2>&1" \
        "${ROOT}" "${PYTHON}" "${module}" "${port}" "${VLM_LOG_DIR}/${name}.log"
    if [[ "${mode}" == "session" ]]; then
        tmux new-session -d -s "${VLM_SESSION}" -n "${name}" "${command}"
    else
        tmux new-window -d -t "${VLM_SESSION}" -n "${name}" "${command}"
    fi
}

ensure_vlm_stack() {
    if vlm_stack_ready; then
        printf '[reuse] Persistent single BLIP2, GroundingDINO, MobileSAM, and D-FINE are healthy\n' |
            tee -a "${LOG_DIR}/controller.log"
        return 0
    fi
    if vlm_stack_present; then
        printf '[error] Partial or foreign VLM stack detected; refusing an automatic restart\n' >&2
        printf '[hint] Inspect with: tmux attach -t %s\n' "${VLM_SESSION}" >&2
        printf '[hint] Stop explicitly with: tmux kill-session -t %s\n' \
            "${VLM_SESSION}" >&2
        return 1
    fi

    wait_for_gpu_idle || return 1
    mkdir -p "${VLM_LOG_DIR}"
    for name in blip2 sam dfine gdino; do
        printf '\n=== persistent VLM cold start %s at %s ===\n' \
            "${name}" "$(date --iso-8601=seconds)" >>"${VLM_LOG_DIR}/${name}.log"
    done

    start_vlm_window session blip2 vlm.itm.blip2itm 12182
    if ! wait_for_persistent_service blip2 12182 \
        ' -m vlm\.itm\.blip2itm( |$)' 600; then
        tmux kill-session -t "${VLM_SESSION}" 2>/dev/null || true
        return 1
    fi
    printf '[cooldown] BLIP2 loaded; waiting %ss before MobileSAM\n' \
        "${MODEL_LOAD_COOLDOWN}" | tee -a "${LOG_DIR}/controller.log"
    sleep "${MODEL_LOAD_COOLDOWN}"

    start_vlm_window window sam vlm.segmentor.sam 12183
    if ! wait_for_persistent_service sam 12183 \
        ' -m vlm\.segmentor\.sam( |$)' 240; then
        tmux kill-session -t "${VLM_SESSION}" 2>/dev/null || true
        return 1
    fi
    printf '[cooldown] MobileSAM loaded; waiting %ss before D-FINE\n' \
        "${MODEL_LOAD_COOLDOWN}" | tee -a "${LOG_DIR}/controller.log"
    sleep "${MODEL_LOAD_COOLDOWN}"

    start_vlm_window window dfine vlm.detector.dfine 12185
    if ! wait_for_persistent_service dfine 12185 \
        ' -m vlm\.detector\.dfine( |$)' 300; then
        tmux kill-session -t "${VLM_SESSION}" 2>/dev/null || true
        return 1
    fi
    printf '[cooldown] D-FINE loaded; waiting %ss before GroundingDINO\n' \
        "${MODEL_LOAD_COOLDOWN}" | tee -a "${LOG_DIR}/controller.log"
    sleep "${MODEL_LOAD_COOLDOWN}"

    start_vlm_window window gdino vlm.detector.grounding_dino 12181
    if ! wait_for_persistent_service gdino 12181 \
        ' -m vlm\.detector\.grounding_dino( |$)' 300; then
        tmux kill-session -t "${VLM_SESSION}" 2>/dev/null || true
        return 1
    fi
    printf '[cooldown] Persistent VLM stack ready; waiting %ss before Habitat\n' \
        "${MODEL_LOAD_COOLDOWN}" | tee -a "${LOG_DIR}/controller.log"
    sleep "${MODEL_LOAD_COOLDOWN}"
}

start_group() {
    local name="$1"
    local log_file="$2"
    shift 2
    setsid "$@" >>"${log_file}" 2>&1 &
    PIDS["${name}"]=$!
    printf '[start] %s pid=%s\n' "${name}" "${PIDS[${name}]}"
}

stop_group() {
    local name="$1"
    local signal="${2:-TERM}"
    local pid="${PIDS[${name}]:-}"
    if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
        kill "-${signal}" -- "-${pid}" 2>/dev/null || true
        for _ in {1..20}; do
            kill -0 "${pid}" 2>/dev/null || break
            sleep 1
        done
        if kill -0 "${pid}" 2>/dev/null; then
            kill -KILL -- "-${pid}" 2>/dev/null || true
        fi
    fi
    unset 'PIDS['"${name}"']' || true
}

completed_count() {
    local continue_file="${OUTPUT_DIR}/continue.txt"
    if [[ ! -f "${continue_file}" ]]; then
        printf '0\n'
        return
    fi
    sed -n 's/^No\.\([0-9][0-9]*\) task is finished$/\1/p' "${continue_file}" |
        head -n 1
}

send_status() {
    [[ "${ENABLE_FEISHU_REPORT}" == "true" ]] || return 0
    "${PYTHON}" "${ROOT}/scripts/report_hm3dv2_greedy1000.py" --send \
        >>"${LOG_DIR}/feishu.log" 2>&1 || true
}

disable_reporter() {
    [[ "${ENABLE_FEISHU_REPORT}" == "true" ]] || return 0
    [[ -n "${CC_CONNECT_DATA_DIR}" ]] || return 0
    if [[ -s "${CRON_ID_FILE}" ]]; then
        local cron_id
        cron_id="$(tr -d '[:space:]' <"${CRON_ID_FILE}")"
        if [[ -n "${cron_id}" ]]; then
            cc-connect cron edit "${cron_id}" enabled false \
                --data-dir "${CC_CONNECT_DATA_DIR}" \
                >>"${LOG_DIR}/feishu.log" 2>&1 || true
        fi
    fi
}

cleanup_stack() {
    stop_group eval INT
    stop_group verifier TERM
    stop_group planner TERM
}

cleanup_runtime() {
    cleanup_stack
    stop_group roscore TERM
    stop_group kernel_trace TERM
    stop_group gpu_trace TERM
}

start_gpu_monitors() {
    start_group gpu_trace "${GPU_TRACE_LOG}" nvidia-smi --id=0 \
        --query-gpu=timestamp,pstate,temperature.gpu,power.draw,power.limit,memory.used,memory.total,utilization.gpu,utilization.memory,pcie.link.gen.current,pcie.link.width.current \
        --format=csv -l 1
    start_group kernel_trace "${KERNEL_GPU_LOG}" journalctl -kf --since now -o short-iso
}

fatal() {
    local reason="$1"
    printf '[fatal] %s\n' "${reason}" | tee -a "${LOG_DIR}/controller.log"
    write_state "stopped" "${reason}"
    send_status
    cleanup_runtime
    exit 1
}

trap 'write_state "stopped" "signal"; cleanup_runtime' INT TERM
trap 'cleanup_runtime' EXIT

if (( START_VLM_ONLY == 1 )); then
    :
elif [[ -e "${OUTPUT_DIR}/continue.txt" ]]; then
    if [[ ! -f "${OUTPUT_DIR}/evaluation_protocol.json" ||
          ! -f "${RUN_PROTOCOL_FILE}" ]]; then
        printf '[error] Refusing legacy checkpoint without protocol manifests: %s\n' \
            "${OUTPUT_DIR}" >&2
        printf '[hint] Start the 1 m protocol in a new output directory\n' >&2
        exit 2
    fi
    if ! protocol_matches; then
        printf '[error] Refusing checkpoint whose run protocol differs from this launch: %s\n' \
            "${OUTPUT_DIR}" >&2
        exit 2
    fi
    printf '[resume] Using existing checkpoint: %s\n' "${OUTPUT_DIR}/continue.txt"
elif [[ -d "${OUTPUT_DIR}" ]] &&
     find "${OUTPUT_DIR}" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null |
         grep -q .; then
    printf '[error] Refusing a non-empty output directory without a checkpoint: %s\n' \
        "${OUTPUT_DIR}" >&2
    exit 2
fi

if pgrep -f '/python([0-9.]*)? .*habitat_evaluation\.py' >/dev/null ||
   pgrep -f '/exploration_node([ ]|$)' >/dev/null; then
    printf '[error] Another evaluation/planner process is already running\n' >&2
    exit 2
fi
if ! [[ -x "${PYTHON}" ]]; then
    printf '[error] Python environment not found: %s\n' "${PYTHON}" >&2
    exit 2
fi
if (( START_VLM_ONLY == 0 )); then
    if ! "${PYTHON}" "${ROOT}/scripts/check_hypothesis_coverage.py" \
        --env-dir "${HYPOTHESIS_DIR}" --total "${TOTAL}"; then
        printf '[cloud] Missing hypotheses will be generated at episode start\n'
    fi
fi
if ! command -v tmux >/dev/null; then
    printf '[error] tmux is required for the resumable evaluation\n' >&2
    exit 2
fi
if ! [[ "${CONTROL_RATE_HZ}" =~ ^[0-9]+([.][0-9]+)?$ ]] ||
   ! awk -v rate="${CONTROL_RATE_HZ}" 'BEGIN { exit !(rate > 0) }' ||
   ! [[ "${TOTAL}" =~ ^[1-9][0-9]*$ ]] ||
   ! [[ "${HABITAT_SEED}" =~ ^[0-9]+$ ]] ||
   ! [[ "${BATCH_SIZE}" =~ ^[1-9][0-9]*$ ]] ||
   ! [[ "${VLM_REQUEST_TIMEOUT}" =~ ^[0-9]+([.][0-9]+)?$ ]] ||
   ! awk -v timeout="${VLM_REQUEST_TIMEOUT}" 'BEGIN { exit !(timeout > 0) }' ||
   ! [[ "${VLM_REQUEST_ATTEMPTS}" =~ ^[1-9][0-9]*$ ]] ||
   ! [[ "${MODEL_LOAD_COOLDOWN}" =~ ^[0-9]+$ ]] ||
   ! awk -v cooldown="${MODEL_LOAD_COOLDOWN}" 'BEGIN { exit !(cooldown >= 30) }' ||
   ! [[ "${GPU_POWER_LIMIT_MAX}" =~ ^[1-9][0-9]*([.][0-9]+)?$ ]] ||
   ! awk -v maximum="${GPU_POWER_LIMIT_MAX}" 'BEGIN { exit !(maximum <= 220) }' ||
   ! [[ "${GPU_IDLE_MEMORY_LIMIT}" =~ ^[1-9][0-9]*$ ]] ||
   ! [[ "${GPU_IDLE_STABLE_SECONDS}" =~ ^[1-9][0-9]*$ ]] ||
   ! awk -v stable="${GPU_IDLE_STABLE_SECONDS}" 'BEGIN { exit !(stable >= 60) }' ||
   ! [[ "${GPU_IDLE_TIMEOUT}" =~ ^[1-9][0-9]*$ ]] ||
   ! [[ "${FORCE_REGENERATE_HYPOTHESES}" =~ ^[01]$ ]] ||
   ! [[ "${SUCCESS_DISTANCE}" =~ ^[0-9]+([.][0-9]+)?$ ]] ||
   ! awk -v distance="${SUCCESS_DISTANCE}" 'BEGIN { exit !(distance > 0) }' ||
   ! [[ "${EXPLORATION_POLICY}" =~ ^(0|1|4)$ ]] ||
   ! [[ "${USE_MULTI_SOURCE}" =~ ^(true|false)$ ]] ||
   ! [[ "${USE_IG_FUSION}" =~ ^(true|false)$ ]] ||
   ! [[ "${USE_ADAPTIVE_ALPHA}" =~ ^(true|false)$ ]] ||
   ! [[ "${FUSION_ALPHA}" =~ ^(0(\.[0-9]+)?|1(\.0+)?)$ ]] ||
   ! [[ "${ALPHA_BASE}" =~ ^(0(\.[0-9]+)?|1(\.0+)?)$ ]] ||
   ! [[ "${TAU_CONF}" =~ ^[0-9]+([.][0-9]+)?$ ]] ||
   ! awk -v threshold="${TAU_CONF}" 'BEGIN { exit !(threshold > 0) }' ||
   ! [[ "${TOP_K_VALUE}" =~ ^[1-9][0-9]*$ ]] ||
   ! [[ "${WTRP_MAX_BRUTE_FORCE}" =~ ^[1-9][0-9]*$ ]] ||
   ! awk -v top_k="${TOP_K_VALUE}" -v brute="${WTRP_MAX_BRUTE_FORCE}" \
         'BEGIN { exit !(top_k <= 8 && brute <= 8 && brute <= top_k) }' ||
   ! [[ "${WTRP_TEMPERATURE}" =~ ^[0-9]+([.][0-9]+)?$ ]] ||
   ! awk -v temperature="${WTRP_TEMPERATURE}" 'BEGIN { exit !(temperature > 0) }' ||
   ! [[ "${WTRP_HYSTERESIS_RATIO}" =~ ^(0(\.[0-9]+)?|1(\.0+)?)$ ]] ||
   ! awk -v hysteresis="${WTRP_HYSTERESIS_RATIO}" 'BEGIN { exit !(hysteresis > 0) }' ||
   ! [[ "${ENABLE_FEISHU_REPORT}" =~ ^(true|false)$ ]] ||
   ! [[ "${VLM_VALIDATION_ENABLED}" =~ ^(true|false)$ ]] ||
   ! [[ "${EXPERIMENT_GIT_COMMIT}" =~ ^[0-9a-f]{40}$ ]] ||
   ! [[ "${EXPERIMENT_GIT_DIRTY_TRACKED}" =~ ^(true|false)$ ]] ||
   ! awk -v stable="${GPU_IDLE_STABLE_SECONDS}" -v timeout="${GPU_IDLE_TIMEOUT}" \
         'BEGIN { exit !(timeout >= stable) }'; then
    printf '[error] Evaluation, cooldown, or GPU guard configuration is invalid\n' >&2
    exit 2
fi
if (( START_VLM_ONLY == 0 )) && [[ -z "${DASHSCOPE_API_KEY:-}" ]]; then
    printf '[error] DASHSCOPE_API_KEY is required for the HM3D approach verifier\n' >&2
    exit 2
fi
if (( START_VLM_ONLY == 0 )) && [[ -z "${DEEPSEEK_API_KEY:-}" ]]; then
    printf '[error] DEEPSEEK_API_KEY is required for cloud hypothesis generation\n' >&2
    exit 2
fi
if ! [[ "${DASHSCOPE_REGION}" =~ ^(beijing|singapore|tokyo|virginia)$ ]]; then
    printf '[error] DASHSCOPE_REGION must be beijing, singapore, tokyo, or virginia\n' >&2
    exit 2
fi
if (( START_VLM_ONLY == 0 )) && [[ "${DASHSCOPE_REGION}" == "tokyo" ]] &&
   [[ -z "${DASHSCOPE_WORKSPACE_ID:-}" ]] &&
   [[ -z "${DASHSCOPE_BASE_URL:-}" ]]; then
    printf '[error] Tokyo DashScope requires DASHSCOPE_WORKSPACE_ID or DASHSCOPE_BASE_URL\n' >&2
    exit 2
fi

# All six HM3D targets should be cached.  DeepSeek V4 Flash is the online
# fallback only when a cache entry is missing, so local Ollama is unnecessary.
missing_cache=()
for label in couch tv chair toilet bed 'potted plant'; do
    if ! grep -F -q -- "${label}:" "${ROOT}/llm/answers/llm_answer_hm3d.txt"; then
        missing_cache+=("${label}")
    fi
done
if (( START_VLM_ONLY == 0 )) && (( ${#missing_cache[@]} > 0 )) &&
   [[ -z "${DEEPSEEK_API_KEY:-}" ]]; then
    printf '[error] Missing HM3D LLM cache entries (%s) and DEEPSEEK_API_KEY is unset\n' \
        "${missing_cache[*]}" >&2
    exit 2
fi

gpu_reason="$(gpu_health_reason)"
if [[ "${gpu_reason}" != "healthy" ]]; then
    if (( PREFLIGHT_ONLY == 0 )); then
        write_state "waiting_for_gpu" "${gpu_reason}"
    fi
    printf '[blocked] GPU is not healthy: %s\n' "${gpu_reason}" >&2
    if [[ "${gpu_reason}" == gpu_power_limit_* ]]; then
        printf '[hint] Apply the validated Prism guard before retrying: sudo nvidia-smi -pl %s\n' \
            "${GPU_POWER_LIMIT_MAX}" >&2
    fi
    exit 3
fi

# The navigation stack is per-run. The four expected local VLM services are
# handled separately and intentionally persist across runner invocations.
guarded_ports=(12184 12192)
if (( START_VLM_ONLY == 0 )); then
    guarded_ports+=(11311)
fi
for port in "${guarded_ports[@]}"; do
    if port_open "${port}"; then
        printf '[error] Unexpected ROS/VLM service on port %s\n' \
            "${port}" >&2
        exit 2
    fi
done

if (( PREFLIGHT_ONLY == 1 )); then
    if vlm_stack_present && ! vlm_stack_ready; then
        printf '[preflight] Partial or foreign VLM stack detected\n' >&2
        exit 2
    fi
    printf '[preflight] Passed without starting tmux, VLM, ROS, or evaluation services\n'
    exit 0
fi

if (( START_VLM_ONLY == 1 )); then
    mkdir -p "${LOG_DIR}"
    printf '\n[vlm-only] start requested at %s\n' "$(date --iso-8601=seconds)" |
        tee -a "${LOG_DIR}/controller.log"
    start_gpu_monitors
    if ! ensure_vlm_stack; then
        printf '[vlm-only] Persistent VLM startup failed\n' |
            tee -a "${LOG_DIR}/controller.log" >&2
        exit 1
    fi
    printf '[vlm-only] Persistent VLM stack is ready in tmux session %s\n' \
        "${VLM_SESSION}" | tee -a "${LOG_DIR}/controller.log"
    exit 0
fi

# Re-enter inside tmux so the long run survives terminal/bridge disconnects.
if [[ "${INFONAV_GREEDY1000_INNER:-0}" != "1" ]]; then
    if tmux has-session -t "${SESSION}" 2>/dev/null; then
        printf '[error] tmux session %s already exists\n' "${SESSION}" >&2
        exit 2
    fi
    # Create a placeholder pane, copy the selected runtime environment into the
    # session without exposing API keys in the process command line, then respawn.
    tmux new-session -d -s "${SESSION}" 'sleep 86400'
    for key in DASHSCOPE_API_KEY DASHSCOPE_BASE_URL DASHSCOPE_WORKSPACE_ID \
               DEEPSEEK_API_KEY DEEPSEEK_BASE_URL; do
        if [[ -n "${!key:-}" ]]; then
            tmux set-environment -t "${SESSION}" "${key}" "${!key}"
        fi
    done
    tmux set-environment -t "${SESSION}" QWEN_VL_MODEL "${QWEN_VL_MODEL}"
    tmux set-environment -t "${SESSION}" DEEPSEEK_MODEL "${DEEPSEEK_MODEL}"
    tmux set-environment -t "${SESSION}" QWEN_HYPOTHESIS_MODEL "${QWEN_HYPOTHESIS_MODEL}"
    tmux set-environment -t "${SESSION}" DEEPSEEK_HYPOTHESIS_MODEL "${DEEPSEEK_HYPOTHESIS_MODEL}"
    tmux respawn-pane -k -t "${SESSION}:0.0" \
        "env INFONAV_GREEDY1000_INNER=1 RUN_TAG='${RUN_TAG}' TOTAL='${TOTAL}' PROTOCOL_ID='${PROTOCOL_ID}' HABITAT_SEED='${HABITAT_SEED}' VLM_VALIDATION_ENABLED='${VLM_VALIDATION_ENABLED}' EXPERIMENT_GIT_COMMIT='${EXPERIMENT_GIT_COMMIT}' EXPERIMENT_GIT_DIRTY_TRACKED='${EXPERIMENT_GIT_DIRTY_TRACKED}' SUCCESS_DISTANCE='${SUCCESS_DISTANCE}' EXPLORATION_POLICY='${EXPLORATION_POLICY}' USE_MULTI_SOURCE='${USE_MULTI_SOURCE}' FUSION_ALPHA='${FUSION_ALPHA}' USE_IG_FUSION='${USE_IG_FUSION}' USE_ADAPTIVE_ALPHA='${USE_ADAPTIVE_ALPHA}' ALPHA_BASE='${ALPHA_BASE}' TAU_CONF='${TAU_CONF}' TOP_K_VALUE='${TOP_K_VALUE}' WTRP_TEMPERATURE='${WTRP_TEMPERATURE}' WTRP_HYSTERESIS_RATIO='${WTRP_HYSTERESIS_RATIO}' WTRP_MAX_BRUTE_FORCE='${WTRP_MAX_BRUTE_FORCE}' ENABLE_FEISHU_REPORT='${ENABLE_FEISHU_REPORT}' BATCH_SIZE='${BATCH_SIZE}' STALL_TIMEOUT='${STALL_TIMEOUT}' GPU_MEMORY_LIMIT='${GPU_MEMORY_LIMIT}' GPU_TEMPERATURE_LIMIT='${GPU_TEMPERATURE_LIMIT}' GPU_POWER_LIMIT_MAX='${GPU_POWER_LIMIT_MAX}' GPU_IDLE_MEMORY_LIMIT='${GPU_IDLE_MEMORY_LIMIT}' GPU_IDLE_STABLE_SECONDS='${GPU_IDLE_STABLE_SECONDS}' GPU_IDLE_TIMEOUT='${GPU_IDLE_TIMEOUT}' VLM_SESSION='${VLM_SESSION}' CONTROL_RATE_HZ='${CONTROL_RATE_HZ}' MODEL_LOAD_COOLDOWN='${MODEL_LOAD_COOLDOWN}' VLM_REQUEST_TIMEOUT='${VLM_REQUEST_TIMEOUT}' VLM_REQUEST_ATTEMPTS='${VLM_REQUEST_ATTEMPTS}' DASHSCOPE_REGION='${DASHSCOPE_REGION}' QWEN_HYPOTHESIS_MODEL='${QWEN_HYPOTHESIS_MODEL}' DEEPSEEK_HYPOTHESIS_MODEL='${DEEPSEEK_HYPOTHESIS_MODEL}' FORCE_REGENERATE_HYPOTHESES='${FORCE_REGENERATE_HYPOTHESES}' '${BASH_SOURCE[0]}' '${OUTPUT_DIR}'"
    printf '[started] tmux=%s output=%s\n' "${SESSION}" "${OUTPUT_DIR}"
    exit 0
fi

mkdir -p "${LOG_DIR}"
if [[ ! -f "${RUN_PROTOCOL_FILE}" ]]; then
    write_run_protocol
fi
write_state "starting"
printf '[config] protocol=%s output=%s total=%s batch=%s policy=%s success_distance=%sm HSVM_multi=%s fusion=(adaptive=%s,base=%s,IG=%s) WTRP=(top_k=%s,temp=%s,hysteresis=%s,brute=%s) rate=%sHz local_models=BLIP2x1,GroundingDINO,MobileSAM,D-FINE terminal_verifier=%s(%s) hypothesis_models=%s+%s\n' \
    "${PROTOCOL_ID}" "${OUTPUT_DIR}" "${TOTAL}" "${BATCH_SIZE}" \
    "${EXPLORATION_POLICY}" "${SUCCESS_DISTANCE}" "${USE_MULTI_SOURCE}" \
    "${USE_ADAPTIVE_ALPHA}" "${ALPHA_BASE}" "${USE_IG_FUSION}" \
    "${TOP_K_VALUE}" "${WTRP_TEMPERATURE}" "${WTRP_HYSTERESIS_RATIO}" \
    "${WTRP_MAX_BRUTE_FORCE}" "${CONTROL_RATE_HZ}" \
    "${VLM_VALIDATION_ENABLED}" "${QWEN_VL_MODEL}" \
    "${QWEN_HYPOTHESIS_MODEL}" "${DEEPSEEK_HYPOTHESIS_MODEL}" | tee -a "${LOG_DIR}/controller.log"

start_gpu_monitors
ensure_vlm_stack || fatal "persistent_vlm_start_failed"

start_group roscore "${LOG_DIR}/roscore.log" bash -lc \
    "source /opt/ros/noetic/setup.bash; exec roscore"
wait_for_port 11311 roscore 60 || fatal "roscore_start_failed"

start_stack() {
    local force_regenerate_arg=""
    if [[ "${FORCE_REGENERATE_HYPOTHESES}" == "1" ]]; then
        force_regenerate_arg="--force-regenerate"
    fi
    start_group planner "${LOG_DIR}/planner.log" bash -lc \
        "cd '${ROOT}'; source /opt/ros/noetic/setup.bash; source devel/setup.bash; exec roslaunch exploration_manager exploration.launch dataset:=hm3dv2 vlm_validation_enabled:='${VLM_VALIDATION_ENABLED}' exploration_policy:='${EXPLORATION_POLICY}' use_multi_source:='${USE_MULTI_SOURCE}' fusion_alpha:='${FUSION_ALPHA}' use_ig_fusion:='${USE_IG_FUSION}' use_adaptive_alpha:='${USE_ADAPTIVE_ALPHA}' alpha_base:='${ALPHA_BASE}' tau_conf:='${TAU_CONF}' top_k_value:='${TOP_K_VALUE}' wtrp_temperature:='${WTRP_TEMPERATURE}' wtrp_hysteresis_ratio:='${WTRP_HYSTERESIS_RATIO}' wtrp_max_brute_force:='${WTRP_MAX_BRUTE_FORCE}'"
    sleep 8
    if ! kill -0 "${PIDS[planner]}" 2>/dev/null; then
        fatal "planner_start_failed"
    fi

    if [[ "${VLM_VALIDATION_ENABLED}" == "true" ]]; then
        start_group verifier "${LOG_DIR}/verifier.log" bash -ic \
            "cd '${ROOT}'; source /opt/ros/noetic/setup.bash; source devel/setup.bash; exec env -u ALL_PROXY -u all_proxy -u HTTPS_PROXY -u https_proxy -u HTTP_PROXY -u http_proxy '${PYTHON}' src/planner/plan_env/scripts/vlm_approach_verifier.py _model:='${QWEN_VL_MODEL}' _api_region:='${DASHSCOPE_REGION}' _debug_save_images:=false _max_completion_tokens:=384"
        sleep 3
        if ! kill -0 "${PIDS[verifier]}" 2>/dev/null; then
            fatal "verifier_start_failed"
        fi
    else
        printf '[config] terminal VLM validation disabled for this planning-policy comparison\n' \
            | tee -a "${LOG_DIR}/controller.log"
    fi

    start_group eval "${LOG_DIR}/eval.log" bash -ic \
        "cd '${ROOT}'; source /opt/ros/noetic/setup.bash; source devel/setup.bash; exec env -u ALL_PROXY -u all_proxy -u HTTPS_PROXY -u https_proxy -u HTTP_PROXY -u http_proxy APEXNAV_LOG_LEVEL=MINIMAL PYTHONUNBUFFERED=1 USE_PARALLEL_BLIP2=false '${PYTHON}' habitat_evaluation.py --dataset hm3dv2 --detector-mode 2 --control-rate-hz '${CONTROL_RATE_HZ}' --max-episodes-per-process '${BATCH_SIZE}' --realtime-hypothesis --vlm-model '${QWEN_HYPOTHESIS_MODEL}' --llm-model '${DEEPSEEK_HYPOTHESIS_MODEL}' ${force_regenerate_arg} habitat.seed='${HABITAT_SEED}' habitat.task.measurements.success.success_distance='${SUCCESS_DISTANCE}' llm.llm_client.llm_client=deepseek need_video=false video_output_path='${OUTPUT_DIR}'"
    sleep 5
    if ! kill -0 "${PIDS[eval]}" 2>/dev/null; then
        fatal "evaluation_start_failed"
    fi
}

start_stack
write_state "running"
send_status

last_count="$(completed_count)"
last_progress_time="$(date +%s)"
batch_start="${last_count}"
last_gpu_log_time=0
printf '[monitor] resume count=%s\n' "${last_count}" | tee -a "${LOG_DIR}/controller.log"

while (( last_count < TOTAL )); do
    sleep 10

    gpu_reason="$(gpu_health_reason)"
    if [[ "${gpu_reason}" != "healthy" ]]; then
        fatal "${gpu_reason}"
    fi
    vlm_stack_ready || fatal "persistent_vlm_unhealthy"

    current_count="$(completed_count)"
    if ! [[ "${current_count}" =~ ^[0-9]+$ ]]; then
        fatal "checkpoint_parse_failed"
    fi
    if (( current_count > last_count )); then
        last_count="${current_count}"
        last_progress_time="$(date +%s)"
        printf '[progress] %s/%s\n' "${last_count}" "${TOTAL}" |
            tee -a "${LOG_DIR}/controller.log"
    fi
    if (( current_count >= TOTAL )); then
        break
    fi

    now="$(date +%s)"
    if (( now - last_gpu_log_time >= 60 )); then
        printf '[gpu] %s\n' "$(gpu_query)" | tee -a "${LOG_DIR}/controller.log"
        last_gpu_log_time="${now}"
    fi
    if (( now - last_progress_time > STALL_TIMEOUT )); then
        printf '[recover] no completed episode for %ss; restarting navigation stack\n' \
            "$((now - last_progress_time))" | tee -a "${LOG_DIR}/controller.log"
        cleanup_stack
        sleep 5
        start_stack
        last_progress_time="$(date +%s)"
        batch_start="${current_count}"
        continue
    fi

    verifier_exited=false
    if [[ "${VLM_VALIDATION_ENABLED}" == "true" ]] &&
       ! kill -0 "${PIDS[verifier]:-}" 2>/dev/null; then
        verifier_exited=true
    fi
    if ! kill -0 "${PIDS[eval]}" 2>/dev/null ||
       ! kill -0 "${PIDS[planner]}" 2>/dev/null ||
       [[ "${verifier_exited}" == "true" ]]; then
        if (( current_count - batch_start >= BATCH_SIZE )); then
            printf '[batch] evaluator completed %s episodes and exited cleanly; restarting navigation stack\n' \
                "$((current_count - batch_start))" | tee -a "${LOG_DIR}/controller.log"
        else
            printf '[recover] navigation process exited; restarting from checkpoint %s\n' \
                "${current_count}" | tee -a "${LOG_DIR}/controller.log"
        fi
        cleanup_stack
        sleep 5
        start_stack
        last_progress_time="$(date +%s)"
        batch_start="${current_count}"
        continue
    fi

done

write_state "completed"
send_status
disable_reporter
printf '[done] completed %s/%s episodes\n' "${last_count}" "${TOTAL}" |
    tee -a "${LOG_DIR}/controller.log"
