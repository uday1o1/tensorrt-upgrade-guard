#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SOURCE_ID="$(git -C "${PROJECT_ROOT}" rev-parse HEAD)"
STATE_BASE="${PROJECT_ROOT}/.upgrade-guard/cuda-pm"
SMOKE_ONLY="${UG_SMOKE_ONLY:-0}"
SANITIZER_ONLY="${UG_SANITIZER_ONLY:-0}"
if [[ "${SMOKE_ONLY}" == "1" && "${SANITIZER_ONLY}" == "1" ]]; then
  printf 'Smoke and sanitizer-only modes are mutually exclusive.\n' >&2
  exit 2
fi
RUN_MODE="full"
STATE_ROOT="${STATE_BASE}/runs/${SOURCE_ID}"
if [[ "${SMOKE_ONLY}" == "1" ]]; then
  RUN_MODE="smoke"
  STATE_ROOT="${STATE_BASE}/checks/${SOURCE_ID}/smoke"
elif [[ "${SANITIZER_ONLY}" == "1" ]]; then
  RUN_MODE="sanitizer"
  STATE_ROOT="${STATE_BASE}/checks/${SOURCE_ID}/sanitizer"
fi
LOG_ROOT="${STATE_ROOT}/logs"
MATRIX_LOCK="${STATE_ROOT}/matrix.lock.json"
CORE_CORPUS="${PROJECT_ROOT}/.upgrade-guard/corpora/v1-core"
PLUGIN_CORPUS="${STATE_ROOT}/corpora/plugin"
MOBILENET_CORPUS="${STATE_ROOT}/corpora/mobilenet"
REGISTRY_NAME="tensorrt-upgrade-guard-registry-${SOURCE_ID:0:12}"
REGISTRY_VOLUME="${REGISTRY_NAME}-data"
REGISTRY_ADDRESS="127.0.0.1:5500"
REGISTRY_IMAGE="registry@sha256:46faa9a1ae6813194b53921a370f2f4f8c5e1aae228a89bceafef5847a6a3278"
BASELINE_BASE="nvcr.io/nvidia/tensorrt:26.06-py3@sha256:2a5a0a9a32ec5ddc1c384c15ddcf3b89ddc4f8647e7ee7ae708d844210183a1e"
CANDIDATE_BASE="nvcr.io/nvidia/tensorrt:26.07-py3@sha256:b82db1abc23750ab0069abc99bbe4ea29138dbdc23ea39861199e2346638b48a"
GPU_INDEX="${UG_GPU_INDEX:-0}"
CURRENT_STEP="initialization"
THROUGH_STEP="${UG_THROUGH_STEP:-}"
GPU_UUID=""

DONE_ROOT="${STATE_ROOT}/done"

mkdir -p "${DONE_ROOT}" "${LOG_ROOT}"

failure_report() {
  local exit_code=$?
  if command -v docker >/dev/null 2>&1 \
    && docker container inspect "${REGISTRY_NAME}" >/dev/null 2>&1; then
    docker container rm --force "${REGISTRY_NAME}" >/dev/null 2>&1 || true
  fi
  if [[ ${exit_code} -ne 0 ]]; then
    printf 'FAILED step=%s exit=%s\n' "${CURRENT_STEP}" "${exit_code}" >&2
    printf 'Resume with: bash scripts/run_cuda_pm_qualification.sh\n' >&2
  fi
  exit "${exit_code}"
}
trap failure_report EXIT

run_step() {
  local name=$1
  shift
  CURRENT_STEP="${name}"
  if "${UV[@]}" run --frozen python scripts/qualification_state.py verify \
    --state "${STATE_ROOT}" --project "${PROJECT_ROOT}" --step "${name}" \
    --source "${SOURCE_ID}" --gpu "${GPU_UUID}" --mode "${RUN_MODE}"; then
    printf 'SKIP completed step: %s\n' "${name}"
    if [[ "${THROUGH_STEP}" == "${name}" ]]; then
      exit 0
    fi
    return
  fi
  if [[ -e "${DONE_ROOT}/${name}" || -e "${DONE_ROOT}/${name}.json" ]]; then
    printf 'RERUN invalid or incomplete step: %s\n' "${name}"
  fi
  printf 'RUN step: %s\n' "${name}"
  "$@" 2>&1 | tee "${LOG_ROOT}/${name}.log"
  "${UV[@]}" run --frozen python scripts/qualification_state.py record \
    --state "${STATE_ROOT}" --project "${PROJECT_ROOT}" --step "${name}" \
    --source "${SOURCE_ID}" --gpu "${GPU_UUID}" --mode "${RUN_MODE}"
  if [[ "${THROUGH_STEP}" == "${name}" ]]; then
    exit 0
  fi
}

invocation_guard() {
  cd "${PROJECT_ROOT}"
  [[ "$(git rev-parse HEAD)" == "${SOURCE_ID}" ]]
  [[ -z "$(git status --porcelain --untracked-files=normal)" ]]
  [[ "$(uname -s)" == "Linux" ]]
  [[ "$(uname -m)" == "x86_64" ]]
  command -v python3
  command -v docker
  command -v nvidia-smi
  command -v curl
  local docker_platform
  docker version >/dev/null
  docker_platform="$(docker info --format '{{.OSType}}/{{.Architecture}}')"
  [[ "${docker_platform}" == "linux/x86_64" || "${docker_platform}" == "linux/amd64" ]]
  if [[ -n "${UG_EXPECTED_GPU_UUID:-}" ]]; then
    GPU_UUID="$(nvidia-smi --id="${UG_EXPECTED_GPU_UUID}" \
      --query-gpu=uuid --format=csv,noheader | tr -d '[:space:]')"
    [[ "${GPU_UUID}" == "${UG_EXPECTED_GPU_UUID}" ]]
  else
    GPU_UUID="$(nvidia-smi --id="${GPU_INDEX}" \
      --query-gpu=uuid --format=csv,noheader | tr -d '[:space:]')"
  fi
  [[ "${GPU_UUID}" == GPU-* ]]
  if [[ -f "${STATE_ROOT}/gpu.uuid" \
    && "$(<"${STATE_ROOT}/gpu.uuid")" != "${GPU_UUID}" ]]; then
    printf 'This source-specific run already belongs to another GPU UUID.\n' >&2
    return 1
  fi
}

select_uv() {
  if command -v uv >/dev/null 2>&1 && [[ "$(uv --version)" == uv\ 0.11.23* ]]; then
    UV=(uv)
    return
  fi
  local bootstrap="${STATE_BASE}/uv-bootstrap"
  if [[ ! -x "${bootstrap}/bin/uv" ]]; then
    python3 -m venv "${bootstrap}"
    "${bootstrap}/bin/python" -m pip install --disable-pip-version-check "uv==0.11.23"
  fi
  UV=("${bootstrap}/bin/uv")
}

preflight() {
  cd "${PROJECT_ROOT}"
  git rev-parse HEAD > "${STATE_ROOT}/source.commit"
  docker version
  nvidia-smi
  printf '%s\n' "${GPU_UUID}" > "${STATE_ROOT}/gpu.uuid"
  nvidia-smi --id="${GPU_UUID}" \
    --query-gpu=name,uuid,compute_cap,memory.total,driver_version,vbios_version,power.limit \
    --format=csv,noheader > "${STATE_ROOT}/gpu-preflight.csv"
}

cpu_verify() {
  cd "${PROJECT_ROOT}"
  "${UV[@]}" sync --frozen
  "${UV[@]}" run --frozen python scripts/generate_schemas.py
  git diff --exit-code -- schemas
  "${UV[@]}" run --frozen python scripts/check_repository_docs.py
  "${UV[@]}" run --frozen ruff check .
  "${UV[@]}" run --frozen ruff format --check .
  "${UV[@]}" run --frozen mypy
  "${UV[@]}" run --frozen pytest --cov=upgrade_guard --cov-report=term-missing
  git diff --exit-code
  [[ -z "$(git status --porcelain --untracked-files=normal)" ]]
}

start_local_registry() {
  docker pull "${REGISTRY_IMAGE}"
  if docker container inspect "${REGISTRY_NAME}" >/dev/null 2>&1; then
    docker container rm --force "${REGISTRY_NAME}" >/dev/null
  fi
  docker volume create \
    --label com.udayarora.upgradeguard.owner=tensorrt-upgrade-guard \
    --label "com.udayarora.upgradeguard.source=${SOURCE_ID}" \
    "${REGISTRY_VOLUME}" >/dev/null
  docker run -d --name "${REGISTRY_NAME}" \
    --label com.udayarora.upgradeguard.owner=tensorrt-upgrade-guard \
    --label "com.udayarora.upgradeguard.source=${SOURCE_ID}" \
    -p "127.0.0.1:5500:5000" \
    --mount "type=volume,src=${REGISTRY_VOLUME},dst=/var/lib/registry" \
    "${REGISTRY_IMAGE}" >/dev/null
  for _ in $(seq 1 30); do
    if curl --fail --silent "http://${REGISTRY_ADDRESS}/v2/" >/dev/null; then
      return
    fi
    sleep 1
  done
  printf 'Local registry did not become ready.\n' >&2
  return 1
}

build_workers() {
  cd "${PROJECT_ROOT}"
  docker pull "${BASELINE_BASE}"
  docker pull "${CANDIDATE_BASE}"
  docker build --pull=false \
    --build-arg "BASE_IMAGE=${BASELINE_BASE}" \
    --build-arg "BASE_MANIFEST_DIGEST=sha256:2a5a0a9a32ec5ddc1c384c15ddcf3b89ddc4f8647e7ee7ae708d844210183a1e" \
    --tag "${REGISTRY_ADDRESS}/upgrade-guard/worker:baseline" \
    --file containers/Dockerfile.worker .
  docker build --pull=false \
    --build-arg "BASE_IMAGE=${CANDIDATE_BASE}" \
    --build-arg "BASE_MANIFEST_DIGEST=sha256:b82db1abc23750ab0069abc99bbe4ea29138dbdc23ea39861199e2346638b48a" \
    --tag "${REGISTRY_ADDRESS}/upgrade-guard/worker:candidate" \
    --file containers/Dockerfile.worker .
  docker push "${REGISTRY_ADDRESS}/upgrade-guard/worker:baseline"
  docker push "${REGISTRY_ADDRESS}/upgrade-guard/worker:candidate"
  "${UV[@]}" run --frozen python scripts/qualification_state.py capture-workers \
    --output "${STATE_ROOT}/worker-images.json" \
    "${REGISTRY_ADDRESS}/upgrade-guard/worker:baseline" \
    "${REGISTRY_ADDRESS}/upgrade-guard/worker:candidate"
}

ensure_worker_registry() {
  local accept='application/vnd.oci.image.manifest.v1+json, application/vnd.docker.distribution.manifest.v2+json'
  if curl --fail --silent --head --header "Accept: ${accept}" \
    "http://${REGISTRY_ADDRESS}/v2/upgrade-guard/worker/manifests/baseline" >/dev/null \
    && curl --fail --silent --head --header "Accept: ${accept}" \
      "http://${REGISTRY_ADDRESS}/v2/upgrade-guard/worker/manifests/candidate" >/dev/null; then
    return
  fi
  printf 'Worker registry content is absent; rebuilding exact workers.\n'
  build_workers
  "${UV[@]}" run --frozen python scripts/qualification_state.py record \
    --state "${STATE_ROOT}" --project "${PROJECT_ROOT}" --step worker-images \
    --source "${SOURCE_ID}" --gpu "${GPU_UUID}" --mode "${RUN_MODE}"
}

preserve_stale_matrix() {
  if [[ ! -f "${MATRIX_LOCK}" ]]; then
    return
  fi
  if "${UV[@]}" run --frozen python scripts/qualification_state.py verify-worker-lock \
    --workers "${STATE_ROOT}/worker-images.json" --matrix "${MATRIX_LOCK}"; then
    return
  fi
  local preserved="${MATRIX_LOCK}.stale-$(date -u +%Y%m%dT%H%M%SZ)"
  mv "${MATRIX_LOCK}" "${preserved}"
  printf 'Preserved stale matrix lock at %s\n' "${preserved}"
}

lock_matrix() {
  cd "${PROJECT_ROOT}"
  local gpu_uuid
  gpu_uuid="$(<"${STATE_ROOT}/gpu.uuid")"
  MATRIX_PATH="${STATE_ROOT}/matrix.yaml" GPU_UUID_VALUE="${gpu_uuid}" \
    "${UV[@]}" run --frozen python -c \
    'import os,yaml; from pathlib import Path; p=Path("matrices/examples/controlled-minor.yaml"); v=yaml.safe_load(p.read_text()); v["gpu_uuid"]=os.environ["GPU_UUID_VALUE"]; Path(os.environ["MATRIX_PATH"]).write_text(yaml.safe_dump(v,sort_keys=False))'
  if [[ -f "${MATRIX_LOCK}" ]]; then
    "${UV[@]}" run --frozen python -c \
      'import sys; from pathlib import Path; from upgrade_guard.contracts.environment import MatrixLock; p=Path(sys.argv[1]); m=MatrixLock.model_validate_json(p.read_text()); assert m.lock_sha256 == m.computed_sha256()' \
      "${MATRIX_LOCK}"
  else
    "${UV[@]}" run --frozen upgrade-guard matrix lock "${STATE_ROOT}/matrix.yaml" \
      --out "${MATRIX_LOCK}" --json
  fi
  QUALIFICATION_PATH="${STATE_ROOT}/full.yaml" GPU_UUID_VALUE="${gpu_uuid}" \
    LOCK_PATH="${MATRIX_LOCK}" \
    "${UV[@]}" run --frozen python -c \
    'import os,yaml; from pathlib import Path; p=Path("qualification/full.yaml"); v=yaml.safe_load(p.read_text()); v["hardware_validity"]["selected_gpu_uuid"]=os.environ["GPU_UUID_VALUE"]; v["environment_lock"]=os.environ["LOCK_PATH"]; Path(os.environ["QUALIFICATION_PATH"]).write_text(yaml.safe_dump(v,sort_keys=False))'
}

materialize_corpora() {
  cd "${PROJECT_ROOT}"
  materialize_corpus_atomic "${CORE_CORPUS}" materialize_core_corpus
  materialize_corpus_atomic "${PLUGIN_CORPUS}" materialize_plugin_corpus
  materialize_corpus_atomic "${MOBILENET_CORPUS}" materialize_mobilenet_corpus
}

materialize_sanitizer_corpora() {
  cd "${PROJECT_ROOT}"
  materialize_corpus_atomic "${CORE_CORPUS}" materialize_core_corpus
  materialize_corpus_atomic "${PLUGIN_CORPUS}" materialize_plugin_corpus
}

materialize_corpus_atomic() {
  local destination=$1
  local producer=$2
  if [[ -d "${destination}" ]] \
    && "${UV[@]}" run --frozen python scripts/qualification_state.py \
      verify-corpus "${destination}"; then
    return
  fi
  if [[ -e "${destination}" ]]; then
    local preserved="${destination}.invalid-${SOURCE_ID:0:12}-$(date -u +%Y%m%dT%H%M%SZ)"
    mv "${destination}" "${preserved}"
    printf 'Preserved invalid corpus at %s\n' "${preserved}"
  fi
  mkdir -p "$(dirname "${destination}")"
  local staging
  staging="$(mktemp -d "$(dirname "${destination}")/.corpus-staging.XXXXXX")"
  local generated="${staging}/corpus"
  "${producer}" "${generated}"
  "${UV[@]}" run --frozen python scripts/qualification_state.py verify-corpus "${generated}"
  mv "${generated}" "${destination}"
  rmdir "${staging}"
}

materialize_core_corpus() {
  "${UV[@]}" run --frozen upgrade-guard corpus materialize corpus/registry.yaml \
    --out "$1" --json
}

materialize_plugin_corpus() {
  "${UV[@]}" run --frozen python scripts/materialize_plugin_corpus.py "$1"
}

materialize_mobilenet_corpus() {
  "${UV[@]}" run --frozen python scripts/materialize_mobilenet_corpus.py "$1"
}

load_worker_identities() {
  mapfile -t WORKER_IMAGES < <(
    "${UV[@]}" run --frozen python -c \
      'import sys; from pathlib import Path; from upgrade_guard.contracts.environment import MatrixLock; m=MatrixLock.model_validate_json(Path(sys.argv[1]).read_text()); [print(e.worker_image.canonical_reference) for e in m.environments]' \
      "${MATRIX_LOCK}"
  )
  BASELINE_WORKER="${WORKER_IMAGES[0]}"
  CANDIDATE_WORKER="${WORKER_IMAGES[1]}"
  GPU_UUID="$(<"${STATE_ROOT}/gpu.uuid")"
}

wait_for_idle_observation() {
  local output=$1
  for _ in $(seq 1 60); do
    if "${UV[@]}" run --frozen python scripts/hardware_validity.py capture \
      --specification "${STATE_ROOT}/full.yaml" --gpu "${GPU_UUID}" \
      --output "${output}"; then
      return
    fi
    sleep 1
  done
  printf 'The selected GPU did not reach the locked idle policy.\n' >&2
  return 1
}

gpu_run() {
  local image=$1
  local corpus=$2
  local output=$3
  shift 3
  mkdir -p "${output}"
  local user_id group_id home container_name status
  user_id="$(id -u)"
  group_id="$(id -g)"
  home=/home/upgrade-guard
  container_name="upgrade-guard-${SOURCE_ID:0:8}-${BASHPID}-${RANDOM}"
  if docker run --rm --name "${container_name}" --init --user "${user_id}:${group_id}" \
    --gpus "device=${GPU_UUID}" \
    --network none --read-only --cap-drop ALL \
    --security-opt no-new-privileges --pids-limit 512 --ipc private \
    --tmpfs /tmp:rw,noexec,nosuid,size=1073741824 \
    --tmpfs "${home}:rw,noexec,nosuid,nodev,size=1073741824,uid=${user_id},gid=${group_id},mode=0700" \
    --mount "type=bind,src=${PROJECT_ROOT},dst=/opt/upgrade-guard,readonly" \
    --mount "type=bind,src=${corpus},dst=/corpus,readonly" \
    --mount "type=bind,src=${output},dst=/output" \
    --env PYTHONPATH=/opt/upgrade-guard/src \
    --env "HOME=${home}" \
    --env "XDG_CACHE_HOME=${home}/.cache" \
    --env "XDG_CONFIG_HOME=${home}/.config" \
    --env "CUDA_CACHE_PATH=${home}/.cache/cuda" \
    --entrypoint "" \
    "${image}" "$@"; then
    return 0
  else
    status=$?
    docker container rm --force "${container_name}" >/dev/null 2>&1 || true
    return "${status}"
  fi
}

run_core_qualification() {
  cd "${PROJECT_ROOT}"
  if [[ -f "${STATE_ROOT}/core-run/qualification-summary.json" ]]; then
    "${UV[@]}" run --frozen upgrade-guard compare "${STATE_ROOT}/core-run" --json
    return
  fi
  "${UV[@]}" run --frozen upgrade-guard qualify "${STATE_ROOT}/full.yaml" \
    --out "${STATE_ROOT}/core-run" --json
  "${UV[@]}" run --frozen upgrade-guard compare "${STATE_ROOT}/core-run" --json
}

compile_plugins() {
  load_worker_identities
  local names=(baseline candidate)
  local images=("${BASELINE_WORKER}" "${CANDIDATE_WORKER}")
  for index in 0 1; do
    local output="${STATE_ROOT}/plugin/${names[${index}]}"
    gpu_run "${images[${index}]}" "${PROJECT_ROOT}" "${output}" \
      cmake -S /opt/upgrade-guard -B /output/build -G Ninja \
        -DCMAKE_BUILD_TYPE=RelWithDebInfo \
        -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
        -DUPGRADE_GUARD_BUILD_TESTS=ON \
        -DUPGRADE_GUARD_BUILD_FAULTS=ON
    gpu_run "${images[${index}]}" "${PROJECT_ROOT}" "${output}" \
      cmake --build /output/build --parallel
    gpu_run "${images[${index}]}" "${PROJECT_ROOT}" "${output}" \
      ctest --test-dir /output/build --output-on-failure
  done
}

run_plugin_benchmark() {
  load_worker_identities
  local output="${STATE_ROOT}/plugin/candidate"
  wait_for_idle_observation "${output}/plugin-benchmark-idle.json"
  gpu_run "${CANDIDATE_WORKER}" "${PROJECT_ROOT}" "${output}" \
    /output/build/upgrade_guard_kernel_benchmark \
    > "${output}/plugin-benchmark-precondition.json"
  "${UV[@]}" run --frozen python scripts/hardware_validity.py capture \
    --specification "${STATE_ROOT}/full.yaml" --gpu "${GPU_UUID}" --loaded \
    --output "${output}/plugin-benchmark-before.json"
  gpu_run "${CANDIDATE_WORKER}" "${PROJECT_ROOT}" "${output}" \
    /output/build/upgrade_guard_kernel_benchmark \
    > "${output}/plugin-benchmark.json"
  "${UV[@]}" run --frozen python scripts/hardware_validity.py capture \
    --specification "${STATE_ROOT}/full.yaml" --gpu "${GPU_UUID}" --loaded \
    --output "${output}/plugin-benchmark-after.json"
  "${UV[@]}" run --frozen python scripts/hardware_validity.py transition \
    --specification "${STATE_ROOT}/full.yaml" \
    --before "${output}/plugin-benchmark-before.json" \
    --after "${output}/plugin-benchmark-after.json" \
    --output "${output}/plugin-benchmark-validity.json"
  "${UV[@]}" run --frozen python -c \
    'import json,sys; v=json.load(open(sys.argv[1])); assert v["status"]=="passed" and not v["profiled"]' \
    "${output}/plugin-benchmark.json"
}

run_gpu_smoke() {
  load_worker_identities
  local output="${STATE_ROOT}/smoke"
  local plugin_source="${STATE_ROOT}/plugin/candidate/build/libupgrade_guard_residual_rmsnorm.so"
  mkdir -p "${output}/plugin" "${output}/standard"
  cp "${plugin_source}" "${output}/plugin/libupgrade_guard_residual_rmsnorm.so"
  write_plugin_profile "${output}/plugin/profile.json"
  cat > "${output}/standard/profile.json" <<'JSON'
{
  "tokens": {"min": [1, 8, 256], "opt": [4, 128, 256], "max": [8, 512, 256]},
  "mask": {"min": [1, 1, 1, 8], "opt": [4, 1, 1, 128], "max": [8, 1, 1, 512]}
}
JSON
  gpu_run "${CANDIDATE_WORKER}" "${CORE_CORPUS}" "${output}" \
    python3 -m upgrade_guard.worker.build_engine \
      --model /corpus/models/tiny-transformer-fp32.onnx \
      --profile /output/standard/profile.json \
      --engine /output/standard/engine.plan \
      --inspector /output/standard/inspector.json \
      --timing-cache /output/standard/timing.cache \
      --result /output/standard/build.json
  for case_name in b1_s8 b1_s128; do
    gpu_run "${CANDIDATE_WORKER}" "${CORE_CORPUS}" "${output}" \
      python3 -m upgrade_guard.worker.run_correctness \
        --engine /output/standard/engine.plan \
        --input "tokens=/corpus/inputs/tiny-transformer-fp32/${case_name}/tokens.npy" \
        --input "mask=/corpus/inputs/tiny-transformer-fp32/${case_name}/mask.npy" \
        --output "/output/standard/${case_name}/outputs" \
        --result "/output/standard/${case_name}/correctness.json" \
        --repetitions 20
  done
  gpu_run "${CANDIDATE_WORKER}" "${PLUGIN_CORPUS}" "${output}" \
    python3 -m upgrade_guard.worker.build_engine \
      --model /corpus/residual-rmsnorm-fp32.onnx \
      --profile /output/plugin/profile.json \
      --engine /output/plugin/engine.plan \
      --inspector /output/plugin/inspector.json \
      --timing-cache /output/plugin/timing.cache \
      --result /output/plugin/build.json \
      --plugin /output/plugin/libupgrade_guard_residual_rmsnorm.so
  gpu_run "${CANDIDATE_WORKER}" "${PLUGIN_CORPUS}" "${output}" \
    python3 -m upgrade_guard.worker.run_correctness \
      --engine /output/plugin/engine.plan \
      --input x=/corpus/fp32/tail-random-h259/x.npy \
      --input residual=/corpus/fp32/tail-random-h259/residual.npy \
      --input gamma=/corpus/fp32/tail-random-h259/gamma.npy \
      --output /output/plugin/outputs \
      --result /output/plugin/correctness.json \
      --repetitions 20 \
      --plugin /output/plugin/libupgrade_guard_residual_rmsnorm.so
  "${UV[@]}" run --frozen python scripts/validate_gpu_smoke.py \
    --core-corpus "${CORE_CORPUS}" --plugin-corpus "${PLUGIN_CORPUS}" \
    --runs "${output}" --output "${output}/validation.json"
}

write_plugin_profile() {
  local destination=$1
  cat > "${destination}" <<'JSON'
{
  "x": {"min": [1, 1, 7], "opt": [2, 17, 256], "max": [8, 512, 259]},
  "residual": {"min": [1, 1, 7], "opt": [2, 17, 256], "max": [8, 512, 259]},
  "gamma": {"min": [7], "opt": [256], "max": [259]}
}
JSON
}

run_plugin_matrix() {
  load_worker_identities
  local corpus="${PLUGIN_CORPUS}"
  local runs="${STATE_ROOT}/plugin-runs"
  local names=(baseline candidate)
  local images=("${BASELINE_WORKER}" "${CANDIDATE_WORKER}")
  for index in 0 1; do
    local environment="${names[${index}]}"
    local image="${images[${index}]}"
    local plugin="${STATE_ROOT}/plugin/${environment}/build/libupgrade_guard_residual_rmsnorm.so"
    mkdir -p "${runs}/${environment}"
    cp "${plugin}" "${runs}/${environment}/libupgrade_guard_residual_rmsnorm.so"
    local plugin_container="/output/${environment}/libupgrade_guard_residual_rmsnorm.so"
    for precision in fp32 fp16; do
      local output="${runs}/${environment}/${precision}"
      mkdir -p "${output}"
      write_plugin_profile "${output}/profile.json"
      gpu_run "${image}" "${corpus}" "${runs}" \
        python3 -m upgrade_guard.worker.build_engine \
          --model "/corpus/residual-rmsnorm-${precision}.onnx" \
          --profile "/output/${environment}/${precision}/profile.json" \
          --engine "/output/${environment}/${precision}/engine.plan" \
          --inspector "/output/${environment}/${precision}/inspector.json" \
          --timing-cache "/output/${environment}/${precision}/timing.cache" \
          --result "/output/${environment}/${precision}/build.json" \
          --plugin "${plugin_container}"
      for case_path in "${corpus}/${precision}"/*; do
        local case_name
        case_name="$(basename "${case_path}")"
        mkdir -p "${output}/${case_name}"
        gpu_run "${image}" "${corpus}" "${runs}" \
          python3 -m upgrade_guard.worker.run_correctness \
            --engine "/output/${environment}/${precision}/engine.plan" \
            --input "x=/corpus/${precision}/${case_name}/x.npy" \
            --input "residual=/corpus/${precision}/${case_name}/residual.npy" \
            --input "gamma=/corpus/${precision}/${case_name}/gamma.npy" \
            --output "/output/${environment}/${precision}/${case_name}/outputs" \
            --result "/output/${environment}/${precision}/${case_name}/correctness.json" \
            --repetitions 20 \
            --plugin "${plugin_container}"
      done
    done
  done
  "${UV[@]}" run --frozen python scripts/validate_plugin_outputs.py \
    --corpus "${corpus}" --runs "${runs}" --output "${runs}/validation.json"
}

write_mobilenet_profile() {
  local destination=$1
  cat > "${destination}" <<'JSON'
{"x": {"min": [1, 3, 160, 160], "opt": [8, 3, 224, 224], "max": [16, 3, 320, 320]}}
JSON
}

run_mobilenet_matrix() {
  load_worker_identities
  local corpus="${MOBILENET_CORPUS}"
  local runs="${STATE_ROOT}/mobilenet-runs"
  local names=(baseline candidate)
  local images=("${BASELINE_WORKER}" "${CANDIDATE_WORKER}")
  for index in 0 1; do
    local environment="${names[${index}]}"
    local image="${images[${index}]}"
    mkdir -p "${runs}/${environment}"
    write_mobilenet_profile "${runs}/${environment}/profile.json"
    gpu_run "${image}" "${corpus}" "${runs}" \
      python3 -m upgrade_guard.worker.build_engine \
        --model /corpus/mobilenetv3-small-075-dynamic.onnx \
        --profile "/output/${environment}/profile.json" \
        --engine "/output/${environment}/engine.plan" \
        --inspector "/output/${environment}/inspector.json" \
        --timing-cache "/output/${environment}/timing.cache" \
        --result "/output/${environment}/build.json"
    for case_path in "${corpus}/inputs"/*; do
      local case_name
      case_name="$(basename "${case_path}")"
      gpu_run "${image}" "${corpus}" "${runs}" \
        python3 -m upgrade_guard.worker.run_correctness \
          --engine "/output/${environment}/engine.plan" \
          --input "x=/corpus/inputs/${case_name}/x.npy" \
          --output "/output/${environment}/${case_name}/outputs" \
          --result "/output/${environment}/${case_name}/correctness.json" \
          --repetitions 20
    done
  done
  "${UV[@]}" run --frozen python scripts/validate_mobilenet_outputs.py \
    --corpus "${corpus}" --runs "${runs}" --output "${runs}/validation.json"
}

run_aa_pilot() {
  load_worker_identities
  local output="${STATE_ROOT}/aa"
  local corpus="${CORE_CORPUS}"
  local trtexec_path
  local stream_option
  if [[ -d "${output}" && ! -f "${output}/validation.json" ]]; then
    mv "${output}" "${output}.partial-$(date -u +%Y%m%dT%H%M%SZ)"
  fi
  mkdir -p "${output}"
  "${UV[@]}" run --frozen python -c \
    'import json,sys; from pathlib import Path; p=Path(sys.argv[1]); json.dump({"tokens":{"min":[1,8,256],"opt":[4,128,256],"max":[8,512,256]},"mask":{"min":[1,1,1,8],"opt":[4,1,1,128],"max":[8,1,1,512]}},p.open("w"))' \
    "${output}/profile.json"
  gpu_run "${BASELINE_WORKER}" "${corpus}" "${STATE_ROOT}" \
    python3 -m upgrade_guard.worker.build_engine \
      --model /corpus/models/tiny-transformer-fp32.onnx \
      --profile /output/aa/profile.json \
      --engine /output/aa/engine.plan \
      --inspector /output/aa/inspector.json \
      --timing-cache /output/aa/timing.cache \
      --result /output/aa/build.json
  trtexec_path="$("${UV[@]}" run --frozen python -c \
    'import sys; from pathlib import Path; from upgrade_guard.contracts.environment import MatrixLock; m=MatrixLock.model_validate_json(Path(sys.argv[1]).read_text()); print(m.environments[0].probe.trtexec.path)' \
    "${MATRIX_LOCK}" \
  )"
  stream_option="$("${UV[@]}" run --frozen python -c \
    'import sys; from pathlib import Path; from upgrade_guard.contracts.environment import MatrixLock; m=MatrixLock.model_validate_json(Path(sys.argv[1]).read_text()); o=m.environments[0].probe.trtexec.options; assert "--infStreams" in o or "--streams" in o; print("--infStreams=1" if "--infStreams" in o else "--streams=1")' \
    "${MATRIX_LOCK}" \
  )"
  local accepted=0
  local attempt=0
  while [[ ${accepted} -lt 20 && ${attempt} -lt 60 ]]; do
    local attempt_id
    attempt_id="$(printf '%02d' "${attempt}")"
    local attempt_root="${output}/attempt-${attempt_id}"
    mkdir -p "${attempt_root}"
    local pair_valid=1
    for side in a b; do
      local idle_valid=0
      for _ in $(seq 1 60); do
        if "${UV[@]}" run --frozen python scripts/hardware_validity.py capture \
          --specification "${STATE_ROOT}/full.yaml" --gpu "${GPU_UUID}" \
          --output "${attempt_root}/${side}-idle.json"; then
          idle_valid=1
          break
        fi
        sleep 1
      done
      if [[ ${idle_valid} -ne 1 ]]; then
        pair_valid=0
        break
      fi
      gpu_run "${BASELINE_WORKER}" "${corpus}" "${STATE_ROOT}" \
        "${trtexec_path}" \
          --loadEngine=/output/aa/engine.plan \
          --shapes=mask:1x1x1x128,tokens:1x128x256 \
          "--exportTimes=/output/aa/attempt-${attempt_id}/${side}-precondition.json" \
          --warmUp=500 --duration=1 "${stream_option}" --noDataTransfers
      if ! "${UV[@]}" run --frozen python scripts/hardware_validity.py capture \
        --specification "${STATE_ROOT}/full.yaml" --gpu "${GPU_UUID}" --loaded \
        --output "${attempt_root}/${side}-before.json"; then
        pair_valid=0
        break
      fi
      gpu_run "${BASELINE_WORKER}" "${corpus}" "${STATE_ROOT}" \
        "${trtexec_path}" \
          --loadEngine=/output/aa/engine.plan \
          --shapes=mask:1x1x1x128,tokens:1x128x256 \
          "--exportTimes=/output/aa/attempt-${attempt_id}/${side}.json" \
          --warmUp=500 --duration=1 "${stream_option}" --noDataTransfers
      "${UV[@]}" run --frozen python scripts/hardware_validity.py capture \
        --specification "${STATE_ROOT}/full.yaml" --gpu "${GPU_UUID}" --loaded \
        --output "${attempt_root}/${side}-after.json" || pair_valid=0
      "${UV[@]}" run --frozen python scripts/hardware_validity.py transition \
        --specification "${STATE_ROOT}/full.yaml" \
        --before "${attempt_root}/${side}-before.json" \
        --after "${attempt_root}/${side}-after.json" \
        --output "${attempt_root}/${side}-validity.json" || pair_valid=0
      if [[ ${pair_valid} -ne 1 ]]; then
        break
      fi
    done
    if [[ ${pair_valid} -eq 1 ]]; then
      "${UV[@]}" run --frozen python -c \
        'import json,sys; from pathlib import Path; root=Path(sys.argv[1]); values=[json.loads((root/f"{side}-validity.json").read_text()) for side in ("a","b")]; assert all(v["status"]=="passed" for v in values); (root/"validity.json").write_text(json.dumps({"status":"passed","blocks":values},indent=2,sort_keys=True)+"\n")' \
        "${attempt_root}"
      mv "${attempt_root}" "${output}/pair-$(printf '%02d' "${accepted}")"
      accepted=$((accepted + 1))
    else
      mv "${attempt_root}" "${output}/rejected-attempt-${attempt_id}"
    fi
    attempt=$((attempt + 1))
  done
  [[ ${accepted} -eq 20 ]]
  "${UV[@]}" run --frozen python scripts/validate_aa.py \
    --pairs "${output}" --output "${output}/validation.json"
}

materialize_fault_inputs() {
  cd "${PROJECT_ROOT}"
  local output="${STATE_ROOT}/faults"
  mkdir -p "${output}"
  if [[ ! -f "${output}/inputs.json" ]]; then
    "${UV[@]}" run --frozen python scripts/materialize_gpu_fault_inputs.py "${output}"
  fi
}

run_gpu_faults() {
  load_worker_identities
  local output="${STATE_ROOT}/faults"
  local build="${STATE_ROOT}/plugin/candidate/build"
  local samples="${output}/gpu-fault-samples.jsonl"
  : > "${samples}"
  for _ in $(seq 1 20); do
    gpu_run "${CANDIDATE_WORKER}" "${PROJECT_ROOT}" "${STATE_ROOT}" \
      /output/plugin/candidate/build/upgrade_guard_gpu_faults >> "${samples}"
  done
  "${UV[@]}" run --frozen python scripts/validate_seeded_gpu_faults.py \
    --samples "${samples}" --output "${output}/validation.json"
  gpu_run "${CANDIDATE_WORKER}" "${PROJECT_ROOT}" "${STATE_ROOT}" \
    /output/plugin/candidate/build/upgrade_guard_serialization_fault \
    > "${output}/G6.json"
  "${UV[@]}" run --frozen python -c \
    'import json,sys; p=sys.argv[1]; v=json.load(open(p)); assert v["detected"] and v["control"]=="passed"' \
    "${output}/G6.json"

  mkdir -p "${output}/g1"
  write_plugin_profile "${output}/g1/profile.json"
  set +e
  gpu_run "${CANDIDATE_WORKER}" "${PLUGIN_CORPUS}" \
    "${STATE_ROOT}" python3 -m upgrade_guard.worker.build_engine \
      --model /corpus/residual-rmsnorm-fp32.onnx \
      --profile /output/faults/g1/profile.json \
      --engine /output/faults/g1/engine.plan \
      --inspector /output/faults/g1/inspector.json \
      --timing-cache /output/faults/g1/timing.cache \
      --result /output/faults/g1/build.json
  local g1_status=$?
  set -e
  if [[ ${g1_status} -eq 0 ]]; then
    printf 'Unsupported custom-domain graph unexpectedly parsed without its plugin.\n' >&2
    return 1
  fi
  "${UV[@]}" run --frozen python -c \
    'import json,sys; from pathlib import Path; from upgrade_guard.contracts.base import sha256_file; src,control,dst=map(Path,sys.argv[1:]); v=json.load(src.open()); c=json.load(control.open()); assert v["status"]=="failed" and "parser" in v["message"].lower(); assert c["status"]=="passed"; json.dump({"expected":"ONNX_PARSE_FAILED","detected":True,"control":"passed","control_build_sha256":sha256_file(control),"worker":v},dst.open("w"),indent=2)' \
    "${output}/g1/build.json" \
    "${STATE_ROOT}/plugin-runs/candidate/fp32/build.json" "${output}/G1.json"

  mkdir -p "${output}/g7"
  gpu_run "${CANDIDATE_WORKER}" "${CORE_CORPUS}" \
    "${STATE_ROOT}" python3 -m upgrade_guard.worker.run_correctness \
      --engine /output/core-run/candidate/fp32/engine-0.plan \
      --input tokens=/corpus/inputs/tiny-transformer-fp32/b8_s512/tokens.npy \
      --input mask=/corpus/inputs/tiny-transformer-fp32/b8_s512/mask.npy \
      --output /output/faults/g7/control-outputs \
      --result /output/faults/g7/control-correctness.json \
      --repetitions 20
  set +e
  gpu_run "${CANDIDATE_WORKER}" "${CORE_CORPUS}" \
    "${STATE_ROOT}" python3 -m upgrade_guard.worker.run_correctness \
      --engine /output/core-run/candidate/fp32/engine-0.plan \
      --input tokens=/output/faults/g7/tokens.npy \
      --input mask=/output/faults/g7/mask.npy \
      --output /output/faults/g7/outputs \
      --result /output/faults/g7/correctness.json \
      --repetitions 20
  local g7_status=$?
  set -e
  if [[ ${g7_status} -eq 0 ]]; then
    printf 'Out-of-profile input unexpectedly executed.\n' >&2
    return 1
  fi
  "${UV[@]}" run --frozen python -c \
    'import json,sys; from pathlib import Path; from upgrade_guard.contracts.base import sha256_file; src,control,dst=map(Path,sys.argv[1:]); v=json.load(src.open()); c=json.load(control.open()); assert v["status"]=="failed" and "shape was rejected" in v["message"]; assert c["status"]=="passed"; json.dump({"expected":"PROFILE_REJECTED","detected":True,"control":"passed","control_run_sha256":sha256_file(control),"worker":v},dst.open("w"),indent=2)' \
    "${output}/g7/correctness.json" "${output}/g7/control-correctness.json" \
    "${output}/G7.json"
  [[ -d "${build}" ]]
}

run_reduction_replay() {
  local root="${STATE_ROOT}/reductions"
  "${UV[@]}" run --frozen python scripts/create_remote_reproductions.py \
    --state "${STATE_ROOT}" --project "${PROJECT_ROOT}" \
    --plugin-corpus "${PLUGIN_CORPUS}"
  for seed in G2 G7; do
    local bundle="${root}/${seed}-clean-bundle"
    local output="${root}/${seed}-clean-replay-output"
    "${UV[@]}" run --frozen upgrade-guard reproduce run "${bundle}" \
      --out "${output}" --trust-source-code --json \
      > "${root}/${seed}-cli-result.json"
  done
  "${UV[@]}" run --frozen python -c \
    'import json,sys; from pathlib import Path; root,out=map(Path,sys.argv[1:]); prepared=json.load((root/"prepared.json").open()); expected={"G2":"NUMERICAL_REGRESSION","G7":"PROFILE_REJECTED"}; replays={};
for seed,code in expected.items():
 value=json.load((root/f"{seed}-clean-replay-output/replay-result.json").open()); cli=json.load((root/f"{seed}-cli-result.json").open()); assert value==cli and value["status"]=="passed" and value["expected_failure_code"]==code; replays[seed]=value
prepared["status"]="passed"; prepared["clean_replays"]=replays; json.dump(prepared,out.open("w"),indent=2,sort_keys=True)' \
    "${root}" "${root}/validation.json"
}

run_memory_seed() {
  load_worker_identities
  local output="${STATE_ROOT}/memory-seed"
  local corpus="${PLUGIN_CORPUS}"
  local plugin="/output/plugin/candidate/build/libupgrade_guard_residual_rmsnorm.so"
  mkdir -p "${output}/control" "${output}/seeded"
  write_plugin_profile "${output}/profile.json"
  for kind in control seeded; do
    local model=/corpus/residual-rmsnorm-fp32.onnx
    if [[ "${kind}" == seeded ]]; then
      model=/output/faults/residual-rmsnorm-workspace-seed.onnx
    fi
    for build_index in 0 1 2; do
      gpu_run "${CANDIDATE_WORKER}" "${corpus}" "${STATE_ROOT}" \
        python3 -m upgrade_guard.worker.build_engine \
          --model "${model}" \
          --profile /output/memory-seed/profile.json \
          --engine "/output/memory-seed/${kind}/engine-${build_index}.plan" \
          --inspector "/output/memory-seed/${kind}/inspector-${build_index}.json" \
          --timing-cache "/output/memory-seed/${kind}/timing.cache" \
          --result "/output/memory-seed/${kind}/build-${build_index}.json" \
          --plugin "${plugin}"
    done
  done
  "${UV[@]}" run --frozen python scripts/validate_memory_seed.py \
    --control "${output}/control" --seeded "${output}/seeded" \
    --output "${output}/validation.json"
}

run_sanitizers() {
  load_worker_identities
  local output="${STATE_ROOT}/plugin/candidate"
  local executable="/output/build/upgrade_guard_kernel_tests"
  for tool in memcheck racecheck initcheck synccheck; do
    gpu_run "${CANDIDATE_WORKER}" "${PROJECT_ROOT}" "${output}" \
      compute-sanitizer --tool "${tool}" --error-exitcode 91 "${executable}" \
      > "${output}/sanitizer-${tool}-control.log" 2>&1
    grep -E 'ERROR SUMMARY: 0 errors|ERROR SUMMARY: 0 error' \
      "${output}/sanitizer-${tool}-control.log" >/dev/null
  done
  set +e
  gpu_run "${CANDIDATE_WORKER}" "${PROJECT_ROOT}" "${output}" \
    compute-sanitizer --tool memcheck --error-exitcode 86 \
      /output/build/upgrade_guard_tail_oob_fault \
      > "${output}/sanitizer-tail-oob.log" 2>&1
  local fault_status=$?
  set -e
  if [[ ${fault_status} -ne 86 ]]; then
    printf 'Quarantined tail defect did not fail with the expected sanitizer code.\n' >&2
    return 1
  fi
  grep -E 'Invalid __global__ (read|write)|out of bounds' \
    "${output}/sanitizer-tail-oob.log" >/dev/null
  grep -E 'ERROR SUMMARY: [1-9][0-9]* error' \
    "${output}/sanitizer-tail-oob.log" >/dev/null
  "${UV[@]}" run --frozen python -c \
    'import json,sys; from pathlib import Path; log=Path(sys.argv[1]); out=Path(sys.argv[2]); json.dump({"expected":"SANITIZER_FAILURE","observed_exit_code":86,"diagnostic":"out_of_bounds_global_access","diagnostic_log_sha256":__import__("upgrade_guard.contracts.base",fromlist=["sha256_file"]).sha256_file(log),"control":"passed"},out.open("w"),indent=2,sort_keys=True)' \
    "${output}/sanitizer-tail-oob.log" "${output}/sanitizer-seed.json"
}

run_profiles() {
  load_worker_identities
  local output="${STATE_ROOT}/plugin/candidate"
  gpu_run "${CANDIDATE_WORKER}" "${PROJECT_ROOT}" "${output}" \
    nsys --version
  gpu_run "${CANDIDATE_WORKER}" "${PROJECT_ROOT}" "${output}" \
    ncu --version
  gpu_run "${CANDIDATE_WORKER}" "${PROJECT_ROOT}" "${output}" \
    nsys profile --help > "${output}/nsys-profile-help.txt"
  grep -F -- '--nvtx-capture' "${output}/nsys-profile-help.txt" >/dev/null
  gpu_run "${CANDIDATE_WORKER}" "${PROJECT_ROOT}" "${output}" \
    ncu --help > "${output}/ncu-help.txt"
  grep -F -- '--kernel-name-base' "${output}/ncu-help.txt" >/dev/null
  gpu_run "${CANDIDATE_WORKER}" "${PROJECT_ROOT}" "${output}" \
    nsys profile --trace=cuda,nvtx --sample=none --capture-range=nvtx \
      --nvtx-capture=residual_rmsnorm_optimized@upgrade_guard \
      --capture-range-end=stop --force-overwrite=true \
      --output=/output/residual-rmsnorm-timeline \
      /output/build/upgrade_guard_kernel_benchmark --profile-only
  gpu_run "${CANDIDATE_WORKER}" "${PROJECT_ROOT}" "${output}" \
    nsys export --type=sqlite --force-overwrite=true \
      --output=/output/residual-rmsnorm-timeline.sqlite \
      /output/residual-rmsnorm-timeline.nsys-rep
  gpu_run "${CANDIDATE_WORKER}" "${PROJECT_ROOT}" "${output}" \
    nsys stats --report cuda_gpu_kern_sum --format csv \
      /output/residual-rmsnorm-timeline.nsys-rep \
      > "${output}/nsys-kernel-summary.csv"
  gpu_run "${CANDIDATE_WORKER}" "${PROJECT_ROOT}" "${output}" \
    ncu --target-processes all --kernel-name-base demangled \
      --kernel-name regex:residualRmsNormFloat4 \
      --launch-count 1 --section SpeedOfLight --section MemoryWorkloadAnalysis \
      --section LaunchStats --section Occupancy --force-overwrite \
      --export /output/residual-rmsnorm-kernel \
      /output/build/upgrade_guard_kernel_benchmark --profile-only
  gpu_run "${CANDIDATE_WORKER}" "${PROJECT_ROOT}" "${output}" \
    ncu --import /output/residual-rmsnorm-kernel.ncu-rep --csv --page details \
      > "${output}/ncu-kernel-summary.csv"
}

generate_sboms() {
  load_worker_identities
  local output="${STATE_ROOT}/sbom"
  mkdir -p "${output}"
  "${UV[@]}" run --frozen python scripts/generate_host_sbom.py \
    --lock uv.lock --output "${output}/host.spdx.json"
  gpu_run "${BASELINE_WORKER}" "${PROJECT_ROOT}" "${output}" \
    python3 /opt/upgrade-guard/scripts/generate_worker_sbom.py \
      --image "${BASELINE_WORKER}" --output /output/baseline.spdx.json
  gpu_run "${CANDIDATE_WORKER}" "${PROJECT_ROOT}" "${output}" \
    python3 /opt/upgrade-guard/scripts/generate_worker_sbom.py \
      --image "${CANDIDATE_WORKER}" --output /output/candidate.spdx.json
}

audit_dependencies() {
  local output="${STATE_ROOT}/supply-chain"
  mkdir -p "${output}"
  "${UV[@]}" export --frozen --no-dev --no-emit-project \
    --format requirements.txt --output-file "${output}/requirements.txt"
  "${UV[@]}" tool run --from pip-audit==2.10.1 pip-audit \
    --requirement "${output}/requirements.txt" \
    --disable-pip --require-hashes \
    --format json --output "${output}/pip-audit.json" \
    --progress-spinner off
  "${UV[@]}" tool run --from pip-audit==2.10.1 pip-audit \
    --requirement containers/requirements-worker.txt \
    --disable-pip --require-hashes \
    --format json --output "${output}/worker-pip-audit.json" \
    --progress-spinner off
}

finalize() {
  cd "${PROJECT_ROOT}"
  "${UV[@]}" run --frozen python scripts/generate_remote_evidence.py \
    --state "${STATE_ROOT}" --output "${STATE_ROOT}/evidence.json"
  docker container rm --force "${REGISTRY_NAME}" >/dev/null || true
  docker volume rm "${REGISTRY_VOLUME}" >/dev/null || true
  printf 'COMPLETE evidence=%s\n' "${STATE_ROOT}/evidence.json"
}

cd "${PROJECT_ROOT}"
select_uv
invocation_guard
run_step preflight preflight
run_step cpu-verify cpu_verify
CURRENT_STEP=local-registry
start_local_registry 2>&1 | tee "${LOG_ROOT}/local-registry.log"
run_step worker-images build_workers
ensure_worker_registry
preserve_stale_matrix
run_step matrix-lock lock_matrix
if [[ "${SANITIZER_ONLY}" == "1" ]]; then
  run_step corpus-materialization materialize_sanitizer_corpora
else
  run_step corpus-materialization materialize_corpora
fi
if [[ "${SMOKE_ONLY}" == "1" ]]; then
  run_step plugin-compile-test compile_plugins
  run_step gpu-smoke run_gpu_smoke
  exit 0
fi
if [[ "${SANITIZER_ONLY}" == "1" ]]; then
  run_step plugin-compile-test compile_plugins
  run_step sanitizers run_sanitizers
  exit 0
fi
run_step aa-pilot run_aa_pilot
run_step core-qualification run_core_qualification
run_step plugin-compile-test compile_plugins
run_step plugin-benchmark run_plugin_benchmark
run_step plugin-matrix run_plugin_matrix
run_step mobilenet-matrix run_mobilenet_matrix
run_step fault-inputs materialize_fault_inputs
run_step gpu-faults run_gpu_faults
run_step reduction-replay run_reduction_replay
run_step memory-seed run_memory_seed
run_step sanitizers run_sanitizers
run_step profiles run_profiles
run_step sboms generate_sboms
run_step dependency-audit audit_dependencies
run_step final-evidence finalize
