#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${PROJECT_ROOT}/scripts/bounded_executor.sh"
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
CORPUS_STORE="${PROJECT_ROOT}/.upgrade-guard/corpora/by-id"
CORE_CORPUS=""
PLUGIN_CORPUS=""
MOBILENET_CORPUS=""
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

if [[ "${UG_QUALIFICATION_LOCK_HELD:-0}" != "1" ]]; then
  LOCK_ROOT="${STATE_BASE}/locks"
  mkdir -p "${LOCK_ROOT}"
  exec python3 "${PROJECT_ROOT}/scripts/run_locked.py" \
    --lock "${LOCK_ROOT}/runner.lock" \
    --source "${SOURCE_ID}" -- bash "${BASH_SOURCE[0]}" "$@"
fi

mkdir -p "${DONE_ROOT}" "${LOG_ROOT}"

failure_report() {
  local exit_code=$?
  if command -v docker >/dev/null 2>&1 \
    && bounded_run cleanup docker container inspect "${REGISTRY_NAME}" >/dev/null 2>&1; then
    bounded_run cleanup docker container rm --force "${REGISTRY_NAME}" \
      >/dev/null 2>&1 || true
  fi
  if [[ ${exit_code} -ne 0 ]]; then
    local classification=auto
    local diagnostic_arguments=(
      --state "${STATE_ROOT}" --step "${CURRENT_STEP}" --exit-code "${exit_code}"
      --classification "${classification}" --source "${SOURCE_ID}" --mode "${RUN_MODE}"
    )
    if [[ -f "${LOG_ROOT}/${CURRENT_STEP}.log" ]] \
      && grep -qi 'no space left on device' "${LOG_ROOT}/${CURRENT_STEP}.log"; then
      classification=enospc
      diagnostic_arguments=(
        --state "${STATE_ROOT}" --step "${CURRENT_STEP}" --exit-code "${exit_code}"
        --classification "${classification}" --source "${SOURCE_ID}" --mode "${RUN_MODE}"
      )
    fi
    if [[ -n "${GPU_UUID}" ]]; then
      diagnostic_arguments+=(--gpu "${GPU_UUID}")
    fi
    python3 scripts/write_failure_diagnostic.py "${diagnostic_arguments[@]}" \
      >/dev/null 2>&1 || true
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
  if bounded_run quick "${UV[@]}" run --frozen python scripts/qualification_state.py verify \
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
  bounded_run quick "${UV[@]}" run --frozen python scripts/qualification_state.py record \
    --state "${STATE_ROOT}" --project "${PROJECT_ROOT}" --step "${name}" \
    --source "${SOURCE_ID}" --gpu "${GPU_UUID}" --mode "${RUN_MODE}"
  if [[ "${THROUGH_STEP}" == "${name}" ]]; then
    exit 0
  fi
}

run_always_step() {
  local name=$1
  shift
  CURRENT_STEP="${name}"
  printf 'RUN required invocation step: %s\n' "${name}"
  "$@"
  printf 'validated step=%s source=%s\n' "${name}" "${SOURCE_ID}" \
    > "${LOG_ROOT}/${name}.log"
  bounded_run quick "${UV[@]}" run --frozen python scripts/qualification_state.py record \
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
  command -v timeout
  local docker_platform
  bounded_run quick docker version >/dev/null
  docker_platform="$(bounded_run quick docker info --format '{{.OSType}}/{{.Architecture}}')"
  [[ "${docker_platform}" == "linux/x86_64" || "${docker_platform}" == "linux/amd64" ]]
  if [[ -n "${UG_EXPECTED_GPU_UUID:-}" ]]; then
    GPU_UUID="$(bounded_run preflight nvidia-smi --id="${UG_EXPECTED_GPU_UUID}" \
      --query-gpu=uuid --format=csv,noheader | tr -d '[:space:]')"
    [[ "${GPU_UUID}" == "${UG_EXPECTED_GPU_UUID}" ]]
  else
    GPU_UUID="$(bounded_run preflight nvidia-smi --id="${GPU_INDEX}" \
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
  if command -v uv >/dev/null 2>&1 \
    && [[ "$(bounded_run quick uv --version)" == uv\ 0.11.23* ]]; then
    UV=(uv)
    return
  fi
  local bootstrap="${STATE_BASE}/uv-bootstrap"
  if [[ ! -x "${bootstrap}/bin/uv" ]]; then
    bounded_run preflight python3 -m venv "${bootstrap}"
    bounded_run network "${bootstrap}/bin/python" -m pip install \
      --disable-pip-version-check "uv==0.11.23"
  fi
  UV=("${bootstrap}/bin/uv")
}

preflight() {
  cd "${PROJECT_ROOT}"
  git rev-parse HEAD > "${STATE_ROOT}/source.commit"
  bounded_run quick docker version
  bounded_run preflight nvidia-smi
  printf '%s\n' "${GPU_UUID}" > "${STATE_ROOT}/gpu.uuid"
  bounded_run preflight nvidia-smi --id="${GPU_UUID}" \
    --query-gpu=name,uuid,compute_cap,memory.total,driver_version,vbios_version,power.limit \
    --format=csv,noheader > "${STATE_ROOT}/gpu-preflight.csv"
}

cpu_verify() {
  cd "${PROJECT_ROOT}"
  bounded_run network "${UV[@]}" sync --frozen
  bounded_run quick "${UV[@]}" run --frozen python scripts/generate_schemas.py
  git diff --exit-code -- schemas
  bounded_run quick "${UV[@]}" run --frozen python scripts/check_repository_docs.py
  bounded_run quick "${UV[@]}" run --frozen ruff check .
  bounded_run quick "${UV[@]}" run --frozen ruff format --check .
  bounded_run build "${UV[@]}" run --frozen mypy
  bounded_run build "${UV[@]}" run --frozen pytest \
    --cov=upgrade_guard --cov-report=term-missing
  git diff --exit-code
  [[ -z "$(git status --porcelain --untracked-files=normal)" ]]
}

gpu_runtime_preflight() {
  ensure_exact_docker_image "${REGISTRY_IMAGE}"
  bounded_run preflight "${UV[@]}" run --frozen python scripts/check_docker_gpu_runtime.py \
    --gpu "${GPU_UUID}" --image "${REGISTRY_IMAGE}" \
    --output "${STATE_ROOT}/gpu-runtime-preflight.json"
}

start_local_registry() {
  ensure_exact_docker_image "${REGISTRY_IMAGE}"
  mapfile -t port_owners < <(
    bounded_run quick docker container ls --filter publish=5500 --format '{{.Names}}'
  )
  local port_owner
  for port_owner in "${port_owners[@]}"; do
    if [[ "${port_owner}" == "${REGISTRY_NAME}" ]]; then
      continue
    fi
    local owner_label
    owner_label="$(bounded_run quick docker container inspect --format \
      '{{index .Config.Labels "com.udayarora.upgradeguard.owner"}}' "${port_owner}")"
    if [[ "${owner_label}" != "tensorrt-upgrade-guard" ]]; then
      printf 'Registry port 5500 belongs to an unrelated container: %s\n' "${port_owner}" >&2
      return 4
    fi
    bounded_run cleanup docker container rm --force "${port_owner}" >/dev/null
    printf 'Removed stale project registry container %s; its volume was retained.\n' "${port_owner}"
  done
  bounded_run quick docker volume create \
    --label com.udayarora.upgradeguard.owner=tensorrt-upgrade-guard \
    --label "com.udayarora.upgradeguard.source=${SOURCE_ID}" \
    "${REGISTRY_VOLUME}" >/dev/null
  bounded_run quick docker volume inspect "${REGISTRY_VOLUME}" \
    | REGISTRY_NAME_VALUE="${REGISTRY_NAME}" REGISTRY_VOLUME_VALUE="${REGISTRY_VOLUME}" \
      SOURCE_ID_VALUE="${SOURCE_ID}" python3 -c \
      'import json,os,sys; v=json.load(sys.stdin); assert len(v)==1; x=v[0]; labels=x.get("Labels") or {}; assert x["Name"]==os.environ["REGISTRY_VOLUME_VALUE"]; assert labels.get("com.udayarora.upgradeguard.owner")=="tensorrt-upgrade-guard"; assert labels.get("com.udayarora.upgradeguard.source")==os.environ["SOURCE_ID_VALUE"]'
  if bounded_run quick docker container inspect "${REGISTRY_NAME}" >/dev/null 2>&1; then
    bounded_run quick docker container inspect "${REGISTRY_NAME}" \
      | REGISTRY_NAME_VALUE="${REGISTRY_NAME}" REGISTRY_VOLUME_VALUE="${REGISTRY_VOLUME}" \
        SOURCE_ID_VALUE="${SOURCE_ID}" REGISTRY_IMAGE_VALUE="${REGISTRY_IMAGE}" \
        python3 -c \
      'import json,os,sys; v=json.load(sys.stdin); assert len(v)==1; x=v[0]; c=x["Config"]; labels=c.get("Labels") or {}; assert x["Name"].removeprefix("/")==os.environ["REGISTRY_NAME_VALUE"]; assert c["Image"]==os.environ["REGISTRY_IMAGE_VALUE"]; assert labels.get("com.udayarora.upgradeguard.owner")=="tensorrt-upgrade-guard"; assert labels.get("com.udayarora.upgradeguard.source")==os.environ["SOURCE_ID_VALUE"]; assert any(m.get("Type")=="volume" and m.get("Name")==os.environ["REGISTRY_VOLUME_VALUE"] and m.get("Destination")=="/var/lib/registry" for m in x["Mounts"]); p=x["HostConfig"]["PortBindings"]["5000/tcp"]; assert p==[{"HostIp":"127.0.0.1","HostPort":"5500"}]'
    if [[ "$(bounded_run quick docker container inspect --format '{{.State.Running}}' "${REGISTRY_NAME}")" != "true" ]]; then
      bounded_run quick docker container start "${REGISTRY_NAME}" >/dev/null
    fi
  else
    bounded_run quick docker run -d --name "${REGISTRY_NAME}" \
      --label com.udayarora.upgradeguard.owner=tensorrt-upgrade-guard \
      --label "com.udayarora.upgradeguard.source=${SOURCE_ID}" \
      -p "127.0.0.1:5500:5000" \
      --mount "type=volume,src=${REGISTRY_VOLUME},dst=/var/lib/registry" \
      "${REGISTRY_IMAGE}" >/dev/null
  fi
  for _ in $(seq 1 30); do
    if bounded_run quick curl --fail --silent --connect-timeout 3 --max-time 5 \
      "http://${REGISTRY_ADDRESS}/v2/" >/dev/null; then
      REGISTRY_IDENTITY_PATH="${STATE_ROOT}/registry-identity.json" \
        REGISTRY_NAME_VALUE="${REGISTRY_NAME}" REGISTRY_VOLUME_VALUE="${REGISTRY_VOLUME}" \
        REGISTRY_IMAGE_VALUE="${REGISTRY_IMAGE}" REGISTRY_ADDRESS_VALUE="${REGISTRY_ADDRESS}" \
        SOURCE_ID_VALUE="${SOURCE_ID}" python3 -c \
        'import json,os,tempfile; from pathlib import Path; out=Path(os.environ["REGISTRY_IDENTITY_PATH"]); value={"schema_version":"upgradeguard.dev/local-registry/v1","source_git_commit":os.environ["SOURCE_ID_VALUE"],"image":os.environ["REGISTRY_IMAGE_VALUE"],"container":os.environ["REGISTRY_NAME_VALUE"],"volume":os.environ["REGISTRY_VOLUME_VALUE"],"address":os.environ["REGISTRY_ADDRESS_VALUE"],"volume_retained":True}; out.parent.mkdir(parents=True,exist_ok=True); f=tempfile.NamedTemporaryFile("w",dir=out.parent,prefix=f".{out.name}.",delete=False); json.dump(value,f,indent=2,sort_keys=True); f.write("\n"); f.flush(); os.fsync(f.fileno()); f.close(); Path(f.name).replace(out)'
      return
    fi
    sleep 1
  done
  printf 'Local registry did not become ready.\n' >&2
  return 1
}

capacity_preflight() {
  local output="${STATE_ROOT}/capacity"
  local docker_filesystem_id
  mkdir -p "${output}"
  bounded_run quick docker exec "${REGISTRY_NAME}" df -Pk /var/lib/registry \
    > "${output}/docker-blocks.df"
  bounded_run quick docker exec "${REGISTRY_NAME}" df -Pi /var/lib/registry \
    > "${output}/docker-inodes.df"
  docker_filesystem_id="$(
    bounded_run quick docker exec "${REGISTRY_NAME}" stat -c %d /var/lib/registry
  )"
  bounded_run quick python3 scripts/check_capacity.py \
    --workspace "${PROJECT_ROOT}" --output "${output}/capacity.json" \
    --docker-blocks-df "${output}/docker-blocks.df" \
    --docker-inodes-df "${output}/docker-inodes.df" \
    --docker-filesystem-id "${docker_filesystem_id}"
}

build_workers() {
  cd "${PROJECT_ROOT}"
  ensure_exact_docker_image "${BASELINE_BASE}"
  ensure_exact_docker_image "${CANDIDATE_BASE}"
  bounded_run build docker build --pull=false \
    --build-arg "BASE_IMAGE=${BASELINE_BASE}" \
    --build-arg "BASE_MANIFEST_DIGEST=sha256:2a5a0a9a32ec5ddc1c384c15ddcf3b89ddc4f8647e7ee7ae708d844210183a1e" \
    --tag "${REGISTRY_ADDRESS}/upgrade-guard/worker:baseline" \
    --file containers/Dockerfile.worker .
  bounded_run build docker build --pull=false \
    --build-arg "BASE_IMAGE=${CANDIDATE_BASE}" \
    --build-arg "BASE_MANIFEST_DIGEST=sha256:b82db1abc23750ab0069abc99bbe4ea29138dbdc23ea39861199e2346638b48a" \
    --tag "${REGISTRY_ADDRESS}/upgrade-guard/worker:candidate" \
    --file containers/Dockerfile.worker .
  bounded_run network docker push "${REGISTRY_ADDRESS}/upgrade-guard/worker:baseline"
  bounded_run network docker push "${REGISTRY_ADDRESS}/upgrade-guard/worker:candidate"
  bounded_run quick "${UV[@]}" run --frozen python \
    scripts/qualification_state.py capture-workers \
    --output "${STATE_ROOT}/worker-images.json" \
    "${REGISTRY_ADDRESS}/upgrade-guard/worker:baseline" \
    "${REGISTRY_ADDRESS}/upgrade-guard/worker:candidate"
}

ensure_worker_registry() {
  local accept='application/vnd.oci.image.manifest.v1+json, application/vnd.docker.distribution.manifest.v2+json'
  if bounded_run quick curl --fail --silent --head --connect-timeout 3 --max-time 5 \
    --header "Accept: ${accept}" \
    "http://${REGISTRY_ADDRESS}/v2/upgrade-guard/worker/manifests/baseline" >/dev/null \
    && bounded_run quick curl --fail --silent --head --connect-timeout 3 --max-time 5 \
      --header "Accept: ${accept}" \
      "http://${REGISTRY_ADDRESS}/v2/upgrade-guard/worker/manifests/candidate" >/dev/null; then
    return
  fi
  printf 'Worker registry content is absent; rebuilding exact workers.\n'
  build_workers 2>&1 | tee -a "${LOG_ROOT}/worker-images.log"
  bounded_run quick "${UV[@]}" run --frozen python scripts/qualification_state.py record \
    --state "${STATE_ROOT}" --project "${PROJECT_ROOT}" --step worker-images \
    --source "${SOURCE_ID}" --gpu "${GPU_UUID}" --mode "${RUN_MODE}"
}

preserve_stale_matrix() {
  if [[ ! -f "${MATRIX_LOCK}" ]]; then
    return
  fi
  if bounded_run quick "${UV[@]}" run --frozen python \
    scripts/qualification_state.py verify-worker-lock \
    --workers "${STATE_ROOT}/worker-images.json" --matrix "${MATRIX_LOCK}"; then
    return
  fi
  local preserved="${STATE_ROOT}/stale/$(date -u +%Y%m%dT%H%M%S.%NZ)-matrix-lock"
  mkdir -p "${preserved}/done" "${preserved}/logs"
  local name
  for name in matrix.lock.json matrix.yaml full.yaml; do
    if [[ -e "${STATE_ROOT}/${name}" ]]; then
      mv "${STATE_ROOT}/${name}" "${preserved}/${name}"
    fi
  done
  if [[ -e "${DONE_ROOT}/matrix-lock.json" ]]; then
    mv "${DONE_ROOT}/matrix-lock.json" "${preserved}/done/matrix-lock.json"
  fi
  if [[ -e "${LOG_ROOT}/matrix-lock.log" ]]; then
    mv "${LOG_ROOT}/matrix-lock.log" "${preserved}/logs/matrix-lock.log"
  fi
  printf 'Preserved stale matrix state under %s\n' "${preserved}"
}

reconcile_state() {
  bounded_run quick "${UV[@]}" run --frozen python scripts/qualification_state.py reconcile \
    --state "${STATE_ROOT}" --project "${PROJECT_ROOT}" \
    --source "${SOURCE_ID}" --gpu "${GPU_UUID}" --mode "${RUN_MODE}"
}

lock_matrix() {
  cd "${PROJECT_ROOT}"
  local gpu_uuid
  gpu_uuid="$(<"${STATE_ROOT}/gpu.uuid")"
  MATRIX_PATH="${STATE_ROOT}/matrix.yaml" GPU_UUID_VALUE="${gpu_uuid}" \
    bounded_run quick "${UV[@]}" run --frozen python -c \
    'import os,yaml; from pathlib import Path; p=Path("matrices/examples/controlled-minor.yaml"); v=yaml.safe_load(p.read_text()); v["gpu_uuid"]=os.environ["GPU_UUID_VALUE"]; Path(os.environ["MATRIX_PATH"]).write_text(yaml.safe_dump(v,sort_keys=False))'
  if [[ -f "${MATRIX_LOCK}" ]]; then
    bounded_run quick "${UV[@]}" run --frozen python -c \
      'import sys; from pathlib import Path; from upgrade_guard.contracts.environment import MatrixLock; p=Path(sys.argv[1]); m=MatrixLock.model_validate_json(p.read_text()); assert m.lock_sha256 == m.computed_sha256()' \
      "${MATRIX_LOCK}"
  else
    bounded_run gpu "${UV[@]}" run --frozen upgrade-guard matrix lock \
      "${STATE_ROOT}/matrix.yaml" \
      --out "${MATRIX_LOCK}" --json
  fi
  local core_corpus_path
  core_corpus_path="$(expected_corpus_path core)"
  QUALIFICATION_PATH="${STATE_ROOT}/full.yaml" GPU_UUID_VALUE="${gpu_uuid}" \
    LOCK_PATH="${MATRIX_LOCK}" CORE_CORPUS_PATH="${core_corpus_path}" \
    bounded_run quick "${UV[@]}" run --frozen python -c \
    'import os,yaml; from pathlib import Path; p=Path("qualification/full.yaml"); v=yaml.safe_load(p.read_text()); v["hardware_validity"]["selected_gpu_uuid"]=os.environ["GPU_UUID_VALUE"]; v["environment_lock"]=os.environ["LOCK_PATH"]; v["corpus_root"]=Path(os.environ["CORE_CORPUS_PATH"]).relative_to(Path.cwd()).as_posix(); Path(os.environ["QUALIFICATION_PATH"]).write_text(yaml.safe_dump(v,sort_keys=False))'
}

expected_corpus_path() {
  local kind=$1
  local identity
  identity="$(bounded_run quick "${UV[@]}" run --frozen python \
    scripts/corpus_store.py identity \
    --project "${PROJECT_ROOT}" --kind "${kind}" \
    | bounded_run quick "${UV[@]}" run --frozen python -c \
      'import json,sys; print(json.load(sys.stdin)["materializer_sha256"].removeprefix("sha256:"))')"
  printf '%s\n' "${CORPUS_STORE}/${kind}/${identity}"
}

materialize_corpora() {
  cd "${PROJECT_ROOT}"
  CORE_CORPUS="$(materialize_corpus_content_addressed core materialize_core_corpus)"
  PLUGIN_CORPUS="$(materialize_corpus_content_addressed plugin materialize_plugin_corpus)"
  MOBILENET_CORPUS="$(materialize_corpus_content_addressed mobilenet materialize_mobilenet_corpus)"
  write_corpus_index core plugin mobilenet
}

materialize_bounded_corpora() {
  cd "${PROJECT_ROOT}"
  CORE_CORPUS="$(materialize_corpus_content_addressed core materialize_core_corpus)"
  PLUGIN_CORPUS="$(materialize_corpus_content_addressed plugin materialize_plugin_corpus)"
  write_corpus_index core plugin
}

materialize_corpus_content_addressed() {
  local kind=$1
  local producer=$2
  local destination
  destination="$(expected_corpus_path "${kind}")"
  if [[ -d "${destination}" ]] \
    && bounded_run quick "${UV[@]}" run --frozen python scripts/corpus_store.py verify \
      --project "${PROJECT_ROOT}" --kind "${kind}" --root "${destination}" \
    && bounded_run quick "${UV[@]}" run --frozen python scripts/qualification_state.py \
      verify-corpus "${destination}"; then
    printf '%s\n' "${destination}"
    return 0
  fi
  if [[ -e "${destination}" ]]; then
    printf 'Immutable corpus identity exists but does not verify: %s\n' "${destination}" >&2
    return 1
  fi
  mkdir -p "${CORPUS_STORE}/${kind}"
  local staging
  staging="$(mktemp -d "${CORPUS_STORE}/${kind}/.corpus-staging.XXXXXX")"
  local generated="${staging}/corpus"
  "${producer}" "${generated}" >&2
  bounded_run quick "${UV[@]}" run --frozen python scripts/corpus_store.py write-sidecar \
    --project "${PROJECT_ROOT}" --kind "${kind}" --root "${generated}"
  bounded_run quick "${UV[@]}" run --frozen python \
    scripts/qualification_state.py verify-corpus "${generated}"
  bounded_run quick "${UV[@]}" run --frozen python scripts/corpus_store.py publish \
    --project "${PROJECT_ROOT}" --kind "${kind}" \
    --staging "${generated}" --destination "${destination}" >/dev/null
  rmdir "${staging}" 2>/dev/null || true
  printf '%s\n' "${destination}"
}

write_corpus_index() {
  local arguments=()
  for kind in "$@"; do
    case "${kind}" in
      core) arguments+=(--corpus "core=${CORE_CORPUS}") ;;
      plugin) arguments+=(--corpus "plugin=${PLUGIN_CORPUS}") ;;
      mobilenet) arguments+=(--corpus "mobilenet=${MOBILENET_CORPUS}") ;;
      *) return 2 ;;
    esac
  done
  bounded_run quick "${UV[@]}" run --frozen python scripts/corpus_store.py write-index \
    --project "${PROJECT_ROOT}" --output "${STATE_ROOT}/corpora.json" \
    "${arguments[@]}"
}

load_corpus_identities() {
  local index="${STATE_ROOT}/corpora.json"
  [[ -f "${index}" ]]
  CORE_CORPUS="$(bounded_run quick "${UV[@]}" run --frozen python -c \
    'import json,sys; from pathlib import Path; v=json.load(open(sys.argv[1])); e=v["corpora"].get("core"); print((Path(sys.argv[2])/e["root"]).resolve()) if e else None' \
    "${index}" "${PROJECT_ROOT}")"
  PLUGIN_CORPUS="$(bounded_run quick "${UV[@]}" run --frozen python -c \
    'import json,sys; from pathlib import Path; v=json.load(open(sys.argv[1])); e=v["corpora"].get("plugin"); print((Path(sys.argv[2])/e["root"]).resolve()) if e else None' \
    "${index}" "${PROJECT_ROOT}")"
  if [[ "${RUN_MODE}" == "full" ]]; then
    MOBILENET_CORPUS="$(bounded_run quick "${UV[@]}" run --frozen python -c \
      'import json,sys; from pathlib import Path; v=json.load(open(sys.argv[1])); e=v["corpora"].get("mobilenet"); print((Path(sys.argv[2])/e["root"]).resolve()) if e else None' \
      "${index}" "${PROJECT_ROOT}")"
  fi
}

materialize_core_corpus() {
  bounded_run build "${UV[@]}" run --frozen upgrade-guard corpus materialize \
    corpus/registry.yaml \
    --out "$1" --json
}

materialize_plugin_corpus() {
  bounded_run build "${UV[@]}" run --frozen python scripts/materialize_plugin_corpus.py "$1"
}

materialize_mobilenet_corpus() {
  bounded_run network "${UV[@]}" run --frozen python \
    scripts/materialize_mobilenet_corpus.py "$1"
}

load_worker_identities() {
  load_corpus_identities
  mapfile -t WORKER_IMAGES < <(
    bounded_run quick "${UV[@]}" run --frozen python -c \
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
    if bounded_run quick "${UV[@]}" run --frozen python \
      scripts/hardware_validity.py capture \
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
  if bounded_run gpu docker run --rm --name "${container_name}" \
    --init --user "${user_id}:${group_id}" \
    --gpus "device=${GPU_UUID}" \
    --network none --read-only --cap-drop ALL \
    --security-opt no-new-privileges --pids-limit 512 --ipc private \
    --tmpfs /tmp:rw,noexec,nosuid,size=1073741824 \
    --tmpfs "${home}:rw,noexec,nosuid,nodev,size=1073741824,uid=${user_id},gid=${group_id},mode=0700" \
    --mount "type=bind,src=${PROJECT_ROOT},dst=/opt/upgrade-guard,readonly" \
    --mount "type=bind,src=${STATE_ROOT},dst=/state,readonly" \
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
    bounded_run cleanup docker container rm --force "${container_name}" \
      >/dev/null 2>&1 || true
    return "${status}"
  fi
}

run_core_qualification() {
  cd "${PROJECT_ROOT}"
  if [[ -f "${STATE_ROOT}/core-run/qualification-summary.json" ]]; then
    bounded_run quick "${UV[@]}" run --frozen upgrade-guard compare \
      "${STATE_ROOT}/core-run" --json
    return
  fi
  bounded_run build "${UV[@]}" run --frozen upgrade-guard qualify \
    "${STATE_ROOT}/full.yaml" \
    --out "${STATE_ROOT}/core-run" --json
  bounded_run quick "${UV[@]}" run --frozen upgrade-guard compare \
    "${STATE_ROOT}/core-run" --json
}

compile_plugins() {
  load_worker_identities
  local names=(baseline candidate)
  local images=("${BASELINE_WORKER}" "${CANDIDATE_WORKER}")
  for index in 0 1; do
    local output="${STATE_ROOT}/plugin-build/${names[${index}]}"
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

run_profiler_preflight() {
  load_worker_identities
  local output="${STATE_ROOT}/profiler-preflight"
  local executable=/state/plugin-build/candidate/build/upgrade_guard_kernel_benchmark
  mkdir -p "${output}"
  gpu_run "${CANDIDATE_WORKER}" "${PROJECT_ROOT}" "${output}" \
    ncu --version > "${output}/ncu-version.txt"
  gpu_run "${CANDIDATE_WORKER}" "${PROJECT_ROOT}" "${output}" \
    ncu --help > "${output}/ncu-help.txt"
  grep -F -- '--kernel-name-base' "${output}/ncu-help.txt" >/dev/null
  set +e
  gpu_run "${CANDIDATE_WORKER}" "${PROJECT_ROOT}" "${output}" \
    ncu --target-processes all --kernel-name-base demangled \
      --kernel-name regex:residualRmsNormFloat4 --launch-count 1 \
      --section LaunchStats --force-overwrite \
      --export /output/counter-permission-probe \
      "${executable}" --profile-only \
      > "${output}/probe.stdout" 2> "${output}/probe.stderr"
  local probe_status=$?
  set -e
  if [[ ${probe_status} -ne 0 ]]; then
    local error_code=NSIGHT_COMPUTE_PROBE_FAILED
    local prerequisite='Verify that Nsight Compute can collect one hardware counter on the selected GPU.'
    if grep -F 'ERR_NVGPUCTRPERM' "${output}/probe.stderr" >/dev/null; then
      error_code=NSIGHT_COMPUTE_COUNTER_PERMISSION_UNAVAILABLE
      prerequisite='Ask the machine administrator to enable NVIDIA GPU performance counters, then rerun the qualification command.'
    fi
    PROFILER_VALIDATION="${output}/validation.json" ERROR_CODE_VALUE="${error_code}" \
      PREREQUISITE_VALUE="${prerequisite}" PROBE_STATUS_VALUE="${probe_status}" \
      bounded_run quick "${UV[@]}" run --frozen python -c \
      'import json,os; from pathlib import Path; p=Path(os.environ["PROFILER_VALIDATION"]); v={"schema_version":"upgradeguard.dev/profiler-preflight/v1","status":"infrastructure_invalid","error_code":os.environ["ERROR_CODE_VALUE"],"observed_exit_code":int(os.environ["PROBE_STATUS_VALUE"]),"resume_prerequisite":os.environ["PREREQUISITE_VALUE"]}; p.write_text(json.dumps(v,indent=2,sort_keys=True)+"\n")'
    cat "${output}/validation.json" >&2
    return 4
  fi
  gpu_run "${CANDIDATE_WORKER}" "${PROJECT_ROOT}" "${output}" \
    ncu --import /output/counter-permission-probe.ncu-rep --csv --page details \
      > "${output}/counter-permission-probe.csv"
  grep -F 'residualRmsNormFloat4' "${output}/counter-permission-probe.csv" >/dev/null
  PROFILER_VALIDATION="${output}/validation.json" \
    bounded_run quick "${UV[@]}" run --frozen python -c \
    'import json,os; from pathlib import Path; p=Path(os.environ["PROFILER_VALIDATION"]); p.write_text(json.dumps({"schema_version":"upgradeguard.dev/profiler-preflight/v1","status":"passed","gpu_counter_collection":True,"kernel":"residualRmsNormFloat4"},indent=2,sort_keys=True)+"\n")'
}

run_plugin_benchmark() {
  load_worker_identities
  local output="${STATE_ROOT}/plugin-benchmark"
  mkdir -p "${output}"
  wait_for_idle_observation "${output}/plugin-benchmark-idle.json"
  gpu_run "${CANDIDATE_WORKER}" "${PROJECT_ROOT}" "${output}" \
    /state/plugin-build/candidate/build/upgrade_guard_kernel_benchmark \
    > "${output}/plugin-benchmark-precondition.json"
  bounded_run quick "${UV[@]}" run --frozen python scripts/hardware_validity.py capture \
    --specification "${STATE_ROOT}/full.yaml" --gpu "${GPU_UUID}" --loaded \
    --output "${output}/plugin-benchmark-before.json"
  gpu_run "${CANDIDATE_WORKER}" "${PROJECT_ROOT}" "${output}" \
    /state/plugin-build/candidate/build/upgrade_guard_kernel_benchmark \
    > "${output}/plugin-benchmark.json"
  bounded_run quick "${UV[@]}" run --frozen python scripts/hardware_validity.py capture \
    --specification "${STATE_ROOT}/full.yaml" --gpu "${GPU_UUID}" --loaded \
    --output "${output}/plugin-benchmark-after.json"
  bounded_run quick "${UV[@]}" run --frozen python scripts/hardware_validity.py transition \
    --specification "${STATE_ROOT}/full.yaml" \
    --before "${output}/plugin-benchmark-before.json" \
    --after "${output}/plugin-benchmark-after.json" \
    --output "${output}/plugin-benchmark-validity.json"
  bounded_run quick "${UV[@]}" run --frozen python -c \
    'import json,sys; v=json.load(open(sys.argv[1])); assert v["status"]=="passed" and not v["profiled"]' \
    "${output}/plugin-benchmark.json"
}

run_gpu_smoke() {
  load_worker_identities
  local output="${STATE_ROOT}/smoke"
  local plugin_source="${STATE_ROOT}/plugin-build/candidate/build/libupgrade_guard_residual_rmsnorm.so"
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
  bounded_run quick "${UV[@]}" run --frozen python scripts/validate_gpu_smoke.py \
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
    local plugin="${STATE_ROOT}/plugin-build/${environment}/build/libupgrade_guard_residual_rmsnorm.so"
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
  bounded_run quick "${UV[@]}" run --frozen python scripts/validate_plugin_outputs.py \
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
  bounded_run quick "${UV[@]}" run --frozen python scripts/validate_mobilenet_outputs.py \
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
  bounded_run quick "${UV[@]}" run --frozen python -c \
    'import json,sys; from pathlib import Path; p=Path(sys.argv[1]); json.dump({"tokens":{"min":[1,8,256],"opt":[4,128,256],"max":[8,512,256]},"mask":{"min":[1,1,1,8],"opt":[4,1,1,128],"max":[8,1,1,512]}},p.open("w"))' \
    "${output}/profile.json"
  gpu_run "${BASELINE_WORKER}" "${corpus}" "${output}" \
    python3 -m upgrade_guard.worker.build_engine \
      --model /corpus/models/tiny-transformer-fp32.onnx \
      --profile /output/profile.json \
      --engine /output/engine.plan \
      --inspector /output/inspector.json \
      --timing-cache /output/timing.cache \
      --result /output/build.json
  trtexec_path="$(bounded_run quick "${UV[@]}" run --frozen python -c \
    'import sys; from pathlib import Path; from upgrade_guard.contracts.environment import MatrixLock; m=MatrixLock.model_validate_json(Path(sys.argv[1]).read_text()); print(m.environments[0].probe.trtexec.path)' \
    "${MATRIX_LOCK}" \
  )"
  stream_option="$(bounded_run quick "${UV[@]}" run --frozen python -c \
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
        if bounded_run quick "${UV[@]}" run --frozen python \
          scripts/hardware_validity.py capture \
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
      gpu_run "${BASELINE_WORKER}" "${corpus}" "${output}" \
        "${trtexec_path}" \
          --loadEngine=/output/engine.plan \
          --shapes=mask:1x1x1x128,tokens:1x128x256 \
          "--exportTimes=/output/attempt-${attempt_id}/${side}-precondition.json" \
          --warmUp=500 --duration=1 "${stream_option}" --noDataTransfers
      if ! bounded_run quick "${UV[@]}" run --frozen python \
        scripts/hardware_validity.py capture \
        --specification "${STATE_ROOT}/full.yaml" --gpu "${GPU_UUID}" --loaded \
        --output "${attempt_root}/${side}-before.json"; then
        pair_valid=0
        break
      fi
      gpu_run "${BASELINE_WORKER}" "${corpus}" "${output}" \
        "${trtexec_path}" \
          --loadEngine=/output/engine.plan \
          --shapes=mask:1x1x1x128,tokens:1x128x256 \
          "--exportTimes=/output/attempt-${attempt_id}/${side}.json" \
          --warmUp=500 --duration=1 "${stream_option}" --noDataTransfers
      bounded_run quick "${UV[@]}" run --frozen python scripts/hardware_validity.py capture \
        --specification "${STATE_ROOT}/full.yaml" --gpu "${GPU_UUID}" --loaded \
        --output "${attempt_root}/${side}-after.json" || pair_valid=0
      bounded_run quick "${UV[@]}" run --frozen python \
        scripts/hardware_validity.py transition \
        --specification "${STATE_ROOT}/full.yaml" \
        --before "${attempt_root}/${side}-before.json" \
        --after "${attempt_root}/${side}-after.json" \
        --output "${attempt_root}/${side}-validity.json" || pair_valid=0
      if [[ ${pair_valid} -ne 1 ]]; then
        break
      fi
    done
    if [[ ${pair_valid} -eq 1 ]]; then
      bounded_run quick "${UV[@]}" run --frozen python -c \
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
  bounded_run quick "${UV[@]}" run --frozen python scripts/validate_aa.py \
    --pairs "${output}" --output "${output}/validation.json"
}

materialize_fault_inputs() {
  cd "${PROJECT_ROOT}"
  local output="${STATE_ROOT}/fault-inputs"
  mkdir -p "${output}"
  if [[ ! -f "${output}/inputs.json" ]]; then
    bounded_run quick "${UV[@]}" run --frozen python \
      scripts/materialize_gpu_fault_inputs.py "${output}"
  fi
}

run_gpu_faults() {
  load_worker_identities
  local output="${STATE_ROOT}/gpu-faults"
  local build="${STATE_ROOT}/plugin-build/candidate/build"
  local samples="${output}/gpu-fault-samples.jsonl"
  mkdir -p "${output}"
  : > "${samples}"
  local accepted=0
  local attempt=0
  while [[ ${accepted} -lt 20 && ${attempt} -lt 60 ]]; do
    local attempt_id
    attempt_id="$(printf '%02d' "${attempt}")"
    local attempt_root="${output}/attempt-${attempt_id}"
    mkdir -p "${attempt_root}"
    local pair_valid=1
    if ! wait_for_idle_observation "${attempt_root}/idle.json"; then
      pair_valid=0
    fi
    if [[ ${pair_valid} -eq 1 ]]; then
      gpu_run "${CANDIDATE_WORKER}" "${PROJECT_ROOT}" "${output}" \
        /state/plugin-build/candidate/build/upgrade_guard_gpu_faults \
        --pair-index "${accepted}" > "${attempt_root}/precondition.json"
      bounded_run quick "${UV[@]}" run --frozen python \
        scripts/hardware_validity.py capture \
        --specification "${STATE_ROOT}/full.yaml" --gpu "${GPU_UUID}" --loaded \
        --output "${attempt_root}/before.json" || pair_valid=0
    fi
    if [[ ${pair_valid} -eq 1 ]]; then
      gpu_run "${CANDIDATE_WORKER}" "${PROJECT_ROOT}" "${output}" \
        /state/plugin-build/candidate/build/upgrade_guard_gpu_faults \
        --pair-index "${accepted}" > "${attempt_root}/sample.json"
      bounded_run quick "${UV[@]}" run --frozen python \
        scripts/hardware_validity.py capture \
        --specification "${STATE_ROOT}/full.yaml" --gpu "${GPU_UUID}" --loaded \
        --output "${attempt_root}/after.json" || pair_valid=0
      bounded_run quick "${UV[@]}" run --frozen python \
        scripts/hardware_validity.py transition \
        --specification "${STATE_ROOT}/full.yaml" \
        --before "${attempt_root}/before.json" --after "${attempt_root}/after.json" \
        --output "${attempt_root}/validity.json" || pair_valid=0
    fi
    if [[ ${pair_valid} -eq 1 ]]; then
      bounded_run quick "${UV[@]}" run --frozen python -c \
        'import json,sys; v=json.load(open(sys.argv[1])); assert v["status"]=="passed"' \
        "${attempt_root}/validity.json"
      mv "${attempt_root}" "${output}/pair-$(printf '%02d' "${accepted}")"
      cat "${output}/pair-$(printf '%02d' "${accepted}")/sample.json" >> "${samples}"
      accepted=$((accepted + 1))
    else
      mv "${attempt_root}" "${output}/rejected-attempt-${attempt_id}"
    fi
    attempt=$((attempt + 1))
  done
  [[ ${accepted} -eq 20 ]]
  bounded_run quick "${UV[@]}" run --frozen python \
    scripts/validate_seeded_gpu_faults.py \
    --samples "${samples}" --output "${output}/validation.json"
  gpu_run "${CANDIDATE_WORKER}" "${PROJECT_ROOT}" "${output}" \
    /state/plugin-build/candidate/build/upgrade_guard_serialization_fault \
    > "${output}/G6.json"
  bounded_run quick "${UV[@]}" run --frozen python -c \
    'import json,sys; p=sys.argv[1]; v=json.load(open(p)); assert v["detected"] and v["control"]=="passed"' \
    "${output}/G6.json"

  mkdir -p "${output}/g1"
  write_plugin_profile "${output}/g1/profile.json"
  set +e
  gpu_run "${CANDIDATE_WORKER}" "${PLUGIN_CORPUS}" \
    "${output}" python3 -m upgrade_guard.worker.build_engine \
      --model /corpus/residual-rmsnorm-fp32.onnx \
      --profile /output/g1/profile.json \
      --engine /output/g1/engine.plan \
      --inspector /output/g1/inspector.json \
      --timing-cache /output/g1/timing.cache \
      --result /output/g1/build.json
  local g1_status=$?
  set -e
  if [[ ${g1_status} -eq 0 ]]; then
    printf 'Unsupported custom-domain graph unexpectedly parsed without its plugin.\n' >&2
    return 1
  fi
  bounded_run quick "${UV[@]}" run --frozen python -c \
    'import json,sys; from pathlib import Path; from upgrade_guard.contracts.base import sha256_file; src,control,dst=map(Path,sys.argv[1:]); v=json.load(src.open()); c=json.load(control.open()); assert v["status"]=="failed" and "parser" in v["message"].lower(); assert c["status"]=="passed"; json.dump({"expected":"ONNX_PARSE_FAILED","detected":True,"control":"passed","control_build_sha256":sha256_file(control),"worker":v},dst.open("w"),indent=2)' \
    "${output}/g1/build.json" \
    "${STATE_ROOT}/plugin-runs/candidate/fp32/build.json" "${output}/G1.json"

  mkdir -p "${output}/g7"
  gpu_run "${CANDIDATE_WORKER}" "${CORE_CORPUS}" \
    "${output}" python3 -m upgrade_guard.worker.run_correctness \
      --engine /state/core-run/candidate/fp32/engine-0.plan \
      --input tokens=/corpus/inputs/tiny-transformer-fp32/b8_s512/tokens.npy \
      --input mask=/corpus/inputs/tiny-transformer-fp32/b8_s512/mask.npy \
      --output /output/g7/control-outputs \
      --result /output/g7/control-correctness.json \
      --repetitions 20
  set +e
  gpu_run "${CANDIDATE_WORKER}" "${CORE_CORPUS}" \
    "${output}" python3 -m upgrade_guard.worker.run_correctness \
      --engine /state/core-run/candidate/fp32/engine-0.plan \
      --input tokens=/state/fault-inputs/g7/tokens.npy \
      --input mask=/state/fault-inputs/g7/mask.npy \
      --output /output/g7/outputs \
      --result /output/g7/correctness.json \
      --repetitions 20
  local g7_status=$?
  set -e
  if [[ ${g7_status} -eq 0 ]]; then
    printf 'Out-of-profile input unexpectedly executed.\n' >&2
    return 1
  fi
  bounded_run quick "${UV[@]}" run --frozen python -c \
    'import json,sys; from pathlib import Path; from upgrade_guard.contracts.base import sha256_file; src,control,dst=map(Path,sys.argv[1:]); v=json.load(src.open()); c=json.load(control.open()); assert v["status"]=="failed" and "shape was rejected" in v["message"]; assert c["status"]=="passed"; json.dump({"expected":"PROFILE_REJECTED","detected":True,"control":"passed","control_run_sha256":sha256_file(control),"worker":v},dst.open("w"),indent=2)' \
    "${output}/g7/correctness.json" "${output}/g7/control-correctness.json" \
    "${output}/G7.json"
  [[ -d "${build}" ]]
}

prepare_reductions() {
  load_corpus_identities
  local root="${STATE_ROOT}/reductions"
  bounded_run build "${UV[@]}" run --frozen python \
    scripts/create_remote_reproductions.py \
    --state "${STATE_ROOT}" --project "${PROJECT_ROOT}" \
    --core-corpus "${CORE_CORPUS}" --plugin-corpus "${PLUGIN_CORPUS}" \
    --output "${root}/prepared"
}

run_replay_seed() {
  local seed=$1
  local root="${STATE_ROOT}/reductions"
  local bundle="${root}/prepared/${seed}-clean-bundle"
  local seed_root="${root}/${seed}"
  mkdir -p "${seed_root}"
  bounded_run build "${UV[@]}" run --frozen upgrade-guard reproduce run "${bundle}" \
    --out "${seed_root}/replay-output" --trust-source-code --json \
    > "${seed_root}/cli-result.json"
  SEED_VALUE="${seed}" PREPARED_PATH="${root}/prepared/prepared.json" \
    REPLAY_PATH="${seed_root}/replay-output/replay-result.json" \
    CLI_PATH="${seed_root}/cli-result.json" \
    bounded_run quick "${UV[@]}" run --frozen python -c \
    'import json,os; expected={"G2":"NUMERICAL_REGRESSION","G7":"PROFILE_REJECTED"}; seed=os.environ["SEED_VALUE"]; prepared=json.load(open(os.environ["PREPARED_PATH"])); replay=json.load(open(os.environ["REPLAY_PATH"])); cli=json.load(open(os.environ["CLI_PATH"])); bundle=prepared["clean_bundles"][seed]; assert replay==cli and replay["status"]=="passed" and replay["expected_failure_code"]==expected[seed] and replay["bundle_manifest_sha256"]==bundle["bundle_manifest_sha256"]'
}

validate_reduction_replays() {
  local root="${STATE_ROOT}/reductions"
  bounded_run quick "${UV[@]}" run --frozen python -c \
    'import json,sys; from pathlib import Path; root,out=map(Path,sys.argv[1:]); prepared=json.load((root/"prepared/prepared.json").open()); expected={"G2":"NUMERICAL_REGRESSION","G7":"PROFILE_REJECTED"}; replays={};
for seed,code in expected.items():
 value=json.load((root/f"{seed}/replay-output/replay-result.json").open()); cli=json.load((root/f"{seed}/cli-result.json").open()); bundle=prepared["clean_bundles"][seed]; assert value==cli and value["status"]=="passed" and value["expected_failure_code"]==code and value["bundle_manifest_sha256"]==bundle["bundle_manifest_sha256"]; replays[seed]=value
prepared["status"]="passed"; prepared["clean_replays"]=replays; json.dump(prepared,out.open("w"),indent=2,sort_keys=True)' \
    "${root}" "${root}/validation.json"
}

run_memory_seed() {
  load_worker_identities
  local output="${STATE_ROOT}/memory-seed"
  local corpus="${PLUGIN_CORPUS}"
  local plugin="/state/plugin-build/candidate/build/libupgrade_guard_residual_rmsnorm.so"
  mkdir -p "${output}/control" "${output}/seeded"
  write_plugin_profile "${output}/profile.json"
  for kind in control seeded; do
    local model=/corpus/residual-rmsnorm-fp32.onnx
    if [[ "${kind}" == seeded ]]; then
      model=/state/fault-inputs/residual-rmsnorm-workspace-seed.onnx
    fi
    for build_index in 0 1 2; do
      gpu_run "${CANDIDATE_WORKER}" "${corpus}" "${output}" \
        python3 -m upgrade_guard.worker.build_engine \
          --model "${model}" \
          --profile /output/profile.json \
          --engine "/output/${kind}/engine-${build_index}.plan" \
          --inspector "/output/${kind}/inspector-${build_index}.json" \
          --timing-cache "/output/${kind}/timing.cache" \
          --result "/output/${kind}/build-${build_index}.json" \
          --plugin "${plugin}"
    done
  done
  bounded_run quick "${UV[@]}" run --frozen python scripts/validate_memory_seed.py \
    --control "${output}/control" --seeded "${output}/seeded" \
    --output "${output}/validation.json"
}

run_sanitizers() {
  load_worker_identities
  local output="${STATE_ROOT}/sanitizers"
  local executable="/state/plugin-build/candidate/build/upgrade_guard_kernel_tests"
  mkdir -p "${output}"
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
      /state/plugin-build/candidate/build/upgrade_guard_tail_oob_fault \
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
  bounded_run quick "${UV[@]}" run --frozen python -c \
    'import json,sys; from pathlib import Path; log=Path(sys.argv[1]); out=Path(sys.argv[2]); json.dump({"expected":"SANITIZER_FAILURE","observed_exit_code":86,"diagnostic":"out_of_bounds_global_access","diagnostic_log_sha256":__import__("upgrade_guard.contracts.base",fromlist=["sha256_file"]).sha256_file(log),"control":"passed"},out.open("w"),indent=2,sort_keys=True)' \
    "${output}/sanitizer-tail-oob.log" "${output}/sanitizer-seed.json"
}

run_profiles() {
  load_worker_identities
  local output="${STATE_ROOT}/profiles"
  mkdir -p "${output}"
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
      /state/plugin-build/candidate/build/upgrade_guard_kernel_benchmark --profile-only
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
      /state/plugin-build/candidate/build/upgrade_guard_kernel_benchmark --profile-only
  gpu_run "${CANDIDATE_WORKER}" "${PROJECT_ROOT}" "${output}" \
    ncu --import /output/residual-rmsnorm-kernel.ncu-rep --csv --page details \
      > "${output}/ncu-kernel-summary.csv"
}

generate_sboms() {
  load_worker_identities
  local output="${STATE_ROOT}/sbom"
  mkdir -p "${output}"
  bounded_run quick "${UV[@]}" run --frozen python scripts/generate_host_sbom.py \
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
  bounded_run quick "${UV[@]}" export --frozen --no-dev --no-emit-project \
    --format requirements.txt --output-file "${output}/requirements.txt"
  bounded_run audit "${UV[@]}" run --frozen pip-audit \
    --requirement "${output}/requirements.txt" \
    --disable-pip --require-hashes \
    --format json --output "${output}/pip-audit.json" \
    --progress-spinner off
  bounded_run audit "${UV[@]}" run --frozen pip-audit \
    --requirement containers/requirements-worker.txt \
    --disable-pip --require-hashes \
    --format json --output "${output}/worker-pip-audit.json" \
    --progress-spinner off
  TRIAGE_PATH="${output}/triage.json" \
    bounded_run quick "${UV[@]}" run --frozen python -c \
    'import json,os; from pathlib import Path; p=Path(os.environ["TRIAGE_PATH"]); v={"schema_version":"upgradeguard.dev/dependency-triage/v1","status":"passed","audited_scopes":["uv.lock Python dependencies","hash-locked Python dependencies added by containers/Dockerfile.worker"],"inventory_only_scopes":["preinstalled NGC Python packages","worker Debian packages","NVIDIA proprietary packages"],"claim":"Passing audits apply only to the explicitly audited Python scopes. Worker SPDX documents inventory the remaining packages but do not claim vulnerability-free images.","release_policy":"Any published result must retain this limitation and the exact worker SBOM hashes."}; p.write_text(json.dumps(v,indent=2,sort_keys=True)+"\n")'
}

finalize() {
  cd "${PROJECT_ROOT}"
  bounded_run evidence "${UV[@]}" run --frozen python scripts/generate_remote_evidence.py \
    --state "${STATE_ROOT}" --output "${STATE_ROOT}/evidence.json"
  printf 'COMPLETE evidence=%s\n' "${STATE_ROOT}/evidence.json"
}

terminal_cleanup() {
  if bounded_run cleanup docker container inspect "${REGISTRY_NAME}" >/dev/null 2>&1; then
    bounded_run cleanup docker container rm --force "${REGISTRY_NAME}" >/dev/null
  fi
  bounded_run cleanup docker volume inspect "${REGISTRY_VOLUME}" >/dev/null
  CLEANUP_PATH="${STATE_ROOT}/cleanup.json" REGISTRY_NAME_VALUE="${REGISTRY_NAME}" \
    REGISTRY_VOLUME_VALUE="${REGISTRY_VOLUME}" SOURCE_ID_VALUE="${SOURCE_ID}" \
    REGISTRY_IDENTITY_PATH="${STATE_ROOT}/registry-identity.json" python3 -c \
    'import hashlib,json,os,tempfile; from pathlib import Path; source=Path(os.environ["REGISTRY_IDENTITY_PATH"]); digest="sha256:"+hashlib.sha256(source.read_bytes()).hexdigest(); out=Path(os.environ["CLEANUP_PATH"]); value={"schema_version":"upgradeguard.dev/terminal-cleanup/v1","status":"passed","source_git_commit":os.environ["SOURCE_ID_VALUE"],"container":os.environ["REGISTRY_NAME_VALUE"],"container_removed":True,"volume":os.environ["REGISTRY_VOLUME_VALUE"],"volume_retained":True,"registry_identity_sha256":digest}; f=tempfile.NamedTemporaryFile("w",dir=out.parent,prefix=f".{out.name}.",delete=False); json.dump(value,f,indent=2,sort_keys=True); f.write("\n"); f.flush(); os.fsync(f.fileno()); f.close(); Path(f.name).replace(out)'
}

cd "${PROJECT_ROOT}"
initialize_bounded_executor
select_uv
invocation_guard
run_step preflight preflight
run_step cpu-verify cpu_verify
run_always_step gpu-runtime-preflight gpu_runtime_preflight
CURRENT_STEP=reconcile
reconcile_state
if [[ "${RUN_MODE}" == "full" ]]; then
  run_step dependency-audit audit_dependencies
fi
run_always_step registry-bootstrap start_local_registry
run_step capacity-preflight capacity_preflight
if [[ "${RUN_MODE}" == "full" ]]; then
  run_step corpus-materialization materialize_corpora
else
  run_step corpus-materialization materialize_bounded_corpora
fi
run_step worker-images build_workers
ensure_worker_registry
preserve_stale_matrix
run_step matrix-lock lock_matrix
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
run_step plugin-compile-test compile_plugins
run_step profiler-preflight run_profiler_preflight
run_step aa-pilot run_aa_pilot
run_step core-qualification run_core_qualification
run_step plugin-benchmark run_plugin_benchmark
run_step plugin-matrix run_plugin_matrix
run_step mobilenet-matrix run_mobilenet_matrix
run_step fault-inputs materialize_fault_inputs
run_step gpu-faults run_gpu_faults
run_step reduction-prepare prepare_reductions
run_step replay-G2 run_replay_seed G2
run_step replay-G7 run_replay_seed G7
run_step reduction-validation validate_reduction_replays
run_step memory-seed run_memory_seed
run_step sanitizers run_sanitizers
run_step profiles run_profiles
run_step sboms generate_sboms
run_step final-evidence finalize
CURRENT_STEP=terminal-cleanup
terminal_cleanup 2>&1 | tee "${LOG_ROOT}/terminal-cleanup.log"
bounded_run quick "${UV[@]}" run --frozen python scripts/qualification_state.py record \
  --state "${STATE_ROOT}" --project "${PROJECT_ROOT}" --step terminal-cleanup \
  --source "${SOURCE_ID}" --gpu "${GPU_UUID}" --mode "${RUN_MODE}"
