#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SOURCE_ID="$(git -C "${PROJECT_ROOT}" rev-parse HEAD)"
STATE_BASE="${PROJECT_ROOT}/.upgrade-guard/cuda-pm"
STATE_ROOT="${STATE_BASE}/runs/${SOURCE_ID}"
LOG_ROOT="${STATE_ROOT}/logs"
MATRIX_LOCK="${STATE_ROOT}/matrix.lock.json"
CORE_CORPUS="${PROJECT_ROOT}/.upgrade-guard/corpora/v1-core"
PLUGIN_CORPUS="${STATE_ROOT}/corpora/plugin"
MOBILENET_CORPUS="${STATE_ROOT}/corpora/mobilenet"
REGISTRY_NAME="upgrade-guard-registry"
REGISTRY_ADDRESS="127.0.0.1:5500"
REGISTRY_IMAGE="registry@sha256:46faa9a1ae6813194b53921a370f2f4f8c5e1aae228a89bceafef5847a6a3278"
BASELINE_BASE="nvcr.io/nvidia/tensorrt:26.06-py3@sha256:2a5a0a9a32ec5ddc1c384c15ddcf3b89ddc4f8647e7ee7ae708d844210183a1e"
CANDIDATE_BASE="nvcr.io/nvidia/tensorrt:26.07-py3@sha256:b82db1abc23750ab0069abc99bbe4ea29138dbdc23ea39861199e2346638b48a"
GPU_INDEX="${UG_GPU_INDEX:-0}"
CURRENT_STEP="initialization"
THROUGH_STEP="${UG_THROUGH_STEP:-}"
SMOKE_ONLY="${UG_SMOKE_ONLY:-0}"

DONE_ROOT="${STATE_ROOT}/done"

mkdir -p "${DONE_ROOT}" "${LOG_ROOT}"

failure_report() {
  local exit_code=$?
  if docker container inspect "${REGISTRY_NAME}" >/dev/null 2>&1; then
    docker stop "${REGISTRY_NAME}" >/dev/null 2>&1 || true
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
  if [[ -f "${DONE_ROOT}/${name}" ]]; then
    printf 'SKIP completed step: %s\n' "${name}"
    if [[ "${THROUGH_STEP}" == "${name}" ]]; then
      exit 0
    fi
    return
  fi
  printf 'RUN step: %s\n' "${name}"
  "$@" 2>&1 | tee "${LOG_ROOT}/${name}.log"
  touch "${DONE_ROOT}/${name}"
  if [[ "${THROUGH_STEP}" == "${name}" ]]; then
    exit 0
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
  [[ -z "$(git status --porcelain --untracked-files=normal)" ]]
  git rev-parse HEAD > "${STATE_ROOT}/source.commit"
  [[ "$(uname -s)" == "Linux" ]]
  [[ "$(uname -m)" == "x86_64" ]]
  command -v docker
  command -v nvidia-smi
  command -v curl
  docker version
  nvidia-smi
  GPU_UUID="$(nvidia-smi --id="${GPU_INDEX}" --query-gpu=uuid --format=csv,noheader | tr -d '[:space:]')"
  [[ "${GPU_UUID}" == GPU-* ]]
  if [[ -n "${UG_EXPECTED_GPU_UUID:-}" && "${GPU_UUID}" != "${UG_EXPECTED_GPU_UUID}" ]]; then
    printf 'Selected GPU UUID %s does not match required UUID %s.\n' \
      "${GPU_UUID}" "${UG_EXPECTED_GPU_UUID}" >&2
    return 1
  fi
  printf '%s\n' "${GPU_UUID}" > "${STATE_ROOT}/gpu.uuid"
  nvidia-smi --id="${GPU_UUID}" \
    --query-gpu=name,uuid,compute_cap,memory.total,driver_version,vbios_version,power.limit \
    --format=csv,noheader > "${STATE_ROOT}/gpu-preflight.csv"
}

cpu_verify() {
  cd "${PROJECT_ROOT}"
  "${UV[@]}" sync --frozen
  "${UV[@]}" run --frozen python scripts/generate_schemas.py
  "${UV[@]}" run --frozen ruff check .
  "${UV[@]}" run --frozen ruff format --check .
  "${UV[@]}" run --frozen mypy
  "${UV[@]}" run --frozen pytest --cov=upgrade_guard --cov-report=term-missing
}

start_local_registry() {
  docker pull "${REGISTRY_IMAGE}"
  if docker container inspect "${REGISTRY_NAME}" >/dev/null 2>&1; then
    docker start "${REGISTRY_NAME}" >/dev/null || true
  else
    docker run -d --name "${REGISTRY_NAME}" \
      -p "127.0.0.1:5500:5000" \
      "${REGISTRY_IMAGE}" >/dev/null
  fi
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
  if [[ ! -f "${CORE_CORPUS}/corpus.lock.json" ]]; then
    "${UV[@]}" run --frozen upgrade-guard corpus materialize corpus/registry.yaml \
      --out "${CORE_CORPUS}" --json
  fi
  if [[ ! -f "${PLUGIN_CORPUS}/plugin-corpus.lock.json" ]]; then
    "${UV[@]}" run --frozen python scripts/materialize_plugin_corpus.py "${PLUGIN_CORPUS}"
  fi
  if [[ ! -f "${MOBILENET_CORPUS}/mobilenet-corpus.lock.json" ]]; then
    "${UV[@]}" run --frozen python scripts/materialize_mobilenet_corpus.py \
      "${MOBILENET_CORPUS}"
  fi
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

gpu_run() {
  local image=$1
  local corpus=$2
  local output=$3
  shift 3
  mkdir -p "${output}"
  docker run --rm --init --user "$(id -u):$(id -g)" \
    --gpus "device=${GPU_UUID}" \
    --network none --read-only --cap-drop ALL \
    --security-opt no-new-privileges --pids-limit 512 --ipc private \
    --tmpfs /tmp:rw,noexec,nosuid,size=1073741824 \
    --mount "type=bind,src=${PROJECT_ROOT},dst=/opt/upgrade-guard,readonly" \
    --mount "type=bind,src=${corpus},dst=/corpus,readonly" \
    --mount "type=bind,src=${output},dst=/output" \
    --env PYTHONPATH=/opt/upgrade-guard/src \
    "${image}" "$@"
}

bundle_gpu_run() {
  local image=$1
  local bundle=$2
  local output=$3
  shift 3
  mkdir -p "${output}"
  docker run --rm --init --user "$(id -u):$(id -g)" \
    --gpus "device=${GPU_UUID}" \
    --network none --read-only --cap-drop ALL \
    --security-opt no-new-privileges --pids-limit 512 --ipc private \
    --tmpfs /tmp:rw,noexec,nosuid,size=1073741824 \
    --mount "type=bind,src=${bundle},dst=/bundle,readonly" \
    --mount "type=bind,src=${output},dst=/output" \
    "${image}" "$@"
}

run_core_qualification() {
  cd "${PROJECT_ROOT}"
  if [[ -f "${STATE_ROOT}/core-run/qualification-summary.json" ]]; then
    "${UV[@]}" run --frozen upgrade-guard compare "${STATE_ROOT}/core-run" --json
    return
  fi
  "${UV[@]}" run --frozen upgrade-guard qualify "${STATE_ROOT}/full.yaml" \
    --out "${STATE_ROOT}/core-run" --json
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
  gpu_run "${CANDIDATE_WORKER}" "${PROJECT_ROOT}" "${output}" \
    /output/build/upgrade_guard_kernel_benchmark \
    > "${output}/plugin-benchmark.json"
  "${UV[@]}" run --frozen python -c \
    'import json,sys; v=json.load(open(sys.argv[1])); assert v["status"]=="passed" and not v["profiled"]' \
    "${output}/plugin-benchmark.json"
}

run_gpu_smoke() {
  load_worker_identities
  local corpus="${PLUGIN_CORPUS}"
  local output="${STATE_ROOT}/smoke"
  local plugin_source="${STATE_ROOT}/plugin/candidate/build/libupgrade_guard_residual_rmsnorm.so"
  mkdir -p "${output}"
  cp "${plugin_source}" "${output}/libupgrade_guard_residual_rmsnorm.so"
  write_plugin_profile "${output}/profile.json"
  gpu_run "${CANDIDATE_WORKER}" "${corpus}" "${output}" \
    python3 -m upgrade_guard.worker.build_engine \
      --model /corpus/residual-rmsnorm-fp32.onnx \
      --profile /output/profile.json \
      --engine /output/engine.plan \
      --inspector /output/inspector.json \
      --timing-cache /output/timing.cache \
      --result /output/build.json \
      --plugin /output/libupgrade_guard_residual_rmsnorm.so
  gpu_run "${CANDIDATE_WORKER}" "${corpus}" "${output}" \
    python3 -m upgrade_guard.worker.run_correctness \
      --engine /output/engine.plan \
      --input x=/corpus/fp32/minimum-zero-h7/x.npy \
      --input residual=/corpus/fp32/minimum-zero-h7/residual.npy \
      --input gamma=/corpus/fp32/minimum-zero-h7/gamma.npy \
      --output /output/outputs \
      --result /output/correctness.json \
      --repetitions 20 \
      --plugin /output/libupgrade_guard_residual_rmsnorm.so
  "${UV[@]}" run --frozen python -c \
    'import json,sys; build=json.load(open(sys.argv[1])); run=json.load(open(sys.argv[2])); assert build["status"]=="passed" and run["status"]=="passed"' \
    "${output}/build.json" "${output}/correctness.json"
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
  mkdir -p "${output}"
  trtexec_path="$("${UV[@]}" run --frozen python -c \
    'import sys; from pathlib import Path; from upgrade_guard.contracts.environment import MatrixLock; m=MatrixLock.model_validate_json(Path(sys.argv[1]).read_text()); print(m.environments[0].probe.trtexec.path)' \
    "${MATRIX_LOCK}" \
  )"
  stream_option="$("${UV[@]}" run --frozen python -c \
    'import sys; from pathlib import Path; from upgrade_guard.contracts.environment import MatrixLock; m=MatrixLock.model_validate_json(Path(sys.argv[1]).read_text()); o=m.environments[0].probe.trtexec.options; print("--infStreams=1" if "--infStreams" in o else "--streams=1")' \
    "${MATRIX_LOCK}" \
  )"
  for pair in $(seq -w 0 19); do
    local pair_root="${output}/pair-${pair}"
    mkdir -p "${pair_root}"
    for side in a b; do
      if nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader,nounits \
        | grep -Fqx "${GPU_UUID}"; then
        printf 'Competing process appeared during A/A pilot.\n' >&2
        return 1
      fi
      gpu_run "${BASELINE_WORKER}" "${corpus}" "${STATE_ROOT}" \
        "${trtexec_path}" \
          --loadEngine=/output/core-run/baseline/fp32/engine-0.plan \
          --shapes=mask:1x1x1x128,tokens:1x128x256 \
          "--exportTimes=/output/aa/pair-${pair}/${side}.json" \
          --warmUp=500 --duration=1 "${stream_option}" --noDataTransfers
    done
  done
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
    'import json,sys; src,dst=sys.argv[1:]; v=json.load(open(src)); assert v["status"]=="failed" and "parser" in v["message"].lower(); json.dump({"expected":"ONNX_PARSE_FAILED","detected":True,"control":"passed","worker":v},open(dst,"w"),indent=2)' \
    "${output}/g1/build.json" "${output}/G1.json"

  mkdir -p "${output}/g7"
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
    'import json,sys; src,dst=sys.argv[1:]; v=json.load(open(src)); assert v["status"]=="failed" and "shape was rejected" in v["message"]; json.dump({"expected":"PROFILE_REJECTED","detected":True,"control":"passed","worker":v},open(dst,"w"),indent=2)' \
    "${output}/g7/correctness.json" "${output}/G7.json"
  [[ -d "${build}" ]]
}

run_reduction_replay() {
  load_worker_identities
  local root="${STATE_ROOT}/reductions"
  local bundle="${root}/clean-bundle"
  local output="${root}/clean-replay-output"
  "${UV[@]}" run --frozen python scripts/create_remote_reproductions.py \
    --state "${STATE_ROOT}" --project "${PROJECT_ROOT}" \
    --plugin-corpus "${PLUGIN_CORPUS}"
  bundle_gpu_run "${CANDIDATE_WORKER}" "${bundle}" "${output}" \
    cmake -S /bundle -B /output/build -G Ninja \
      -DCMAKE_BUILD_TYPE=RelWithDebInfo \
      -DUPGRADE_GUARD_BUILD_TESTS=OFF \
      -DUPGRADE_GUARD_BUILD_FAULTS=ON
  bundle_gpu_run "${CANDIDATE_WORKER}" "${bundle}" "${output}" \
    cmake --build /output/build --target upgrade_guard_gpu_faults
  : > "${output}/samples.jsonl"
  for _ in 1 2; do
    bundle_gpu_run "${CANDIDATE_WORKER}" "${bundle}" "${output}" \
      /output/build/upgrade_guard_gpu_faults >> "${output}/samples.jsonl"
  done
  "${UV[@]}" run --frozen python -c \
    'import json,sys; root,out=sys.argv[1:]; records=[json.loads(x) for x in open(root+"/clean-replay-output/samples.jsonl") if x.strip()]; assert len(records)==2 and all(r["G2"]["detected"] and r["G2"]["control"]=="passed" for r in records); prepared=json.load(open(root+"/prepared.json")); prepared["status"]="passed"; prepared["clean_gpu_confirmations"]=len(records); json.dump(prepared,open(out,"w"),indent=2,sort_keys=True)' \
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
      compute-sanitizer --tool "${tool}" --error-exitcode 91 "${executable}"
  done
  set +e
  gpu_run "${CANDIDATE_WORKER}" "${PROJECT_ROOT}" "${output}" \
    compute-sanitizer --tool memcheck --error-exitcode 86 \
      /output/build/upgrade_guard_tail_oob_fault
  local fault_status=$?
  set -e
  if [[ ${fault_status} -ne 86 ]]; then
    printf 'Quarantined tail defect did not fail with the expected sanitizer code.\n' >&2
    return 1
  fi
  printf '{"expected":"SANITIZER_FAILURE","observed_exit_code":86,"control":"passed"}\n' \
    > "${output}/sanitizer-seed.json"
}

run_profiles() {
  load_worker_identities
  local output="${STATE_ROOT}/plugin/candidate"
  gpu_run "${CANDIDATE_WORKER}" "${PROJECT_ROOT}" "${output}" \
    nsys --version
  gpu_run "${CANDIDATE_WORKER}" "${PROJECT_ROOT}" "${output}" \
    ncu --version
  gpu_run "${CANDIDATE_WORKER}" "${PROJECT_ROOT}" "${output}" \
    nsys profile --trace=cuda,nvtx --sample=none --capture-range=nvtx \
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
    ncu --target-processes all --kernel-name regex:residualRmsNormFloat4 \
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
}

finalize() {
  cd "${PROJECT_ROOT}"
  "${UV[@]}" run --frozen python scripts/generate_remote_evidence.py \
    --state "${STATE_ROOT}" --output "${STATE_ROOT}/evidence.json"
  docker stop "${REGISTRY_NAME}" >/dev/null || true
  printf 'COMPLETE evidence=%s\n' "${STATE_ROOT}/evidence.json"
}

cd "${PROJECT_ROOT}"
select_uv
run_step preflight preflight
GPU_UUID="$(<"${STATE_ROOT}/gpu.uuid")"
run_step cpu-verify cpu_verify
run_step local-registry start_local_registry
docker start "${REGISTRY_NAME}" >/dev/null || true
run_step worker-images build_workers
run_step matrix-lock lock_matrix
run_step corpus-materialization materialize_corpora
if [[ "${SMOKE_ONLY}" == "1" ]]; then
  run_step plugin-compile-test compile_plugins
  run_step gpu-smoke run_gpu_smoke
  exit 0
fi
run_step core-qualification run_core_qualification
run_step aa-pilot run_aa_pilot
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
