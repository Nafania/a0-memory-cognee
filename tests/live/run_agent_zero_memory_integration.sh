#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ENV_FILE="${ROOT_DIR}/tests/live/.env.local"

if [[ -f "${ENV_FILE}" ]]; then
  while IFS= read -r line || [[ -n "${line}" ]]; do
    line="${line%%#*}"
    [[ "${line}" == *"="* ]] || continue
    key="${line%%=*}"
    value="${line#*=}"
    key="$(echo "${key}" | xargs)"
    [[ -n "${key}" ]] || continue
    if [[ -z "${!key+x}" ]]; then
      export "${key}=${value}"
    fi
  done < "${ENV_FILE}"
fi

IMAGE="${A0_MEMORY_COGNEE_IMAGE:-agent0ai/agent-zero:latest}"
CONTAINER="${A0_MEMORY_COGNEE_CONTAINER:-a0-memory-cognee-itest}"
PORT="${A0_MEMORY_COGNEE_PORT:-18183}"
CODE_DIR="${A0_MEMORY_COGNEE_CODE_DIR:-/git/agent-zero}"
FACT_COUNT="${A0_MEMORY_COGNEE_FACT_COUNT:-20}"
TIMEOUT="${A0_MEMORY_COGNEE_TIMEOUT:-1800}"
KEEP_CONTAINER="${A0_MEMORY_COGNEE_KEEP_CONTAINER:-0}"
API_KEY="${API_KEY_OPENAI:-${OPENAI_API_KEY:-}}"
USR_PARENT="${A0_MEMORY_COGNEE_USR_PARENT:-}"
USR_DIR=""

usage() {
  cat <<'EOF'
Usage:
  tests/live/run_agent_zero_memory_integration.sh [test|start|stop]

Modes:
  test   start fresh container and run live integration test
  start  start fresh container and leave it running
  stop   stop test container

Required local secret:
  tests/live/.env.local with API_KEY_OPENAI=...
EOF
}

mode="${1:-test}"
case "${mode}" in
  test|start|stop) ;;
  -h|--help|help) usage; exit 0 ;;
  *) usage >&2; exit 2 ;;
esac

if [[ "${mode}" == "stop" ]]; then
  docker rm -f "${CONTAINER}" >/dev/null 2>&1 || true
  exit 0
fi

if [[ -z "${API_KEY}" ]]; then
  echo "Missing API_KEY_OPENAI. Put it in tests/live/.env.local." >&2
  exit 2
fi

cleanup() {
  if [[ "${mode}" == "test" && "${KEEP_CONTAINER}" != "1" ]]; then
    docker rm -f "${CONTAINER}" >/dev/null 2>&1 || true
    if [[ -n "${USR_PARENT}" && -d "${USR_PARENT}" ]]; then
      rm -rf "${USR_PARENT}"
    fi
  fi
}
trap cleanup EXIT

if [[ -z "${USR_PARENT}" ]]; then
  USR_PARENT="$(mktemp -d "${TMPDIR:-/tmp}/a0-memory-cognee-itest.XXXXXX")"
fi
USR_DIR="${USR_PARENT}/usr"
mkdir -p \
  "${USR_DIR}/plugins/_memory" \
  "${USR_DIR}/plugins/_model_config" \
  "${USR_DIR}/memory/default" \
  "${USR_DIR}/workdir"

cat > "${USR_DIR}/.env" <<EOF
API_KEY_OPENAI=${API_KEY}
ALLOWED_ORIGINS=*://localhost,*://localhost:*,*://127.0.0.1,*://127.0.0.1:*
DEFAULT_USER_TIMEZONE=Europe/Paris
DEFAULT_USER_UTC_OFFSET_MINUTES=120
ANONYMIZED_TELEMETRY=false
FLASK_SECRET_KEY=a0-memory-cognee-live-test-secret
TOKENIZERS_PARALLELISM=false
A0_SET_cognee_debug_enabled=true
A0_SET_cognee_cognify_interval=1
EOF

cat > "${USR_DIR}/plugins/_model_config/config.json" <<'EOF'
{
  "allow_chat_override": false,
  "chat_model": {
    "provider": "openai",
    "name": "gpt-5.5",
    "api_base": "",
    "ctx_length": 200000,
    "ctx_history": 0.7,
    "vision": true,
    "rl_requests": 0,
    "rl_input": 0,
    "rl_output": 0,
    "kwargs": {},
    "max_embeds": 10
  },
  "utility_model": {
    "provider": "openai",
    "name": "gpt-5.4-mini",
    "api_base": "",
    "ctx_length": 100000,
    "ctx_input": 0.7,
    "rl_requests": 5000,
    "rl_input": 4000000,
    "rl_output": 4000000,
    "kwargs": {}
  },
  "embedding_model": {
    "provider": "huggingface",
    "name": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "api_base": "",
    "rl_requests": 0,
    "rl_input": 0,
    "kwargs": {}
  }
}
EOF

: > "${USR_DIR}/plugins/_memory/.toggle-0"

docker rm -f "${CONTAINER}" >/dev/null 2>&1 || true
docker run -d \
  --name "${CONTAINER}" \
  -p "${PORT}:${PORT}" \
  -v "${USR_DIR}:/a0/usr" \
  -v "${ROOT_DIR}:/a0/usr/plugins/memory_cognee" \
  -v "${USR_DIR}:${CODE_DIR}/usr" \
  -v "${ROOT_DIR}:${CODE_DIR}/usr/plugins/memory_cognee" \
  "${IMAGE}" \
  sleep infinity >/dev/null

echo "Installing memory_cognee dependencies in ${CONTAINER}..."
docker exec -e A0_CODE_DIR="${CODE_DIR}" "${CONTAINER}" sh -lc 'cd "$A0_CODE_DIR" && /opt/venv-a0/bin/python - <<'"'"'PY'"'"'
import importlib.util
from pathlib import Path
import os

hooks_path = Path(os.environ["A0_CODE_DIR"]) / "usr/plugins/memory_cognee/hooks.py"
spec = importlib.util.spec_from_file_location("memory_cognee_hooks", hooks_path)
hooks = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(hooks)
hooks.install()
PY'

echo "Starting Agent Zero on http://127.0.0.1:${PORT}..."
docker exec -d -e A0_CODE_DIR="${CODE_DIR}" -e A0_PORT="${PORT}" "${CONTAINER}" sh -lc \
  'cd "$A0_CODE_DIR" && PYTHONUNBUFFERED=1 /opt/venv-a0/bin/python run_ui.py --host 0.0.0.0 --port "$A0_PORT" --dockerized=true > /tmp/a0-memory-cognee-itest.log 2>&1'

deadline=$((SECONDS + 240))
until curl -fsS "http://127.0.0.1:${PORT}/api/health" >/dev/null 2>&1; do
  if (( SECONDS > deadline )); then
    echo "Agent Zero did not become healthy. Last logs:" >&2
    docker exec "${CONTAINER}" sh -lc 'tail -120 /tmp/a0-memory-cognee-itest.log' >&2 || true
    exit 1
  fi
  sleep 2
done

echo "Agent Zero ready: http://127.0.0.1:${PORT}"

if [[ "${mode}" == "start" ]]; then
  echo "Container: ${CONTAINER}"
  echo "Usr dir: ${USR_DIR}"
  exit 0
fi

A0_MEMORY_COGNEE_INTEGRATION=1 \
A0_INTEGRATION_BASE_URL="http://127.0.0.1:${PORT}" \
A0_MEMORY_COGNEE_FACT_COUNT="${FACT_COUNT}" \
A0_MEMORY_COGNEE_TIMEOUT="${TIMEOUT}" \
A0_MEMORY_COGNEE_EXPECT_DEBUG=1 \
python3 -m unittest tests.test_agent_zero_memory_integration

rebuild_count="$(docker exec "${CONTAINER}" sh -lc 'grep -ac "Cognee rebuild started for dataset" /tmp/a0-memory-cognee-itest.log || true')"
max_rebuilds="${A0_MEMORY_COGNEE_MAX_REBUILDS:-3}"
if (( rebuild_count > max_rebuilds )); then
  echo "Too many Cognee rebuilds during live memory flow: ${rebuild_count} > ${max_rebuilds}" >&2
  docker exec "${CONTAINER}" sh -lc 'grep -a "Cognee rebuild started for dataset\\|Cognee rebuild readiness\\|Memory search unavailable\\|Cognee session memory write failed" /tmp/a0-memory-cognee-itest.log | tail -120' >&2 || true
  exit 1
fi

docker exec "${CONTAINER}" sh -lc '
  if grep -aq "Cognee session memory write failed\\|Memory search unavailable\\|Timed out waiting for Cognee operation gate" /tmp/a0-memory-cognee-itest.log; then
    grep -a "Cognee session memory write failed\\|Memory search unavailable\\|Timed out waiting for Cognee operation gate" /tmp/a0-memory-cognee-itest.log | tail -80 >&2
    exit 1
  fi
'
