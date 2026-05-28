#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

A0_MEMORY_COGNEE_INTEGRATION=0 python3 -m unittest discover -s tests

live_mode="${A0_MEMORY_COGNEE_LIVE_PRECOMMIT:-auto}"
case "${live_mode}" in
  0|false|False|no|No)
    echo "Skipping live Agent Zero integration test."
    exit 0
    ;;
  1|true|True|yes|Yes)
    tests/live/run_agent_zero_memory_integration.sh test
    exit 0
    ;;
  auto)
    if [[ -f tests/live/.env.local ]]; then
      tests/live/run_agent_zero_memory_integration.sh test
    else
      echo "Skipping live Agent Zero integration test; tests/live/.env.local not found."
    fi
    ;;
  *)
    echo "Invalid A0_MEMORY_COGNEE_LIVE_PRECOMMIT=${live_mode}" >&2
    exit 2
    ;;
esac
