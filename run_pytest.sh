#!/bin/bash
# Pytest runner with ReCoN mode parity vs run_experiments.sh (strict by default).

set -euo pipefail

if [[ -z "${RECON_STRICT+x}" ]]; then
  export RECON_STRICT=1
fi

RECON_MODE="${RECON_MODE:-strict}"
case "${RECON_STRICT,,}" in
  1|true|yes|on)
    RECON_MODE="strict"
    ;;
  0|false|no|off)
    RECON_MODE="compat"
    ;;
esac

echo "ReCoN mode: ${RECON_MODE}"

if [[ -z "${PYTHONPATH+x}" ]]; then
  export PYTHONPATH=.
elif [[ ":${PYTHONPATH}:" != *":.:"* ]]; then
  export PYTHONPATH=".:${PYTHONPATH}"
fi

pytest "$@"
