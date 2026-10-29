#!/usr/bin/env bash
# Build the dooly-collective ns-3 scenario against an installed ns-3.
#
#   NS3_PREFIX  ns-3 install prefix (default ~/ns-3-dev/install, see setup_ns3.sh)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NS3_PREFIX="${NS3_PREFIX:-$HOME/ns-3-dev/install}"

if [ ! -d "$NS3_PREFIX" ]; then
    echo "ns-3 install prefix not found at $NS3_PREFIX; run ns3_backend/setup_ns3.sh first" >&2
    exit 1
fi

cmake -S "$HERE" -B "$HERE/build" -G Ninja -DCMAKE_PREFIX_PATH="$NS3_PREFIX" \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
cmake --build "$HERE/build"
echo "built $HERE/build/dooly-collective"
