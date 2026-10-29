#!/usr/bin/env bash
# Clone, configure, build and install a minimal ns-3 for the DoolySim network backend.
#
# Only the core, network and point-to-point modules are built; that is all the
# collective simulator needs and it keeps the build to a few minutes.
#
# Environment overrides:
#   NS3_VERSION  git tag to build            (default ns-3.49)
#   NS3_DIR      where to clone ns-3         (default ~/ns-3-dev)
#   NS3_PREFIX   where to install ns-3       (default $NS3_DIR/install)
#   JOBS         parallel compile jobs       (default nproc)
set -euo pipefail

NS3_VERSION="${NS3_VERSION:-ns-3.49}"
NS3_DIR="${NS3_DIR:-$HOME/ns-3-dev}"
NS3_PREFIX="${NS3_PREFIX:-$NS3_DIR/install}"
JOBS="${JOBS:-$(nproc)}"

if [ ! -d "$NS3_DIR/.git" ]; then
    git clone --depth 1 --branch "$NS3_VERSION" https://gitlab.com/nsnam/ns-3-dev.git "$NS3_DIR"
fi

cd "$NS3_DIR"
./ns3 configure \
    --build-profile=default \
    --enable-modules=core,network,point-to-point \
    --disable-examples --disable-tests --disable-werror \
    --prefix="$NS3_PREFIX"
./ns3 build -j "$JOBS"
./ns3 install

echo
echo "ns-3 $NS3_VERSION installed to $NS3_PREFIX"
echo "Next: NS3_PREFIX=$NS3_PREFIX ns3_backend/build.sh"
