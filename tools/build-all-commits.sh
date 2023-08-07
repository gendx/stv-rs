#!/bin/bash

set -eux

COMMITS=$@
echo "Building at commits: '${COMMITS}'"

mkdir -p bin

ITER=0
for COMMIT in ${COMMITS}
do
    ITER=$(expr ${ITER} + 1)
    printf -v INDEX "%02d" ${ITER}
    echo "[${INDEX}] Processing ${COMMIT}"

    if [ -e "bin/stv-rs-${INDEX}-${COMMIT}" ]; then
        echo "Binary already exists at 'bin/stv-rs-${INDEX}-${COMMIT}', skipping..."
    else
        git checkout ${COMMIT}
        git status
        cargo +nightly build --release
        cp target/release/stv-rs bin/stv-rs-${INDEX}-${COMMIT}
    fi
done
