#!/bin/bash

set -eux

COMMITS=( "$@" )

if (( ${#COMMITS[@]} == 0 )); then
    echo "Please provide a non-empty list of commits as arguments."
    exit 42
fi

echo "Building at commits: " "${COMMITS[@]}"

mkdir -p bin

ITER=0
for COMMIT in "${COMMITS[@]}"
do
    ITER=$((ITER + 1))
    printf -v INDEX "%02d" "${ITER}"
    echo "[${INDEX}] Processing ${COMMIT}"

    if [ -e "bin/stv-rs-${INDEX}-${COMMIT}" ]; then
        echo "Binary already exists at 'bin/stv-rs-${INDEX}-${COMMIT}', skipping..."
    else
        if [ ! -e "bin/stv-rs-${COMMIT}" ]; then
            echo "Compiling 'bin/stv-rs-${COMMIT}'"
            git checkout "${COMMIT}"
            git status
            cargo +nightly build --release
            cp target/release/stv-rs "bin/stv-rs-${COMMIT}"
        fi
        ln -s "stv-rs-${COMMIT}" "bin/stv-rs-${INDEX}-${COMMIT}"
    fi
done
