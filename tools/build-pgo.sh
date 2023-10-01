#!/bin/bash

set -eux

if [ -e "${HOME}/.cargo/bin/hyperfine" ]; then
    HYPERFINE_PATH=${HOME}/.cargo/bin/hyperfine
elif [ -e "/usr/bin/hyperfine" ]; then
    HYPERFINE_PATH=/usr/bin/hyperfine
else
    echo "Hyperfine is not installed. Please install it with 'apt install hyperfine' or 'cargo +nightly install hyperfine'."
    exit 42
fi

LLVM_PROFDATA=${HOME}/.rustup/toolchains/stable-x86_64-unknown-linux-gnu/lib/rustlib/x86_64-unknown-linux-gnu/bin/llvm-profdata

PGO_DATA_DIR=/tmp/pgo-data
SLEEP_SECONDS=20
ARGUMENTS="--arithmetic fixed9 --input testdata/ballots/random/rand_hypergeometric.blt meek --parallel=true"

function build_pgo() {
    # See https://doc.rust-lang.org/rustc/profile-guided-optimization.html
    # STEP 0: Make sure there is no left-over profiling data from previous runs
    rm -rf ${PGO_DATA_DIR}

    # STEP 1: Build the instrumented binaries
    RUSTFLAGS="-Cprofile-generate=${PGO_DATA_DIR}" cargo +nightly build --release

    # STEP 2: Run the instrumented binaries with some typical data
    sleep ${SLEEP_SECONDS}
    ${HYPERFINE_PATH} "./target/release/stv-rs ${ARGUMENTS} > /dev/null"

    # STEP 3: Merge the `.profraw` files into a `.profdata` file
    ${LLVM_PROFDATA} merge -o ${PGO_DATA_DIR}/merged.profdata ${PGO_DATA_DIR}

    # STEP 4: Use the `.profdata` file for guiding optimizations
    RUSTFLAGS="-Cprofile-use=${PGO_DATA_DIR}/merged.profdata -Cllvm-args=-pgo-warn-missing-function" cargo +nightly build --release
}

mkdir -p bin

cargo +nightly build --release
cp target/release/stv-rs bin/stv-rs-nopgo

build_pgo
cp target/release/stv-rs bin/stv-rs-pgo

sleep ${SLEEP_SECONDS}
${HYPERFINE_PATH} \
    "./bin/stv-rs-nopgo ${ARGUMENTS} > /dev/null" \
    "./bin/stv-rs-pgo ${ARGUMENTS} > /dev/null"
