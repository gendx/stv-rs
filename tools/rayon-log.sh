#!/bin/bash

set -eux

RUSTFLAGS="--cfg rayon_rs_log" cargo build --release
RAYON_LOG=profile:rayon.log ./target/release/stv-rs --arithmetic bigfixed9 --input testdata/ballots/random/rand_hypergeometric.blt meek --parallel=rayon > /dev/null
