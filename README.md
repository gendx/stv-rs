# Single Transferable Vote implementation in Rust

[![Crate](https://img.shields.io/crates/v/stv-rs.svg?logo=rust)](https://crates.io/crates/stv-rs)
[![Documentation](https://img.shields.io/docsrs/stv-rs?logo=rust)](https://docs.rs/stv-rs)
[![Safety Dance](https://img.shields.io/badge/unsafe-forbidden-success.svg?logo=rust)](https://github.com/rust-secure-code/safety-dance/)
[![Minimum Rust 1.70.0](https://img.shields.io/badge/rust-1.70.0%2B-orange.svg?logo=rust)](https://releases.rs/docs/1.70.0/)
[![Dependencies](https://deps.rs/repo/github/gendx/stv-rs/status.svg)](https://deps.rs/repo/github/gendx/stv-rs)
[![Codecov](https://codecov.io/gh/gendx/stv-rs/branch/main/graph/badge.svg?token=JB5S8MYBZ0)](https://codecov.io/gh/gendx/stv-rs)
[![Lines of Code](https://tokei.rs/b1/github/gendx/stv-rs?category=code)](https://github.com/XAMPPRocky/tokei_rs)
[![Build Status](https://github.com/gendx/stv-rs/workflows/Build/badge.svg)](https://github.com/gendx/stv-rs/actions/workflows/build.yml)
[![Test Status](https://github.com/gendx/stv-rs/workflows/Tests/badge.svg)](https://github.com/gendx/stv-rs/actions/workflows/tests.yml)
[![Integration tests Status](https://github.com/gendx/stv-rs/workflows/Integration%20tests/badge.svg)](https://github.com/gendx/stv-rs/actions/workflows/integration.yml)

This program is an implementation of
[Single Transferable Vote](https://en.wikipedia.org/wiki/Single_transferable_vote)
in Rust. The goal is to provide vote-counting transcripts that are reproducible
with other vote-counting software, such as
[Droop.py](https://github.com/jklundell/droop).

For now, only
[Meek's counting method](https://en.wikipedia.org/wiki/Counting_single_transferable_votes#Meek)
is implemented.

You can find more details in the following blog article: [STV-rs: Single Transferable Vote implementation in Rust](https://gendignoux.com/blog/2023/03/27/single-transferable-vote.html).

## Usage

With Cargo:

```bash
$ RUST_LOG=$LOG_LEVEL cargo run \
  --release -- \
  --arithmetic $ARITHMETIC \
  --input ballots.txt \
  meek \
  --parallel=<true|false>
```

```bash
$ RUST_LOG=$LOG_LEVEL cargo run \
  --release -- \
  --arithmetic $ARITHMETIC \
  --input ballots.txt \
  meek \
  --parallel=<true|false> \
  --equalize
```

### Arithmetic implementations

You can control the arithmetic used to count votes via the `--arithmetic`
command-line flag. The following implementations are available.

-   `fixed9`: Each arithmetic operation is rounded to 9 decimal places. Rounding
    is downwards except for explicitly-marked operations (computing keep
    factors). This is backed by Rust's `i64` and therefore might overflow.
    Compiling with the `checked_i64` feature (enabled by default) will trap
    integer overflows and make the program panic, rather than continuing with
    incorrect numbers.
-   `bigfixed9`: Same as `fixed9`, but this is backed by a big integer type
    (from the [`num` crate](https://crates.io/crates/num)) and therefore won't
    overflow. On the flip side, this will be slower than `fixed9`.
-   `float64`: Use 64-bit floating-point arithmetic (Rust's `f64`). Generally
    fast but more brittle to reproduce, because the rounding introduced by
    floating-point arithmetic means that basic properties such as
    [associativity](https://en.wikipedia.org/wiki/Associative_property) and
    [distributivity](https://en.wikipedia.org/wiki/Distributive_property) don't
    hold.
-   `exact`: Use exact rational numbers without rounding. The computational
    complexity is generally too high to complete more than a few rounds.
-   `approx`: Use exact rational numbers within each STV round, but then round
    the Meek keep factors after each round, to avoid computational complexity
    explosion.

### Equalized counting

In this mode, ballots where candidates are ranked equally are counted as fairly
as possible, by simulating a superposition of all possible permutations of
equally-ranked candidates.

For example, the ballot `a b=c` becomes a superposition of `a b c` (with weight
1/2) and `a c b` (with weight 1/2). Likewise, the ballot `a b=c=d` is counted as
a superposition of 6 ballots, each with weight 1/6: `a b c d`, `a b d c`, `a c b
d`, `a c d b`, `a d b c`, `a d c b`.

### Log levels

Besides the election transcript written to the standard output (which aims to be
consistent with [Droop.py](https://github.com/jklundell/droop) for
reproducibility), you can get more details via Rust's logging capabilities,
controlled by setting the `$RUST_LOG` environment variable.

The log levels will provide the following information.

-   `info`: Print high-level results: election setup, elected/defeated
    candidates.
-   `debug`: `info` + print debug information about each STV round.
-   `trace`: `debug` + print how each ballot is counted in each round.

For more advanced logging control, please check the
[`env_logger` crate documentation](https://crates.io/crates/env_logger).

### Parallelism

To speed up the computation, you can enable parallelism via the `--parallel`
command-line flag.

The vote-counting process involves accumulating votes across all ballots,
summing the outcomes of counting each ballot. Without parallelism, this is done
by a simple serial loop over the ballots. With parallelism enabled, a parallel
loop is used instead, where each ballot is counted independently on any thread,
and the sum is computed in any order.

Because the sum is computed in an arbitrary order, it is important to use an
arithmetic where addition is
[commutative](https://en.wikipedia.org/wiki/Commutative_property) and
[associative](https://en.wikipedia.org/wiki/Associative_property), otherwise
results won't be reproducible. This excludes `float64`, as addition is not
associative.

Under the hood, the [`rayon` crate](https://crates.io/crates/rayon) is used to
automatically schedule and spread the work across available CPU cores
(map-reduce architecture).

## Other STV implementations

Here is a non-exhaustive list of STV implementations.

-   Python:
    -   [Droop.py](https://github.com/jklundell/droop).
-   Rust:
    -   [OpenTally](https://yingtongli.me/git/OpenTally)
        ([website](https://yingtongli.me/opentally/)),
    -   [wybr](https://gitlab.com/mbq/wybr)
        ([crates.io](https://crates.io/crates/wybr)),
    -   [tallystick](https://github.com/phayes/tallystick)
        ([crates.io](https://crates.io/crates/tallystick)).

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for details.

## License

Apache 2.0; see [`LICENSE`](LICENSE) for details.

## Disclaimer

This project is not an official Google project. It is not supported by Google
and Google specifically disclaims all warranties as to its quality,
merchantability, or fitness for a particular purpose.
