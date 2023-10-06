# STV test cases for Meek's method

This folder contains ballot files and corresponding transcripts to test various
implementations of Single Transferable Vote.

It is organized as follows.

-   `ballots/`: input ballot files.
-   `histogram/`: histograms of the rankings of each candidate derived from the
    ballots. These were created by the
    [histogram.rs](https://github.com/gendx/stv-rs/blob/main/examples/histogram.rs)
    example in this repository.
-   `meek/`: election transcripts using Meek's method of STV.
-   `numeric/`: ballots converted into numeric format (without nicknames). These
    were created by the
    [remove_nicknames.rs](https://github.com/gendx/stv-rs/blob/main/examples/remove_nicknames.rs)
    example in this repository.
-   `plurality_block_voting/`: election transcripts using plurality block
    voting.
-   `shuffle_ballots/`: ballot files where ballots are ordered following various
    heuristics (to test performance).

In detail, the following ballot files are available.

-   `vegetables.blt`: simple ballot file to illustrate Meek's method.
-   `bigint/`: input files that require a big integer implementation to count
    correctly.
-   `crashes/`: input files that trigger a crash for
    [Droop.py](https://github.com/jklundell/droop) (v0.14).
-   `equal_preference/`: simple cases of ballots containing candidates ranked
    equally, to observe the effect of various counting methods on the election
    outcome.
-   `negative_surplus/`: ballot file that causes an elected candidate to have
    fewer votes than the threshold.
-   `random/`: ballot files created using various random distributions, intended
    for load testing and benchmarking. These were created by the
    [random_ballots.rs](https://github.com/gendx/stv-rs/blob/main/examples/random_ballots.rs)
    example in this repository.
-   `recursive_descent/`: ballot files that show how the recursive descent
    implemented by [Droop.py](https://github.com/jklundell/droop) (v0.14)
    influences the results.
-   `ties/`: simple cases where ties between candidates need to be broken, and
    how the explicit tie-break order affects the election outcome.
