# STV test cases for Meek's method

This folder contains ballot files to test implementations of Meek's method for
Single Transferable Vote.

It is organized as follows.

-   `vegetables.blt`: simple ballot file to illustrate Meek's method.
-   `crashes/`: input files that trigger a crash for
    [Droop.py](https://github.com/jklundell/droop) (v0.14).
-   `equal_preference/`: simple cases of ballots containing candidates ranked
    equally, to observe the effect of various counting methods on the election
    outcome.
-   `numeric/`: ballot files in numeric format, without nicknames.
-   `random/`: ballot files created using various random distributions, intended
    for load testing and benchmarking. These were created by the
    [random_ballots.rs](https://github.com/gendx/stv-rs/blob/main/examples/random_ballots.rs)
    example in this repository.
-   `recursive_descent/`: ballot files that show how the recursive descent
    implemented by [Droop.py](https://github.com/jklundell/droop) (v0.14)
    influences the results.
-   `ties/`: simple cases where ties between candidates need to be broken, and
    how the explicit tie-break order affects the election outcome.
