// Copyright 2023 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Command-line program for Single Transferable Vote.

#![forbid(missing_docs, unsafe_code)]

use clap::{Parser, Subcommand, ValueEnum};
use num::{BigInt, BigRational};
use std::fs::File;
use std::io::{self, stdin, stdout, BufRead, BufReader, Write};
use std::num::NonZeroUsize;
use stv_rs::{
    arithmetic::{
        ApproxRational, BigFixedDecimal9, FixedDecimal9, Integer, Integer64, IntegerRef, Rational,
        RationalRef,
    },
    cli::Parallel,
    meek::stv_droop,
    parse::{parse_election, ParsingOptions},
    pbv::plurality_block_voting,
    types::Election,
};

/// Rust implementation of Single Transferable Vote counting.
#[derive(Parser, Debug, PartialEq, Eq)]
#[command(version)]
struct Cli {
    /// Package name to show in the election report.
    #[arg(long)]
    package_name: Option<String>,

    /// Arithmetic to use.
    #[arg(long, value_enum)]
    arithmetic: Arithmetic,

    /// Input ballot file. If no input is provided, fallback to reading from
    /// stdin.
    #[arg(long)]
    input: Option<String>,

    /// Counting algorithm to use.
    #[command(subcommand)]
    algorithm: Algorithm,
}

#[derive(Subcommand, Debug, PartialEq, Eq)]
enum Algorithm {
    /// Use Meek's flavor of Single Transferable Vote.
    Meek(MeekParams),

    /// Simulate Plurality Block Voting based on ranked ballots.
    Plurality(PluralityParams),
}

#[derive(Parser, Debug, PartialEq, Eq)]
struct MeekParams {
    /// Base-10 logarithm of the "omega" value, i.e. `omega =
    /// 10^omega_exponent`.
    #[arg(long, default_value_t = 6)]
    omega_exponent: usize,

    /// Enable parallel ballot counting.
    #[arg(long, value_enum, default_value = "rayon")]
    parallel: Parallel,

    /// Explicitly specify the number of threads to use in `--parallel` modes.
    /// Ignored if parallelism is disabled.
    #[arg(long)]
    num_threads: Option<NonZeroUsize>,

    /// Disable work stealing and use a simple partitioning strategy. Ignored if
    /// `--parallel` isn't set to "custom".
    #[arg(long)]
    disable_work_stealing: bool,

    /// Enable a bug-fix in the surplus calculation, preventing it from being
    /// negative. Results may differ from Droop.py, but this prevents
    /// crashes.
    #[arg(long)]
    force_positive_surplus: bool,

    /// Enable "equalized counting".
    #[arg(long)]
    equalize: bool,
}

#[derive(Parser, Debug, PartialEq, Eq)]
struct PluralityParams {
    /// Maximum number of candidates that a ballot is allowed to rank. Defaults
    /// to the number of seats.
    #[arg(long)]
    votes_per_ballot: Option<usize>,
}

/// Arithmetic for rational numbers.
#[derive(ValueEnum, Clone, Debug, PartialEq, Eq)]
enum Arithmetic {
    /// Fixed-point with 9 decimals of precision, backed by a [`i64`].
    Fixed9,
    /// Fixed-point with 9 decimals of precision, backed by a [`BigInt`].
    Bigfixed9,
    /// Exact rational arithmetic.
    Exact,
    /// Exact rational arithmetic, with rounding of keep factors.
    Approx,
    /// 64-bit floating-point arithmetic.
    Float64,
}

impl Cli {
    /// Parses and runs an election based on the command-line parameters.
    fn run(self) -> Result<(), Box<dyn std::error::Error>> {
        match self.input {
            Some(ref filename) => {
                let file = File::open(filename).expect("Couldn't open input file");
                self.run_io(BufReader::new(file), stdout())
            }
            None => self.run_io(stdin().lock(), stdout()),
        }
    }

    /// Parses and runs an election based on the command-line parameters, using
    /// the given input/output.
    fn run_io(
        self,
        input: impl BufRead,
        mut output: impl Write,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let election = parse_election(
            input,
            ParsingOptions {
                remove_withdrawn_candidates: true,
                remove_empty_ballots: true,
            },
        )?;
        self.run_election(&election, &mut output)?;
        Ok(())
    }

    /// Run the given election based on the command-line parameters.
    fn run_election(self, election: &Election, output: &mut impl Write) -> io::Result<()> {
        match self.arithmetic {
            Arithmetic::Fixed9 => {
                self.dispatch_algorithm::<Integer64, FixedDecimal9>(election, output)
            }
            Arithmetic::Bigfixed9 => {
                self.dispatch_algorithm::<BigInt, BigFixedDecimal9>(election, output)
            }
            Arithmetic::Exact => self.dispatch_algorithm::<BigInt, BigRational>(election, output),
            Arithmetic::Approx => {
                self.dispatch_algorithm::<BigInt, ApproxRational>(election, output)
            }
            Arithmetic::Float64 => self.dispatch_algorithm::<f64, f64>(election, output),
        }
    }

    /// Run the given election based on the command-line parameters, using the
    /// arithmetic given by the generic parameters.
    fn dispatch_algorithm<I, R>(
        self,
        election: &Election,
        output: &mut impl Write,
    ) -> io::Result<()>
    where
        I: Integer + Send + Sync,
        for<'a> &'a I: IntegerRef<I>,
        R: Rational<I> + Send + Sync,
        for<'a> &'a R: RationalRef<&'a I, R>,
    {
        match self.algorithm {
            Algorithm::Meek(meek_params) => {
                let package_name: &str =
                    self.package_name
                        .as_deref()
                        .unwrap_or(if meek_params.equalize {
                            "Implementation: STV-rs (equalized counting)"
                        } else {
                            "Implementation: STV-rs"
                        });
                stv_droop::<I, R>(
                    output,
                    election,
                    package_name,
                    meek_params.omega_exponent,
                    meek_params.parallel,
                    meek_params.num_threads,
                    meek_params.disable_work_stealing,
                    meek_params.force_positive_surplus,
                    meek_params.equalize,
                )?;
            }
            Algorithm::Plurality(plurality_params) => {
                let package_name: &str = self
                    .package_name
                    .as_deref()
                    .unwrap_or("Implementation: STV-rs");
                plurality_block_voting::<I, R>(
                    output,
                    election,
                    package_name,
                    plurality_params.votes_per_ballot,
                )?;
            }
        };
        Ok(())
    }
}

fn main() {
    env_logger::builder().format_timestamp(None).init();
    Cli::parse().run().unwrap();
}

#[cfg(test)]
mod test {
    use super::*;
    use clap::error::ErrorKind;
    use std::io::Cursor;
    use stv_rs::types::{Ballot, Candidate};

    #[test]
    fn test_parse_missing_subcommand() {
        let error = Cli::try_parse_from(["stv-rs"]).unwrap_err();
        assert_eq!(
            error.kind(),
            ErrorKind::DisplayHelpOnMissingArgumentOrSubcommand
        );
    }

    #[test]
    fn test_parse_help() {
        let error = Cli::try_parse_from(["stv-rs", "--help"]).unwrap_err();
        assert_eq!(error.kind(), ErrorKind::DisplayHelp);
    }

    #[test]
    fn test_parse_meek_incomplete() {
        let error = Cli::try_parse_from(["stv-rs", "meek"]).unwrap_err();
        assert_eq!(error.kind(), ErrorKind::MissingRequiredArgument);
    }

    #[test]
    fn test_parse_meek_minimal() {
        let cli = Cli::try_parse_from(["stv-rs", "--arithmetic=fixed9", "meek"]).unwrap();
        assert_eq!(
            cli,
            Cli {
                package_name: None,
                arithmetic: Arithmetic::Fixed9,
                input: None,
                algorithm: Algorithm::Meek(MeekParams {
                    omega_exponent: 6,
                    parallel: Parallel::Rayon,
                    num_threads: None,
                    disable_work_stealing: false,
                    force_positive_surplus: false,
                    equalize: false,
                })
            }
        );
    }

    #[test]
    fn test_parse_meek_typo() {
        let error = Cli::try_parse_from(["stv-rs", "--arithmetic=Fixed9", "meek"]).unwrap_err();
        assert_eq!(error.kind(), ErrorKind::InvalidValue);
    }

    #[test]
    fn test_parse_meek_zero_threads() {
        let error =
            Cli::try_parse_from(["stv-rs", "--arithmetic=fixed9", "meek", "--num-threads=0"])
                .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::ValueValidation);
    }

    #[test]
    fn test_parse_meek_full() {
        let cli = Cli::try_parse_from([
            "stv-rs",
            "--arithmetic=float64",
            "--package-name=foo bar",
            "--input=abc def",
            "meek",
            "--omega-exponent=42",
            "--parallel=no",
            "--num-threads=37",
            "--disable-work-stealing",
            "--force-positive-surplus",
            "--equalize",
        ])
        .unwrap();
        assert_eq!(
            cli,
            Cli {
                package_name: Some("foo bar".to_owned()),
                arithmetic: Arithmetic::Float64,
                input: Some("abc def".to_owned()),
                algorithm: Algorithm::Meek(MeekParams {
                    omega_exponent: 42,
                    parallel: Parallel::No,
                    num_threads: Some(NonZeroUsize::new(37).unwrap()),
                    disable_work_stealing: true,
                    force_positive_surplus: true,
                    equalize: true,
                }),
            }
        );
    }

    #[test]
    fn test_parse_meek_full_spaces() {
        #[rustfmt::skip]
        let cli = Cli::try_parse_from([
            "stv-rs",
            "--arithmetic", "approx",
            "--package-name", "foo bar",
            "--input", "abc def",
            "meek",
            "--omega-exponent", "42",
            "--parallel", "rayon",
            "--num-threads", "37",
            "--disable-work-stealing",
            "--force-positive-surplus",
            "--equalize",
        ])
        .unwrap();
        assert_eq!(
            cli,
            Cli {
                package_name: Some("foo bar".to_owned()),
                arithmetic: Arithmetic::Approx,
                input: Some("abc def".to_owned()),
                algorithm: Algorithm::Meek(MeekParams {
                    omega_exponent: 42,
                    parallel: Parallel::Rayon,
                    num_threads: Some(NonZeroUsize::new(37).unwrap()),
                    disable_work_stealing: true,
                    force_positive_surplus: true,
                    equalize: true,
                }),
            }
        );
    }

    #[test]
    fn test_parse_plurality_incomplete() {
        let error = Cli::try_parse_from(["stv-rs", "plurality"]).unwrap_err();
        assert_eq!(error.kind(), ErrorKind::MissingRequiredArgument);
    }

    #[test]
    fn test_parse_plurality_minimal() {
        let cli = Cli::try_parse_from(["stv-rs", "--arithmetic=fixed9", "plurality"]).unwrap();
        assert_eq!(
            cli,
            Cli {
                package_name: None,
                arithmetic: Arithmetic::Fixed9,
                input: None,
                algorithm: Algorithm::Plurality(PluralityParams {
                    votes_per_ballot: None,
                })
            }
        );
    }

    #[test]
    fn test_parse_plurality_full() {
        let cli = Cli::try_parse_from([
            "stv-rs",
            "--arithmetic=float64",
            "--package-name=foo bar",
            "--input=abc def",
            "plurality",
            "--votes-per-ballot=5",
        ])
        .unwrap();
        assert_eq!(
            cli,
            Cli {
                package_name: Some("foo bar".to_owned()),
                arithmetic: Arithmetic::Float64,
                input: Some("abc def".to_owned()),
                algorithm: Algorithm::Plurality(PluralityParams {
                    votes_per_ballot: Some(5),
                })
            }
        );
    }

    /// Returns a purposefully simple [`Election`] to gather test coverage on
    /// the CLI dispatch function.
    fn make_simplest_election() -> Election {
        Election::builder()
            .title("Vegetable contest")
            .num_seats(1)
            .candidates([Candidate::new("apple", false)])
            .ballots([Ballot::new(1, [vec![0]])])
            .build()
    }

    fn make_cli_meek(arithmetic: Arithmetic, equalize: bool) -> Cli {
        Cli {
            package_name: None,
            arithmetic,
            input: None,
            algorithm: Algorithm::Meek(MeekParams {
                omega_exponent: 6,
                parallel: Parallel::No,
                num_threads: None,
                disable_work_stealing: false,
                force_positive_surplus: false,
                equalize,
            }),
        }
    }

    fn make_cli_plurality(arithmetic: Arithmetic) -> Cli {
        Cli {
            package_name: None,
            arithmetic,
            input: None,
            algorithm: Algorithm::Plurality(PluralityParams {
                votes_per_ballot: None,
            }),
        }
    }

    #[test]
    fn test_cli_run_meek() {
        let cli = make_cli_meek(Arithmetic::Fixed9, false);

        let input = r#"1 1
[nick apple]
1 apple 0
0
"Apple"
"Vegetable contest"
"#;

        let mut buf = Vec::new();
        cli.run_io(Cursor::new(&input), &mut buf).unwrap();

        assert_eq!(
            std::str::from_utf8(&buf).unwrap(),
            r"
Election: Vegetable contest

	Implementation: STV-rs
	Rule: Meek Parametric (omega = 1/10^6)
	Arithmetic: fixed-point decimal arithmetic (9 places)
	Seats: 1
	Ballots: 1
	Quota: 0.500000001
	Omega: 0.000001000

	Add eligible: Apple
Action: Begin Count
	Hopeful:  Apple (1.000000000)
	Quota: 0.500000001
	Votes: 1.000000000
	Residual: 0.000000000
	Total: 1.000000000
	Surplus: 0.000000000
Action: Elect remaining: Apple
	Elected:  Apple (1.000000000)
	Quota: 0.500000001
	Votes: 1.000000000
	Residual: 0.000000000
	Total: 1.000000000
	Surplus: 0.000000000
Action: Count Complete
	Elected:  Apple (1.000000000)
	Quota: 0.500000001
	Votes: 1.000000000
	Residual: 0.000000000
	Total: 1.000000000
	Surplus: 0.000000000

"
        );
    }

    #[test]
    fn test_cli_run_election_meek_fixed9() {
        let election = make_simplest_election();

        let cli = make_cli_meek(Arithmetic::Fixed9, false);
        let mut buf_fixed9 = Vec::new();
        cli.run_election(&election, &mut buf_fixed9).unwrap();

        let cli = make_cli_meek(Arithmetic::Bigfixed9, false);
        let mut buf_bigfixed9 = Vec::new();
        cli.run_election(&election, &mut buf_bigfixed9).unwrap();

        let expected = r"
Election: Vegetable contest

	Implementation: STV-rs
	Rule: Meek Parametric (omega = 1/10^6)
	Arithmetic: fixed-point decimal arithmetic (9 places)
	Seats: 1
	Ballots: 1
	Quota: 0.500000001
	Omega: 0.000001000

	Add eligible: Apple
Action: Begin Count
	Hopeful:  Apple (1.000000000)
	Quota: 0.500000001
	Votes: 1.000000000
	Residual: 0.000000000
	Total: 1.000000000
	Surplus: 0.000000000
Action: Elect remaining: Apple
	Elected:  Apple (1.000000000)
	Quota: 0.500000001
	Votes: 1.000000000
	Residual: 0.000000000
	Total: 1.000000000
	Surplus: 0.000000000
Action: Count Complete
	Elected:  Apple (1.000000000)
	Quota: 0.500000001
	Votes: 1.000000000
	Residual: 0.000000000
	Total: 1.000000000
	Surplus: 0.000000000

";

        assert_eq!(std::str::from_utf8(&buf_fixed9).unwrap(), expected);
        assert_eq!(std::str::from_utf8(&buf_bigfixed9).unwrap(), expected);
    }

    #[test]
    fn test_cli_run_election_meek_equalize_fixed9() {
        let election = make_simplest_election();

        let cli = make_cli_meek(Arithmetic::Fixed9, true);
        let mut buf_fixed9 = Vec::new();
        cli.run_election(&election, &mut buf_fixed9).unwrap();

        let cli = make_cli_meek(Arithmetic::Bigfixed9, true);
        let mut buf_bigfixed9 = Vec::new();
        cli.run_election(&election, &mut buf_bigfixed9).unwrap();

        let expected = r"
Election: Vegetable contest

	Implementation: STV-rs (equalized counting)
	Rule: Meek Parametric (omega = 1/10^6)
	Arithmetic: fixed-point decimal arithmetic (9 places)
	Seats: 1
	Ballots: 1
	Quota: 0.500000001
	Omega: 0.000001000

	Add eligible: Apple
Action: Begin Count
	Hopeful:  Apple (1.000000000)
	Quota: 0.500000001
	Votes: 1.000000000
	Residual: 0.000000000
	Total: 1.000000000
	Surplus: 0.000000000
Action: Elect remaining: Apple
	Elected:  Apple (1.000000000)
	Quota: 0.500000001
	Votes: 1.000000000
	Residual: 0.000000000
	Total: 1.000000000
	Surplus: 0.000000000
Action: Count Complete
	Elected:  Apple (1.000000000)
	Quota: 0.500000001
	Votes: 1.000000000
	Residual: 0.000000000
	Total: 1.000000000
	Surplus: 0.000000000

";

        assert_eq!(std::str::from_utf8(&buf_fixed9).unwrap(), expected);
        assert_eq!(std::str::from_utf8(&buf_bigfixed9).unwrap(), expected);
    }

    #[test]
    fn test_cli_run_election_meek_exact() {
        let election = make_simplest_election();

        let cli = make_cli_meek(Arithmetic::Exact, false);
        let mut buf = Vec::new();
        cli.run_election(&election, &mut buf).unwrap();

        assert_eq!(
            std::str::from_utf8(&buf).unwrap(),
            r"
Election: Vegetable contest

	Implementation: STV-rs
	Rule: Meek Parametric (omega = 1/10^6)
	Arithmetic: exact rational arithmetic
	Seats: 1
	Ballots: 1
	Quota: 1/2
	Omega: 1/1000000

	Add eligible: Apple
Action: Begin Count
	Hopeful:  Apple (1)
	Quota: 1/2
	Votes: 1
	Residual: 0
	Total: 1
	Surplus: 0
Action: Elect remaining: Apple
	Elected:  Apple (1)
	Quota: 1/2
	Votes: 1
	Residual: 0
	Total: 1
	Surplus: 0
Action: Count Complete
	Elected:  Apple (1)
	Quota: 1/2
	Votes: 1
	Residual: 0
	Total: 1
	Surplus: 0

"
        );
    }

    #[test]
    fn test_cli_run_election_meek_approx() {
        let election = make_simplest_election();

        let cli = make_cli_meek(Arithmetic::Approx, false);
        let mut buf = Vec::new();
        cli.run_election(&election, &mut buf).unwrap();

        assert_eq!(
            std::str::from_utf8(&buf).unwrap(),
            r"
Election: Vegetable contest

	Implementation: STV-rs
	Rule: Meek Parametric (omega = 1/10^6)
	Arithmetic: exact rational arithmetic with rounding of keep factors (6 decimal places)
	Seats: 1
	Ballots: 1
	Quota: 1/2
	Omega: 1/1000000

	Add eligible: Apple
Action: Begin Count
	Hopeful:  Apple (1)
	Quota: 1/2
	Votes: 1
	Residual: 0
	Total: 1
	Surplus: 0
Action: Elect remaining: Apple
	Elected:  Apple (1)
	Quota: 1/2
	Votes: 1
	Residual: 0
	Total: 1
	Surplus: 0
Action: Count Complete
	Elected:  Apple (1)
	Quota: 1/2
	Votes: 1
	Residual: 0
	Total: 1
	Surplus: 0

"
        );
    }

    #[test]
    fn test_cli_run_election_meek_float64() {
        let election = make_simplest_election();

        let cli = make_cli_meek(Arithmetic::Float64, false);
        let mut buf = Vec::new();
        cli.run_election(&election, &mut buf).unwrap();

        assert_eq!(
            std::str::from_utf8(&buf).unwrap(),
            r"
Election: Vegetable contest

	Implementation: STV-rs
	Rule: Meek Parametric (omega = 1/10^6)
	Arithmetic: 64-bit floating-point arithmetic
	Seats: 1
	Ballots: 1
	Quota: 0.5
	Omega: 0.000001

	Add eligible: Apple
Action: Begin Count
	Hopeful:  Apple (1)
	Quota: 0.5
	Votes: 1
	Residual: 0
	Total: 1
	Surplus: 0
Action: Elect remaining: Apple
	Elected:  Apple (1)
	Quota: 0.5
	Votes: 1
	Residual: 0
	Total: 1
	Surplus: 0
Action: Count Complete
	Elected:  Apple (1)
	Quota: 0.5
	Votes: 1
	Residual: 0
	Total: 1
	Surplus: 0

"
        );
    }

    #[test]
    fn test_cli_run_election_plurality_fixed9() {
        let election = make_simplest_election();

        let cli = make_cli_plurality(Arithmetic::Fixed9);
        let mut buf_fixed9 = Vec::new();
        cli.run_election(&election, &mut buf_fixed9).unwrap();

        let cli = make_cli_plurality(Arithmetic::Bigfixed9);
        let mut buf_bigfixed9 = Vec::new();
        cli.run_election(&election, &mut buf_bigfixed9).unwrap();

        let expected = r"
Election: Vegetable contest

	Implementation: STV-rs
	Rule: Simulated Plurality Block Voting
	Arithmetic: fixed-point decimal arithmetic (9 places)
	Seats: 1
	Ballots: 1
	Votes per ballot: 1

Action: Count Complete
Action: Elect: Apple
Action: Count Complete
	Elected:  Apple (1.000000000)
";

        assert_eq!(std::str::from_utf8(&buf_fixed9).unwrap(), expected);
        assert_eq!(std::str::from_utf8(&buf_bigfixed9).unwrap(), expected);
    }
}
