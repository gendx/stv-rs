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

use clap::Parser;
use num::{BigInt, BigRational};
use std::fs::File;
use std::io::{self, stdin, stdout, BufReader, Write};
use stv_rs::{
    arithmetic::{ApproxRational, BigFixedDecimal9, FixedDecimal9},
    meek::stv_droop,
    parse::parse_election,
    types::Election,
};

/// Rust implementation of Single Transferable Vote counting.
#[derive(Parser, Debug, PartialEq, Eq)]
#[command(version)]
struct Cli {
    /// Package name to show in the election report.
    #[arg(long)]
    package_name: Option<String>,

    /// Base-10 logarithm of the "omega" value, i.e. `omega =
    /// 10^omega_exponent`.
    #[arg(long, default_value_t = 6)]
    omega_exponent: usize,

    /// Arithmetic to use.
    #[arg(long, value_enum)]
    arithmetic: Arithmetic,

    /// Input ballot file. If no input is provided, fallback to reading from
    /// stdin.
    #[arg(long)]
    input: Option<String>,

    /// Enable parallel ballot counting based on the rayon crate.
    #[arg(long, action = clap::ArgAction::Set, default_value = "true")]
    parallel: bool,

    /// Enable a bug-fix in the surplus calculation, preventing it from being
    /// negative. Results may differ from Droop.py, but this prevents
    /// crashes.
    #[arg(long)]
    force_positive_surplus: bool,

    /// Enable "equalized counting".
    #[arg(long)]
    equalize: bool,
}

/// Arithmetic for rational numbers.
#[derive(clap::ValueEnum, Clone, Debug, PartialEq, Eq)]
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
    /// Run the given election based on the command-line parameters.
    fn run(self, election: &Election, output: &mut impl Write) -> io::Result<()> {
        let package_name: &str = self.package_name.as_deref().unwrap_or(if self.equalize {
            "Implementation: STV-rs (equalized counting)"
        } else {
            "Implementation: STV-rs"
        });
        match self.arithmetic {
            Arithmetic::Fixed9 => stv_droop::<i64, FixedDecimal9>(
                output,
                election,
                package_name,
                self.omega_exponent,
                self.parallel,
                self.force_positive_surplus,
                self.equalize,
            )?,
            Arithmetic::Bigfixed9 => stv_droop::<BigInt, BigFixedDecimal9>(
                output,
                election,
                package_name,
                self.omega_exponent,
                self.parallel,
                self.force_positive_surplus,
                self.equalize,
            )?,
            Arithmetic::Exact => stv_droop::<BigInt, BigRational>(
                output,
                election,
                package_name,
                self.omega_exponent,
                self.parallel,
                self.force_positive_surplus,
                self.equalize,
            )?,
            Arithmetic::Approx => stv_droop::<BigInt, ApproxRational>(
                output,
                election,
                package_name,
                self.omega_exponent,
                self.parallel,
                self.force_positive_surplus,
                self.equalize,
            )?,
            Arithmetic::Float64 => stv_droop::<f64, f64>(
                output,
                election,
                package_name,
                self.omega_exponent,
                self.parallel,
                self.force_positive_surplus,
                self.equalize,
            )?,
        };
        Ok(())
    }
}

fn main() {
    env_logger::init();

    let cli = Cli::parse();

    let election = match &cli.input {
        Some(filename) => {
            let file = File::open(filename).expect("Couldn't open input file");
            parse_election(BufReader::new(file)).unwrap()
        }
        None => parse_election(stdin().lock()).unwrap(),
    };

    cli.run(&election, &mut stdout().lock()).unwrap();
}

#[cfg(test)]
mod test {
    use super::*;
    use clap::error::ErrorKind;
    use stv_rs::types::{Ballot, Candidate};

    #[test]
    fn test_parse_incomplete() {
        let error = Cli::try_parse_from(["stv-rs"]).unwrap_err();
        assert_eq!(error.kind(), ErrorKind::MissingRequiredArgument);
    }

    #[test]
    fn test_parse_help() {
        let error = Cli::try_parse_from(["stv-rs", "--help"]).unwrap_err();
        assert_eq!(error.kind(), ErrorKind::DisplayHelp);
    }

    #[test]
    fn test_parse_minimal() {
        let cli = Cli::try_parse_from(["stv-rs", "--arithmetic=fixed9"]).unwrap();
        assert_eq!(
            cli,
            Cli {
                package_name: None,
                omega_exponent: 6,
                arithmetic: Arithmetic::Fixed9,
                input: None,
                parallel: true,
                force_positive_surplus: false,
                equalize: false,
            }
        );
    }

    #[test]
    fn test_parse_typo() {
        let error = Cli::try_parse_from(["stv-rs", "--arithmetic=Fixed9"]).unwrap_err();
        assert_eq!(error.kind(), ErrorKind::InvalidValue);
    }

    #[test]
    fn test_parse_full() {
        let cli = Cli::try_parse_from([
            "stv-rs",
            "--arithmetic=float64",
            "--package-name=foo bar",
            "--omega-exponent=42",
            "--input=abc def",
            "--parallel=false",
            "--force-positive-surplus",
            "--equalize",
        ])
        .unwrap();
        assert_eq!(
            cli,
            Cli {
                package_name: Some("foo bar".to_owned()),
                omega_exponent: 42,
                arithmetic: Arithmetic::Float64,
                input: Some("abc def".to_owned()),
                parallel: false,
                force_positive_surplus: true,
                equalize: true,
            }
        );
    }

    #[test]
    fn test_parse_full_spaces() {
        #[rustfmt::skip]
        let cli = Cli::try_parse_from([
            "stv-rs",
            "--arithmetic", "approx",
            "--package-name", "foo bar",
            "--omega-exponent", "42",
            "--input", "abc def",
            "--parallel", "false",
            "--force-positive-surplus",
            "--equalize",
        ])
        .unwrap();
        assert_eq!(
            cli,
            Cli {
                package_name: Some("foo bar".to_owned()),
                omega_exponent: 42,
                arithmetic: Arithmetic::Approx,
                input: Some("abc def".to_owned()),
                parallel: false,
                force_positive_surplus: true,
                equalize: true,
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

    fn make_cli(arithmetic: Arithmetic) -> Cli {
        Cli {
            package_name: None,
            omega_exponent: 6,
            arithmetic,
            input: None,
            parallel: false,
            force_positive_surplus: false,
            equalize: false,
        }
    }

    #[test]
    fn test_cli_run_fixed9() {
        let election = make_simplest_election();

        let cli = make_cli(Arithmetic::Fixed9);
        let mut buf_fixed9 = Vec::new();
        cli.run(&election, &mut buf_fixed9).unwrap();

        let cli = make_cli(Arithmetic::Bigfixed9);
        let mut buf_bigfixed9 = Vec::new();
        cli.run(&election, &mut buf_bigfixed9).unwrap();

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
    fn test_cli_run_exact() {
        let election = make_simplest_election();

        let cli = make_cli(Arithmetic::Exact);
        let mut buf = Vec::new();
        cli.run(&election, &mut buf).unwrap();

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
    fn test_cli_run_approx() {
        let election = make_simplest_election();

        let cli = make_cli(Arithmetic::Approx);
        let mut buf = Vec::new();
        cli.run(&election, &mut buf).unwrap();

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
    fn test_cli_run_float64() {
        let election = make_simplest_election();

        let cli = make_cli(Arithmetic::Float64);
        let mut buf = Vec::new();
        cli.run(&election, &mut buf).unwrap();

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
}
