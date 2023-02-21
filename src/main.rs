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

#![deny(missing_docs)]
#![forbid(unsafe_code)]

use clap::Parser;
use num::{BigInt, BigRational};
use std::io;
use stv_rs::{
    arithmetic::{ApproxRational, BigFixedDecimal9, FixedDecimal9},
    meek::State,
    parse::parse_election,
    types::Election,
};

/// Rust implementation of Single Transferable Vote counting.
#[derive(Parser, Debug, PartialEq, Eq)]
struct Cli {
    /// Package name to show in the election report.
    #[arg(long, default_value = "Implementation: STV-rs")]
    package_name: String,

    /// Base-10 logarithm of the "omega" value, i.e. `omega = 10^omega_exponent`.
    #[arg(long, default_value_t = 6)]
    omega_exponent: usize,

    /// Arithmetic to use.
    #[arg(long, value_enum)]
    arithmetic: Arithmetic,

    /// Enable parallel ballot counting based on the rayon crate.
    #[arg(long, action = clap::ArgAction::Set, default_value = "true")]
    parallel: bool,
}

/// Arithmetic for rational numbers.
#[derive(clap::ValueEnum, Clone, Debug, PartialEq, Eq)]
enum Arithmetic {
    /// Fixed-point precisions with 9 decimals.
    Fixed9,
    /// Fixed-point precisions with 9 decimals, backed by a [`BigInt`].
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
    fn run(self, election: &Election) -> io::Result<()> {
        match self.arithmetic {
            Arithmetic::Fixed9 => State::<i64, FixedDecimal9>::stv_droop(
                &mut io::stdout().lock(),
                election,
                &self.package_name,
                self.omega_exponent,
                self.parallel,
            )?,
            Arithmetic::Bigfixed9 => State::<BigInt, BigFixedDecimal9>::stv_droop(
                &mut io::stdout().lock(),
                election,
                &self.package_name,
                self.omega_exponent,
                self.parallel,
            )?,
            Arithmetic::Exact => State::<BigInt, BigRational>::stv_droop(
                &mut io::stdout().lock(),
                election,
                &self.package_name,
                self.omega_exponent,
                self.parallel,
            )?,
            Arithmetic::Approx => State::<BigInt, ApproxRational>::stv_droop(
                &mut io::stdout().lock(),
                election,
                &self.package_name,
                self.omega_exponent,
                self.parallel,
            )?,
            Arithmetic::Float64 => State::<f64, f64>::stv_droop(
                &mut io::stdout().lock(),
                election,
                &self.package_name,
                self.omega_exponent,
                self.parallel,
            )?,
        };
        Ok(())
    }
}

fn main() {
    env_logger::init();

    let cli = Cli::parse();

    let election = parse_election(io::stdin().lock()).unwrap();

    cli.run(&election).unwrap();
}

#[cfg(test)]
mod test {
    use super::*;
    use clap::error::ErrorKind;

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
                package_name: "Implementation: STV-rs".to_owned(),
                omega_exponent: 6,
                arithmetic: Arithmetic::Fixed9,
                parallel: true
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
            "--parallel=false",
        ])
        .unwrap();
        assert_eq!(
            cli,
            Cli {
                package_name: "foo bar".to_owned(),
                omega_exponent: 42,
                arithmetic: Arithmetic::Float64,
                parallel: false
            }
        );
    }

    #[test]
    fn test_parse_full_spaces() {
        #[rustfmt::skip]
        let cli = Cli::try_parse_from([
            "stv-rs",
            "--arithmetic", "float64",
            "--package-name", "foo bar",
            "--omega-exponent", "42",
            "--parallel", "false",
        ])
        .unwrap();
        assert_eq!(
            cli,
            Cli {
                package_name: "foo bar".to_owned(),
                omega_exponent: 42,
                arithmetic: Arithmetic::Float64,
                parallel: false
            }
        );
    }
}
