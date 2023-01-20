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

use clap::Parser;
use num::{BigInt, BigRational};
use stv_rs::{
    arithmetic::{ApproxRational, BigFixedDecimal9, FixedDecimal9},
    meek::State,
    parse::parse_election,
};

#[derive(Parser)]
struct Cli {
    #[arg(long, default_value = "Implementation: STV-rs")]
    package_name: String,

    #[arg(long, default_value_t = 6)]
    omega_exponent: usize,

    #[arg(long, value_enum)]
    arithmetic: Arithmetic,

    /// Enable parallel ballot counting based on the rayon crate.
    #[arg(long, action = clap::ArgAction::Set, default_value = "true")]
    parallel: bool,
}

#[derive(clap::ValueEnum, Clone)]
enum Arithmetic {
    Fixed9,
    Bigfixed9,
    Exact,
    Approx,
    Float64,
}

fn main() {
    env_logger::init();

    let cli = Cli::parse();

    let election = parse_election(std::io::stdin().lock()).unwrap();

    match cli.arithmetic {
        Arithmetic::Fixed9 => State::<i64, FixedDecimal9>::stv_droop(
            &election,
            &cli.package_name,
            cli.omega_exponent,
            cli.parallel,
        ),
        Arithmetic::Bigfixed9 => State::<BigInt, BigFixedDecimal9>::stv_droop(
            &election,
            &cli.package_name,
            cli.omega_exponent,
            cli.parallel,
        ),
        Arithmetic::Exact => State::<BigInt, BigRational>::stv_droop(
            &election,
            &cli.package_name,
            cli.omega_exponent,
            cli.parallel,
        ),
        Arithmetic::Approx => State::<BigInt, ApproxRational>::stv_droop(
            &election,
            &cli.package_name,
            cli.omega_exponent,
            cli.parallel,
        ),
        Arithmetic::Float64 => State::<f64, f64>::stv_droop(
            &election,
            &cli.package_name,
            cli.omega_exponent,
            cli.parallel,
        ),
    };
}
