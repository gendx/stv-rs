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

//! Script to re-order ballots according to various sorting strategies.
//!
//! Reads from stdin and writes to stdout.

#![forbid(missing_docs, unsafe_code)]

use clap::{Parser, ValueEnum};
use log::{debug, info};
use std::io::{stdin, stdout, BufWriter, Write};
use stv_rs::{
    blt::{write_blt, CandidateFormat, WriteTieOrder},
    parse::{parse_election, ParsingOptions},
    types::Election,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();

    let cli = Cli::parse();

    info!("Sorting strategy is {:?}", cli.strategy);

    let election = parse_election(
        stdin().lock(),
        ParsingOptions {
            remove_withdrawn_candidates: false,
            remove_empty_ballots: false,
        },
    )?;
    cli.write_election(BufWriter::new(stdout().lock()), election)?;
    Ok(())
}

/// Script to re-order ballots according to various sorting strategies.
#[derive(Parser, Debug, PartialEq, Eq)]
#[command(version)]
struct Cli {
    /// Sorting strategy of ballots.
    #[arg(long, value_enum)]
    strategy: Strategy,

    /// Replaces the election title by the given one.
    #[arg(long)]
    set_title: Option<String>,
}

/// Sorting strategy of ballots.
#[derive(ValueEnum, Clone, Debug, PartialEq, Eq)]
enum Strategy {
    /// Sort ballots in increasing order of weight, where each ballot's weight
    /// is the product of the rank lengths.
    Product,
    /// Sort ballots in lexicographical order of rank lengths.
    Lexicographic,
    /// Sort ballots in increasing order of weight (where each ballot's weight
    /// is the product of the rank lengths), then lexicographically by the rank
    /// lengths among ballots of the same weight.
    LexicoProduct,
}

impl Cli {
    fn write_election(
        self,
        mut output: impl Write,
        mut election: Election,
    ) -> Result<(), Box<dyn std::error::Error>> {
        if let Some(title) = self.set_title {
            election.title = title;
        }

        match self.strategy {
            Strategy::Product => {
                election
                    .ballots
                    .sort_by_cached_key(|b| b.order().map(|rank| rank.len()).product::<usize>());
            }
            Strategy::Lexicographic => {
                election.ballots.sort_by(|a, b| {
                    let ita = a.order().map(|rank| rank.len());
                    let itb = b.order().map(|rank| rank.len());
                    ita.cmp(itb)
                });
            }
            Strategy::LexicoProduct => {
                election.ballots.sort_by(|a, b| {
                    let proda = a.order().map(|rank| rank.len()).product::<usize>();
                    let prodb = b.order().map(|rank| rank.len()).product::<usize>();
                    let ita = a.order().map(|rank| rank.len());
                    let itb = b.order().map(|rank| rank.len());
                    proda.cmp(&prodb).then(ita.cmp(itb))
                });
            }
        }

        for (i, ballot) in election.ballots.iter().enumerate() {
            debug!(
                "Ballot #{i}: product = {}, lexicographic = {:?}",
                ballot.order().map(|rank| rank.len()).product::<usize>(),
                ballot.order().map(|rank| rank.len()).collect::<Vec<_>>()
            );
        }

        write_blt(
            &mut output,
            &election,
            WriteTieOrder::OnlyNonTrivial,
            CandidateFormat::Nicknames,
        )?;
        Ok(())
    }
}
