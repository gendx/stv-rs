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

//! Script to convert ballots containing nicknames to the standard format with
//! numeric indices.
//!
//! Reads from stdin and writes to stdout.

#![forbid(missing_docs, unsafe_code)]

use clap::Parser;
use log::info;
use std::io::{stdin, stdout, BufWriter, Write};
use stv_rs::{
    parse::{parse_election, ParsingOptions},
    types::Election,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();

    let cli = Cli::parse();

    info!(
        "Removal of empty ballots is {}",
        if cli.remove_empty_ballots {
            "enabled"
        } else {
            "disabled"
        }
    );

    let election = parse_election(stdin().lock(), cli.parsing_options())?;
    cli.write_election(BufWriter::new(stdout().lock()), &election)?;
    Ok(())
}

/// Script to convert ballots containing nicknames to the standard format with
/// numeric indices.
#[derive(Parser, Debug, PartialEq, Eq)]
#[command(version)]
struct Cli {
    /// Whether to remove withdrawn candidates from the ballots they appear in.
    #[arg(long, default_value_t = false)]
    remove_withdrawn_candidates: bool,

    /// Whether to remove ballots that rank no candidate. When
    /// `remove_withdrawn_candidates` is true, this also removes ballots
    /// that only rank withdrawn candidates.
    #[arg(long, default_value_t = false)]
    remove_empty_ballots: bool,

    /// Replaces the election title by the given one.
    #[arg(long)]
    set_title: Option<String>,
}

impl Cli {
    fn parsing_options(&self) -> ParsingOptions {
        ParsingOptions {
            remove_withdrawn_candidates: self.remove_withdrawn_candidates,
            remove_empty_ballots: self.remove_empty_ballots,
        }
    }

    fn write_election(
        &self,
        mut output: impl Write,
        election: &Election,
    ) -> Result<(), Box<dyn std::error::Error>> {
        writeln!(output, "{} {}", election.num_candidates, election.num_seats)?;

        if !self.remove_withdrawn_candidates
            && election
                .candidates
                .iter()
                .any(|candidate| candidate.is_withdrawn)
        {
            write!(output, "[withdrawn")?;
            for candidate in election
                .candidates
                .iter()
                .enumerate()
                .filter_map(|(i, candidate)| {
                    if candidate.is_withdrawn {
                        Some(i + 1)
                    } else {
                        None
                    }
                })
            {
                write!(output, " {candidate}")?;
            }
            writeln!(output, "]")?;
        }

        let has_tie_order = election.tie_order.iter().any(|(x, y)| x != y);
        if has_tie_order {
            let mut tie_order = vec![0; election.num_candidates];
            for (&candidate, &rank) in &election.tie_order {
                tie_order[rank] = candidate + 1;
            }

            write!(output, "[tie")?;
            for candidate in tie_order {
                assert_ne!(candidate, 0);
                write!(output, " {candidate}")?;
            }
            writeln!(output, "]")?;
        }

        for ballot in &election.ballots {
            write!(output, "{}", ballot.count)?;
            for ranking in &ballot.order {
                write!(output, " ")?;
                for (i, candidate) in ranking.iter().enumerate() {
                    if i != 0 {
                        write!(output, "=")?;
                    }
                    write!(output, "{}", candidate + 1)?;
                }
            }
            writeln!(output, " 0")?;
        }

        writeln!(output, "0")?;
        for candidate in &election.candidates {
            writeln!(output, "\"{}\"", candidate.name)?;
        }
        writeln!(
            output,
            "\"{}\"",
            self.set_title.as_ref().unwrap_or(&election.title)
        )?;

        Ok(())
    }
}
