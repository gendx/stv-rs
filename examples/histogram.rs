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

//! Script to generate histograms of the ranked positions of each candidate.
//!
//! Reads from stdin and writes to stdout.

#![forbid(missing_docs, unsafe_code)]

use num::{BigInt, BigRational, ToPrimitive, Zero};
use std::io::{stdin, stdout, BufWriter, Write};
use stv_rs::{
    parse::{parse_election, ParsingOptions},
    types::{BallotView, Election},
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let election = parse_election(
        stdin().lock(),
        ParsingOptions {
            remove_withdrawn_candidates: false,
            remove_empty_ballots: false,
            optimize_layout: false,
        },
    )?;
    write_histograms(BufWriter::new(stdout().lock()), &election)?;
    Ok(())
}

fn write_histograms(
    mut output: impl Write,
    election: &Election,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut histograms =
        vec![vec![BigRational::zero(); election.num_candidates]; election.num_candidates];

    for ballot in election.ballots() {
        let mut index = 0;
        for ranking in ballot.order() {
            let ranking_len = ranking.len();
            let weight = BigRational::new(BigInt::from(ballot.count()), BigInt::from(ranking_len));
            for &candidate in ranking {
                for i in 0..ranking_len {
                    histograms[candidate.into()][index + i] += &weight;
                }
            }
            index += ranking_len;
        }
    }

    write!(output, "rank")?;
    for i in 0..election.num_candidates {
        write!(output, ",{}", i + 1)?;
    }
    writeln!(output)?;

    for (i, hist) in histograms.iter().enumerate() {
        write!(output, "{}", election.candidates[i].name)?;
        for x in hist {
            if x.is_integer() {
                write!(output, ",{x}")?;
            } else {
                write!(output, ",{}", x.to_f64().unwrap_or(f64::NAN))?;
            }
        }
        writeln!(output)?;
    }

    Ok(())
}

#[cfg(test)]
mod test {
    use super::*;
    use stv_rs::types::{Ballot, Candidate, ElectionBuilder};

    fn new_election_builder() -> ElectionBuilder {
        Election::builder().title("Title").num_seats(2).candidates([
            Candidate::new("apple", false),
            Candidate::new("banana", false),
            Candidate::new("cherry", false),
            Candidate::new("date", false),
        ])
    }

    #[test]
    fn test_histogram_simple() {
        let election = new_election_builder()
            .ballots([
                Ballot::new(1, [vec![0]]),
                Ballot::new(1, [vec![0], vec![1]]),
                Ballot::new(1, [vec![0], vec![1], vec![2]]),
                Ballot::new(1, [vec![0], vec![1], vec![2], vec![3]]),
            ])
            .build();

        let mut buf = Vec::new();
        write_histograms(&mut buf, &election).unwrap();

        assert_eq!(
            std::str::from_utf8(&buf).unwrap(),
            r"rank,1,2,3,4
Apple,4,0,0,0
Banana,0,3,0,0
Cherry,0,0,2,0
Date,0,0,0,1
"
        );
    }

    #[test]
    fn test_histogram_equal_ranks() {
        let election = new_election_builder()
            .ballots([Ballot::new(3, [vec![0], vec![1, 2, 3]])])
            .build();
        let mut buf = Vec::new();
        write_histograms(&mut buf, &election).unwrap();
        assert_eq!(
            std::str::from_utf8(&buf).unwrap(),
            r"rank,1,2,3,4
Apple,3,0,0,0
Banana,0,1,1,1
Cherry,0,1,1,1
Date,0,1,1,1
"
        );

        let election = new_election_builder()
            .ballots([Ballot::new(1, [vec![2, 1], vec![0, 3]])])
            .build();
        let mut buf = Vec::new();
        write_histograms(&mut buf, &election).unwrap();
        assert_eq!(
            std::str::from_utf8(&buf).unwrap(),
            r"rank,1,2,3,4
Apple,0,0,0.5,0.5
Banana,0.5,0.5,0,0
Cherry,0.5,0.5,0,0
Date,0,0,0.5,0.5
"
        );
    }

    #[test]
    fn test_histogram_mix() {
        let election = new_election_builder()
            .ballots([
                Ballot::new(1, [vec![0]]),
                Ballot::new(2, [vec![2, 1], vec![3]]),
                Ballot::new(3, [vec![1, 2]]),
                Ballot::new(4, [vec![3, 1], vec![0, 2]]),
                Ballot::new(5, [vec![1, 2, 3, 0]]),
            ])
            .build();

        let mut buf = Vec::new();
        write_histograms(&mut buf, &election).unwrap();

        assert_eq!(
            std::str::from_utf8(&buf).unwrap(),
            r"rank,1,2,3,4
Apple,2.25,1.25,3.25,3.25
Banana,5.75,5.75,1.25,1.25
Cherry,3.75,3.75,3.25,3.25
Date,3.25,3.25,3.25,1.25
"
        );
    }
}
