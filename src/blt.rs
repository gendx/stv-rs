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

//! Utilities to write elections into the BLT format.

use crate::types::{Candidate, Election};
use std::io::{self, Write};

/// Policy to write the tie order.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum WriteTieOrder {
    /// Always write the tie order, even if it is trivial.
    Always,
    /// Never write the tie order, but panics if it is non-trivial.
    Never,
    /// Only write the tie order if it is non-trivial.
    OnlyNonTrivial,
}

/// Whether to write candidates using their nicknames or in numeric format.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum CandidateFormat {
    /// Use nicknames.
    Nicknames,
    /// Use numeric format.
    Numeric,
}

/// Serializes an election into the BLT format.
pub fn write_blt(
    output: &mut impl Write,
    election: &Election,
    write_tie_order: WriteTieOrder,
    candidate_format: CandidateFormat,
) -> io::Result<()> {
    let has_non_trivial_tie_order = election
        .tie_order
        .iter()
        .any(|(candidate, order)| candidate != order);
    if write_tie_order == WriteTieOrder::Never && has_non_trivial_tie_order {
        panic!("Writing the tie order is disabled, but the tie order is non-trivial");
    }

    writeln!(output, "{} {}", election.num_candidates, election.num_seats)?;

    if candidate_format == CandidateFormat::Nicknames {
        write!(output, "[nick")?;
        for candidate in &election.candidates {
            write!(output, " {}", candidate.nickname)?;
        }
        writeln!(output, "]")?;
    }

    if election
        .candidates
        .iter()
        .any(|candidate| candidate.is_withdrawn)
    {
        write!(output, "[withdrawn")?;
        for (i, candidate) in election
            .candidates
            .iter()
            .enumerate()
            .filter(|(_, candidate)| candidate.is_withdrawn)
        {
            write!(output, " ")?;
            write_candidate(output, i, candidate, candidate_format)?;
        }
        writeln!(output, "]")?;
    }

    let write_tie_order = match write_tie_order {
        WriteTieOrder::Always => true,
        WriteTieOrder::Never => false,
        WriteTieOrder::OnlyNonTrivial => has_non_trivial_tie_order,
    };
    if write_tie_order {
        let mut tie_order = vec![!0; election.num_candidates];
        for (&candidate, &rank) in &election.tie_order {
            tie_order[rank] = candidate;
        }

        write!(output, "[tie")?;
        for candidate in tie_order {
            assert_ne!(candidate, !0);
            write!(output, " ")?;
            write_candidate(
                output,
                candidate,
                &election.candidates[candidate],
                candidate_format,
            )?;
        }
        writeln!(output, "]")?;
    }

    for ballot in &election.ballots {
        write!(output, "{}", ballot.count)?;
        for ranking in &ballot.order {
            write!(output, " ")?;
            for (i, &candidate) in ranking.iter().enumerate() {
                if i != 0 {
                    write!(output, "=")?;
                }
                write_candidate(
                    output,
                    candidate,
                    &election.candidates[candidate],
                    candidate_format,
                )?;
            }
        }
        writeln!(output, " 0")?;
    }

    writeln!(output, "0")?;
    for candidate in &election.candidates {
        writeln!(output, "\"{}\"", candidate.name)?;
    }
    writeln!(output, "\"{}\"", election.title)?;

    Ok(())
}

fn write_candidate(
    output: &mut impl Write,
    index: usize,
    candidate: &Candidate,
    candidate_format: CandidateFormat,
) -> io::Result<()> {
    match candidate_format {
        CandidateFormat::Nicknames => write!(output, "{}", candidate.nickname),
        CandidateFormat::Numeric => write!(output, "{}", index + 1),
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::types::{Ballot, Candidate, ElectionBuilder};

    fn test_election_builder() -> ElectionBuilder {
        Election::builder()
            .title("Vegetable contest")
            .num_seats(2)
            .candidates([
                Candidate::new("apple", false),
                Candidate::new("banana", false),
                Candidate::new("cherry", false),
                Candidate::new("date", false),
            ])
            .ballots([
                Ballot::new(1, [vec![0]]),
                Ballot::new(2, [vec![2, 1], vec![3]]),
                Ballot::new(3, [vec![1, 2]]),
                Ballot::new(4, [vec![3, 1], vec![0, 2]]),
                Ballot::new(5, [vec![1, 2, 3, 0]]),
            ])
    }

    #[test]
    fn test_write_blt_with_ties_trivial_ties() {
        let mut buf = Vec::new();
        write_blt(
            &mut buf,
            &test_election_builder().build(),
            WriteTieOrder::Always,
            CandidateFormat::Nicknames,
        )
        .unwrap();

        assert_eq!(
            std::str::from_utf8(&buf).unwrap(),
            r#"4 2
[nick apple banana cherry date]
[tie apple banana cherry date]
1 apple 0
2 cherry=banana date 0
3 banana=cherry 0
4 date=banana apple=cherry 0
5 banana=cherry=date=apple 0
0
"Apple"
"Banana"
"Cherry"
"Date"
"Vegetable contest"
"#
        );
    }

    #[test]
    fn test_write_blt_with_ties_has_ties() {
        let mut buf = Vec::new();
        write_blt(
            &mut buf,
            &test_election_builder().tie_order([1, 3, 2, 0]).build(),
            WriteTieOrder::Always,
            CandidateFormat::Nicknames,
        )
        .unwrap();

        assert_eq!(
            std::str::from_utf8(&buf).unwrap(),
            r#"4 2
[nick apple banana cherry date]
[tie banana date cherry apple]
1 apple 0
2 cherry=banana date 0
3 banana=cherry 0
4 date=banana apple=cherry 0
5 banana=cherry=date=apple 0
0
"Apple"
"Banana"
"Cherry"
"Date"
"Vegetable contest"
"#
        );
    }

    #[test]
    fn test_write_blt_no_ties_trivial_ties() {
        let mut buf = Vec::new();
        write_blt(
            &mut buf,
            &test_election_builder().build(),
            WriteTieOrder::Never,
            CandidateFormat::Nicknames,
        )
        .unwrap();

        assert_eq!(
            std::str::from_utf8(&buf).unwrap(),
            r#"4 2
[nick apple banana cherry date]
1 apple 0
2 cherry=banana date 0
3 banana=cherry 0
4 date=banana apple=cherry 0
5 banana=cherry=date=apple 0
0
"Apple"
"Banana"
"Cherry"
"Date"
"Vegetable contest"
"#
        );
    }

    #[test]
    #[should_panic(
        expected = "Writing the tie order is disabled, but the tie order is non-trivial"
    )]
    fn test_write_blt_no_ties_has_ties() {
        let mut buf = Vec::new();
        write_blt(
            &mut buf,
            &test_election_builder().tie_order([1, 3, 2, 0]).build(),
            WriteTieOrder::Never,
            CandidateFormat::Nicknames,
        )
        .unwrap();

        assert_eq!(
            std::str::from_utf8(&buf).unwrap(),
            r#"4 2
[nick apple banana cherry date]
[tie banana date cherry apple]
1 apple 0
2 cherry=banana date 0
3 banana=cherry 0
4 date=banana apple=cherry 0
5 banana=cherry=date=apple 0
0
"Apple"
"Banana"
"Cherry"
"Date"
"Vegetable contest"
"#
        );
    }

    #[test]
    fn test_write_blt_maybe_ties_trivial_ties() {
        let mut buf = Vec::new();
        write_blt(
            &mut buf,
            &test_election_builder().build(),
            WriteTieOrder::OnlyNonTrivial,
            CandidateFormat::Nicknames,
        )
        .unwrap();

        assert_eq!(
            std::str::from_utf8(&buf).unwrap(),
            r#"4 2
[nick apple banana cherry date]
1 apple 0
2 cherry=banana date 0
3 banana=cherry 0
4 date=banana apple=cherry 0
5 banana=cherry=date=apple 0
0
"Apple"
"Banana"
"Cherry"
"Date"
"Vegetable contest"
"#
        );
    }

    #[test]
    fn test_write_blt_maybe_ties_has_ties() {
        let mut buf = Vec::new();
        write_blt(
            &mut buf,
            &test_election_builder().tie_order([1, 3, 2, 0]).build(),
            WriteTieOrder::OnlyNonTrivial,
            CandidateFormat::Nicknames,
        )
        .unwrap();

        assert_eq!(
            std::str::from_utf8(&buf).unwrap(),
            r#"4 2
[nick apple banana cherry date]
[tie banana date cherry apple]
1 apple 0
2 cherry=banana date 0
3 banana=cherry 0
4 date=banana apple=cherry 0
5 banana=cherry=date=apple 0
0
"Apple"
"Banana"
"Cherry"
"Date"
"Vegetable contest"
"#
        );
    }

    #[test]
    fn test_write_blt_numeric_with_ties() {
        let mut buf = Vec::new();
        write_blt(
            &mut buf,
            &test_election_builder().build(),
            WriteTieOrder::Always,
            CandidateFormat::Numeric,
        )
        .unwrap();

        assert_eq!(
            std::str::from_utf8(&buf).unwrap(),
            r#"4 2
[tie 1 2 3 4]
1 1 0
2 3=2 4 0
3 2=3 0
4 4=2 1=3 0
5 2=3=4=1 0
0
"Apple"
"Banana"
"Cherry"
"Date"
"Vegetable contest"
"#
        );
    }

    #[test]
    fn test_write_blt_numeric_no_ties() {
        let mut buf = Vec::new();
        write_blt(
            &mut buf,
            &test_election_builder().build(),
            WriteTieOrder::Never,
            CandidateFormat::Numeric,
        )
        .unwrap();

        assert_eq!(
            std::str::from_utf8(&buf).unwrap(),
            r#"4 2
1 1 0
2 3=2 4 0
3 2=3 0
4 4=2 1=3 0
5 2=3=4=1 0
0
"Apple"
"Banana"
"Cherry"
"Date"
"Vegetable contest"
"#
        );
    }
}
