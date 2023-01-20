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

//! Module to parse STV ballot files.

use crate::types::{Ballot, Candidate, Election};
use log::{info, trace, warn};
use regex::Regex;
use std::collections::{HashMap, HashSet};
use std::io::BufRead;

// TODO: Remove unwrap()s.
/// Parses a ballot file into an election input.
pub fn parse_election(input: impl BufRead) -> Result<Election, Box<dyn std::error::Error>> {
    let re_count = Regex::new(r"^([0-9]+) ([0-9]+)$").unwrap();
    let re_option = Regex::new(r"^\[[a-z]+(?: [a-z]+)+\]$").unwrap();
    let re_ballot = Regex::new(r"^([0-9]+) ([a-z= ]+) 0$").unwrap();

    let mut lines = input.lines().peekable();

    let header = lines.next().unwrap().unwrap();
    let cap_count = re_count.captures(&header).unwrap();
    let num_candidates = cap_count.get(1).unwrap().as_str().parse::<usize>().unwrap();
    let num_seats = cap_count.get(2).unwrap().as_str().parse::<usize>().unwrap();

    info!("{num_seats} seats / {num_candidates} candidates");

    // Parse the options
    let mut nicknames = None;
    let mut withdrawn: HashSet<String> = HashSet::new();
    loop {
        if let Some(line) = lines.peek() {
            let line = line.as_ref().unwrap();
            if re_option.is_match(line) {
                let mut items = line[1..line.len() - 1].split(' ');
                let title = items.next().unwrap();

                // TODO: parse ties.
                match title {
                    "nick" => {
                        let values = items.map(|x| x.to_owned()).collect::<Vec<String>>();
                        info!("Nicknames: {values:?}");
                        nicknames = Some(values);
                    }
                    "withdrawn" => {
                        withdrawn = items.map(|x| x.to_owned()).collect::<HashSet<String>>();
                        info!("Withdrawn: {withdrawn:?}");
                    }
                    "tie" => {
                        let values = items.collect::<Vec<_>>();
                        info!("Tie-break order: {values:?}");
                    }
                    _ => warn!("Unknown option: {title}"),
                }

                lines.next();
                continue;
            }
            break;
        }
        break;
    }

    let nicknames: Vec<String> = nicknames.unwrap();
    info!("Candidates (by nickname): {nicknames:?}");
    assert_eq!(nicknames.len(), num_candidates);

    let hash_nicknames: HashMap<&str, usize> =
        nicknames.iter().enumerate().map(|(i, c)| (c.as_str(), i)).collect();

    let mut ballots = Vec::new();
    loop {
        let line = lines.next().unwrap().unwrap();
        if line == "0" {
            break;
        }
        match re_ballot.captures(&line) {
            Some(cap_ballots) => {
                let count = cap_ballots.get(1).unwrap().as_str().parse::<usize>().unwrap();
                let order_str = cap_ballots.get(2).unwrap().as_str();
                let order: Vec<Vec<usize>> = order_str
                    .split(' ')
                    .filter_map(|level| {
                        let level_candidates: Vec<usize> = level
                            .split('=')
                            .filter_map(|candidate| {
                                if withdrawn.contains(candidate) {
                                    None
                                } else {
                                    Some(*hash_nicknames.get(candidate).unwrap())
                                }
                            })
                            .collect();
                        if level_candidates.is_empty() { None } else { Some(level_candidates) }
                    })
                    .collect();

                trace!("Parsed ballot: count {count} for {order:?}");
                if order.is_empty() {
                    warn!("Removing ballot only containing withdrawn candidates: {line}");
                } else {
                    let ballot = Ballot { count, order };
                    ballot.validate();
                    ballots.push(ballot);
                }
            }
            None => {
                warn!("Ignored line: {line:?}");
            }
        }
    }

    let num_ballots = ballots.iter().map(|b| b.count).sum::<usize>();
    info!("Number of ballots: {num_ballots}");

    let candidates: Vec<Candidate> = nicknames
        .into_iter()
        .map(|nickname| {
            let is_withdrawn = withdrawn.contains(&nickname);
            Candidate {
                name: remove_quotes(&lines.next().unwrap().unwrap()).to_string(),
                nickname,
                is_withdrawn,
            }
        })
        .collect();

    let title = remove_quotes(&lines.next().unwrap().unwrap()).to_string();
    info!("Election title: {title}");

    Ok(Election { title, num_candidates, num_seats, num_ballots, candidates, ballots })
}

/// Removes the leading and trailing quotes. The input string must start with a
/// double-quote character and end with a double-quote character -- only these
/// two characters are removed.
fn remove_quotes(x: &str) -> &str {
    // TODO: Implement a more robust parsing of quoted strings.
    assert!(x.len() >= 2);
    assert_eq!(*x.as_bytes().first().unwrap(), b'"');
    assert_eq!(*x.as_bytes().last().unwrap(), b'"');
    &x[1..x.len() - 1]
}
