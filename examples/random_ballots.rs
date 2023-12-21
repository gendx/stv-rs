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

//! Script to generate random ballot files.

#![forbid(missing_docs, unsafe_code)]
#![feature(array_windows)]

use clap::{Parser, ValueEnum};
use rand::distributions::{Distribution, Uniform};
use rand::seq::SliceRandom;
use rand::{thread_rng, RngCore, SeedableRng};
use rand_chacha::ChaChaRng;
use rand_distr::{Bernoulli, Beta, Binomial, Geometric, Hypergeometric};
use std::collections::BTreeMap;
use std::fs::File;
use std::io::{stdout, BufWriter, Result, Write};
use stv_rs::blt::{write_blt, CandidateFormat, WriteTieOrder};
use stv_rs::types::{Ballot, Candidate, Election, ElectionBuilder};

const VEGETABLES: [&str; 20] = [
    "apple", "banana", "cherry", "date", "eggplant", "fig", "grape", "hazelnut", "jalapeno",
    "kiwi", "litchi", "mushroom", "nut", "orange", "pear", "quinoa", "radish", "soy", "tomato",
    "vanilla",
];

const MANY_CANDIDATES: [&str; 80] = [
    "apple",
    "banana",
    "cherry",
    "date",
    "eggplant",
    "fig",
    "grape",
    "hazelnut",
    "jalapeno",
    "kiwi",
    "litchi",
    "mushroom",
    "nut",
    "orange",
    "pear",
    "quinoa",
    "radish",
    "soy",
    "tomato",
    "vanilla",
    "apple1",
    "banana1",
    "cherry1",
    "date1",
    "eggplant1",
    "fig1",
    "grape1",
    "hazelnut1",
    "jalapeno1",
    "kiwi1",
    "litchi1",
    "mushroom1",
    "nut1",
    "orange1",
    "pear1",
    "quinoa1",
    "radish1",
    "soy1",
    "tomato1",
    "vanilla1",
    "apple2",
    "banana2",
    "cherry2",
    "date2",
    "eggplant2",
    "fig2",
    "grape2",
    "hazelnut2",
    "jalapeno2",
    "kiwi2",
    "litchi2",
    "mushroom2",
    "nut2",
    "orange2",
    "pear2",
    "quinoa2",
    "radish2",
    "soy2",
    "tomato2",
    "vanilla2",
    "apple3",
    "banana3",
    "cherry3",
    "date3",
    "eggplant3",
    "fig3",
    "grape3",
    "hazelnut3",
    "jalapeno3",
    "kiwi3",
    "litchi3",
    "mushroom3",
    "nut3",
    "orange3",
    "pear3",
    "quinoa3",
    "radish3",
    "soy3",
    "tomato3",
    "vanilla3",
];

fn main() -> Result<()> {
    let cli = Cli::parse();
    cli.dispatch_output()
}

/// Script to create random ballots.
#[derive(Parser, Debug, PartialEq, Eq)]
#[command(version)]
struct Cli {
    /// Election title.
    #[arg(long, default_value_t = {"Vegetable contest".to_string()})]
    title: String,

    /// Type of ballot file to create.
    #[arg(long, value_enum)]
    ballot_pattern: BallotPattern,

    /// Number of ballots to create.
    #[arg(long, default_value_t = 1000)]
    num_ballots: usize,

    /// Random seed to use.
    #[arg(long)]
    seed: Option<u64>,

    /// Output file. If nothing is provided, use stdout.
    #[arg(long)]
    output: Option<String>,
}

/// Type of ballot file to create.
#[derive(ValueEnum, Clone, Debug, PartialEq, Eq)]
enum BallotPattern {
    /// Each ballot contains 10 pairs of candidates.
    Tuples2x10,
    /// Each ballot contains 4 tuples of 5 candidates.
    Tuples5x4,
    /// 20 candidates following a geometric distribution.
    Geometric20,
    /// 20 candidates following a hypergeometric distribution.
    Hypergeometric20,
    /// 80 candidates following a hypergeometric distribution.
    Hypergeometric80,
    /// 20 candidates following a mixed distribution aiming to model real
    /// elections.
    Mixed20,
}

impl Cli {
    fn dispatch_output(&self) -> Result<()> {
        match &self.output {
            None => self.dispatch_rng(&mut stdout().lock()),
            Some(file) => self.dispatch_rng(&mut BufWriter::new(File::create(file)?)),
        }
    }

    fn dispatch_rng(&self, output: &mut impl Write) -> Result<()> {
        match self.seed {
            None => self.dispatch_pattern(output, &mut thread_rng()),
            Some(seed) => self.dispatch_pattern(output, &mut ChaChaRng::seed_from_u64(seed)),
        }
    }

    fn dispatch_pattern(&self, output: &mut impl Write, rng: &mut impl RngCore) -> Result<()> {
        match self.ballot_pattern {
            BallotPattern::Tuples2x10 => {
                write_blt_2x10(rng, output, &self.title, &VEGETABLES, self.num_ballots)
            }
            BallotPattern::Tuples5x4 => {
                write_blt_5x4(rng, output, &self.title, &VEGETABLES, self.num_ballots)
            }
            BallotPattern::Geometric20 => {
                write_blt_geometric(rng, output, &self.title, &VEGETABLES, self.num_ballots)
            }
            BallotPattern::Hypergeometric20 => {
                write_blt_hypergeometric(rng, output, &self.title, &VEGETABLES, self.num_ballots)
            }
            BallotPattern::Hypergeometric80 => write_blt_hypergeometric(
                rng,
                output,
                &self.title,
                &MANY_CANDIDATES,
                self.num_ballots,
            ),
            BallotPattern::Mixed20 => {
                write_blt_mixed(rng, output, &self.title, &VEGETABLES, self.num_ballots)
            }
        }
    }
}

fn new_election_builder(title: &str, nicknames: &[&str]) -> ElectionBuilder {
    Election::builder().title(title).num_seats(10).candidates(
        nicknames
            .iter()
            .map(|&nickname| Candidate::new(nickname, /* is_withdrawn = */ false))
            .collect::<Vec<_>>(),
    )
}

fn write_blt_2x10(
    rng: &mut impl RngCore,
    output: &mut impl Write,
    title: &str,
    nicknames: &[&str],
    ballot_count: usize,
) -> Result<()> {
    let election = new_election_builder(title, nicknames)
        .ballots(generate_tuples(rng, nicknames.len(), ballot_count, 2))
        .build();
    write_blt(
        output,
        &election,
        WriteTieOrder::Never,
        CandidateFormat::Nicknames,
    )?;
    Ok(())
}

fn write_blt_5x4(
    rng: &mut impl RngCore,
    output: &mut impl Write,
    title: &str,
    nicknames: &[&str],
    ballot_count: usize,
) -> Result<()> {
    let election = new_election_builder(title, nicknames)
        .ballots(generate_tuples(rng, nicknames.len(), ballot_count, 5))
        .build();
    write_blt(
        output,
        &election,
        WriteTieOrder::Never,
        CandidateFormat::Nicknames,
    )?;
    Ok(())
}

fn write_blt_geometric(
    rng: &mut impl RngCore,
    output: &mut impl Write,
    title: &str,
    nicknames: &[&str],
    ballot_count: usize,
) -> Result<()> {
    let election = new_election_builder(title, nicknames)
        .ballots(generate_geometric(rng, nicknames.len(), ballot_count))
        .build();
    write_blt(
        output,
        &election,
        WriteTieOrder::Never,
        CandidateFormat::Nicknames,
    )?;
    Ok(())
}

fn write_blt_hypergeometric(
    rng: &mut impl RngCore,
    output: &mut impl Write,
    title: &str,
    nicknames: &[&str],
    ballot_count: usize,
) -> Result<()> {
    let election = new_election_builder(title, nicknames)
        .ballots(generate_hypergeometric(rng, nicknames.len(), ballot_count))
        .build();
    write_blt(
        output,
        &election,
        WriteTieOrder::Never,
        CandidateFormat::Nicknames,
    )?;
    Ok(())
}

fn write_blt_mixed(
    rng: &mut impl RngCore,
    output: &mut impl Write,
    title: &str,
    nicknames: &[&str],
    ballot_count: usize,
) -> Result<()> {
    let election = new_election_builder(title, nicknames)
        .ballots(generate_mixed(rng, nicknames.len(), ballot_count))
        .build();
    write_blt(
        output,
        &election,
        WriteTieOrder::Never,
        CandidateFormat::Nicknames,
    )?;
    Ok(())
}

fn generate_geometric(
    rng: &mut impl RngCore,
    candidate_count: usize,
    ballot_count: usize,
) -> Vec<Ballot> {
    let distributions = (0..candidate_count)
        .map(|i| Geometric::new(0.3 + i as f64 / 50.0).unwrap())
        .collect::<Vec<_>>();
    generate_distributions(rng, ballot_count, &distributions)
}

fn generate_hypergeometric(
    rng: &mut impl RngCore,
    candidate_count: usize,
    ballot_count: usize,
) -> Vec<Ballot> {
    let distributions = (0..candidate_count)
        .map(|i| Hypergeometric::new(100, 50, 20 + i as u64).unwrap())
        .collect::<Vec<_>>();
    generate_distributions(rng, ballot_count, &distributions)
}

fn generate_tuples(
    rng: &mut impl RngCore,
    candidate_count: usize,
    ballot_count: usize,
    tuple_size: usize,
) -> Vec<Ballot> {
    let count_dist = Uniform::from(1..100);

    let mut ballots = Vec::new();
    for _ in 0..ballot_count {
        let count = count_dist.sample(rng);
        let order = rand::seq::index::sample(rng, candidate_count, candidate_count);
        ballots.push(Ballot::new(
            count,
            order
                .into_vec()
                .chunks(tuple_size)
                .map(|rank| rank.to_vec())
                .collect::<Vec<_>>(),
        ));
    }
    ballots
}

fn generate_distributions<D: Distribution<u64>>(
    rng: &mut impl RngCore,
    ballot_count: usize,
    distributions: &[D],
) -> Vec<Ballot> {
    let count_dist = Uniform::from(1..100);

    let mut ballots = Vec::new();
    for _ in 0..ballot_count {
        let count = count_dist.sample(rng);
        let order = order_from_distributions(rng, distributions);
        ballots.push(Ballot::new(count, order));
    }

    ballots
}

fn order_from_distributions<D: Distribution<u64>>(
    rng: &mut impl RngCore,
    distributions: &[D],
) -> Vec<Vec<usize>> {
    let mut order: BTreeMap<u64, Vec<usize>> = BTreeMap::new();
    for (i, d) in distributions.iter().enumerate() {
        let value = d.sample(rng);
        order.entry(value).or_default().push(i);
    }
    order.into_values().collect::<Vec<_>>()
}

fn generate_mixed(
    rng: &mut impl RngCore,
    candidate_count: usize,
    ballot_count: usize,
) -> Vec<Ballot> {
    // Each ballot has a 15% chance of ranking all the candidates.
    let all_ranked = Bernoulli::new(0.15).unwrap();
    // Distribution for the number of candidates ranked in a ballot.
    let rank_beta = Beta::new(/* alpha = */ 1.2, /* beta = */ 2.8).unwrap();
    // Distribution for the number of ranks that a ballot is split into.
    let cuts_beta = Beta::new(/* alpha = */ 0.6, /* beta = */ 0.4).unwrap();
    // Distributions to order candidates within the ballot.
    let hypergeometrics = (0..candidate_count)
        .map(|i| Hypergeometric::new(100, 50, 20 + i as u64).unwrap())
        .collect::<Vec<_>>();

    let mut ballots = Vec::new();
    for _ in 0..ballot_count {
        // Step 1: sample the number of candidates ranked in this ballot.
        let ranked_candidates: usize = if all_ranked.sample(rng) {
            candidate_count
        } else {
            let p = rank_beta.sample(rng);
            1 + Binomial::new((candidate_count - 2) as u64, p)
                .unwrap()
                .sample(rng) as usize
        };
        assert!(ranked_candidates != 0 && ranked_candidates <= candidate_count);

        // Step 2: sample the number of cuts within this ballot.
        let num_cuts: usize = {
            let p = cuts_beta.sample(rng);
            Binomial::new((ranked_candidates - 1) as u64, p)
                .unwrap()
                .sample(rng) as usize
        };

        // Step 3: sample the ordering of candidates in the ballot.
        let mut order = order_from_distributions(rng, &hypergeometrics);
        for rank in order.iter_mut() {
            rank.shuffle(rng);
        }
        let order: Vec<usize> = order.into_iter().flatten().collect();

        // Step 4: sample the cuts.
        let mut cuts: Vec<usize> = rand::seq::index::sample(rng, ranked_candidates - 1, num_cuts)
            .iter()
            .map(|x| x + 1)
            .collect();
        cuts.push(0);
        cuts.push(ranked_candidates);
        cuts.sort_unstable();
        assert_eq!(*cuts.first().unwrap(), 0);
        assert_eq!(*cuts.last().unwrap(), ranked_candidates);

        // Step 5: create the ballot.
        ballots.push(Ballot::new(
            1,
            cuts.array_windows::<2>().map(|&[start, end]| {
                assert!(start != end);
                let mut rank = order[start..end].to_vec();
                rank.sort_unstable();
                rank
            }),
        ));
    }

    ballots
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_blt_2x10() {
        let mut buf = Vec::new();
        let mut rng = ChaChaRng::seed_from_u64(42);
        write_blt_2x10(
            &mut rng,
            &mut buf,
            "Vegetable contest",
            &VEGETABLES,
            /* ballot_count = */ 7,
        )
        .unwrap();

        assert_eq!(
            std::str::from_utf8(&buf).unwrap(),
            r#"20 10
[nick apple banana cherry date eggplant fig grape hazelnut jalapeno kiwi litchi mushroom nut orange pear quinoa radish soy tomato vanilla]
51 jalapeno=banana radish=pear soy=tomato nut=orange hazelnut=apple mushroom=eggplant quinoa=litchi fig=date kiwi=cherry grape=vanilla 0
64 soy=eggplant kiwi=tomato fig=banana nut=radish cherry=apple jalapeno=hazelnut orange=quinoa pear=date grape=litchi mushroom=vanilla 0
92 jalapeno=grape hazelnut=vanilla nut=quinoa tomato=cherry litchi=date orange=fig radish=eggplant apple=soy mushroom=kiwi pear=banana 0
46 quinoa=vanilla kiwi=tomato pear=apple soy=hazelnut fig=nut banana=mushroom cherry=orange date=eggplant grape=jalapeno radish=litchi 0
71 radish=eggplant cherry=orange quinoa=pear banana=litchi date=fig mushroom=tomato vanilla=apple kiwi=jalapeno grape=hazelnut nut=soy 0
52 eggplant=pear vanilla=nut hazelnut=kiwi mushroom=soy cherry=radish banana=orange date=grape apple=litchi quinoa=fig jalapeno=tomato 0
73 orange=mushroom date=hazelnut banana=cherry eggplant=grape kiwi=fig quinoa=tomato soy=radish pear=litchi vanilla=jalapeno nut=apple 0
0
"Apple"
"Banana"
"Cherry"
"Date"
"Eggplant"
"Fig"
"Grape"
"Hazelnut"
"Jalapeno"
"Kiwi"
"Litchi"
"Mushroom"
"Nut"
"Orange"
"Pear"
"Quinoa"
"Radish"
"Soy"
"Tomato"
"Vanilla"
"Vegetable contest"
"#
        );
    }

    #[test]
    fn test_blt_5x4() {
        let mut buf = Vec::new();
        let mut rng = ChaChaRng::seed_from_u64(42);
        write_blt_5x4(
            &mut rng,
            &mut buf,
            "Vegetable contest",
            &VEGETABLES,
            /* ballot_count = */ 7,
        )
        .unwrap();

        assert_eq!(
            std::str::from_utf8(&buf).unwrap(),
            r#"20 10
[nick apple banana cherry date eggplant fig grape hazelnut jalapeno kiwi litchi mushroom nut orange pear quinoa radish soy tomato vanilla]
51 jalapeno=banana=radish=pear=soy tomato=nut=orange=hazelnut=apple mushroom=eggplant=quinoa=litchi=fig date=kiwi=cherry=grape=vanilla 0
64 soy=eggplant=kiwi=tomato=fig banana=nut=radish=cherry=apple jalapeno=hazelnut=orange=quinoa=pear date=grape=litchi=mushroom=vanilla 0
92 jalapeno=grape=hazelnut=vanilla=nut quinoa=tomato=cherry=litchi=date orange=fig=radish=eggplant=apple soy=mushroom=kiwi=pear=banana 0
46 quinoa=vanilla=kiwi=tomato=pear apple=soy=hazelnut=fig=nut banana=mushroom=cherry=orange=date eggplant=grape=jalapeno=radish=litchi 0
71 radish=eggplant=cherry=orange=quinoa pear=banana=litchi=date=fig mushroom=tomato=vanilla=apple=kiwi jalapeno=grape=hazelnut=nut=soy 0
52 eggplant=pear=vanilla=nut=hazelnut kiwi=mushroom=soy=cherry=radish banana=orange=date=grape=apple litchi=quinoa=fig=jalapeno=tomato 0
73 orange=mushroom=date=hazelnut=banana cherry=eggplant=grape=kiwi=fig quinoa=tomato=soy=radish=pear litchi=vanilla=jalapeno=nut=apple 0
0
"Apple"
"Banana"
"Cherry"
"Date"
"Eggplant"
"Fig"
"Grape"
"Hazelnut"
"Jalapeno"
"Kiwi"
"Litchi"
"Mushroom"
"Nut"
"Orange"
"Pear"
"Quinoa"
"Radish"
"Soy"
"Tomato"
"Vanilla"
"Vegetable contest"
"#
        );
    }

    #[test]
    fn test_blt_geometric() {
        let mut buf = Vec::new();
        let mut rng = ChaChaRng::seed_from_u64(42);
        write_blt_geometric(
            &mut rng,
            &mut buf,
            "Vegetable contest",
            &VEGETABLES,
            /* ballot_count = */ 7,
        )
        .unwrap();

        assert_eq!(
            std::str::from_utf8(&buf).unwrap(),
            r#"20 10
[nick apple banana cherry date eggplant fig grape hazelnut jalapeno kiwi litchi mushroom nut orange pear quinoa radish soy tomato vanilla]
51 cherry=date=hazelnut=jalapeno=mushroom=orange=pear=quinoa=radish=vanilla kiwi=litchi=tomato eggplant=nut=soy grape banana fig apple 0
28 apple=cherry=grape=hazelnut=jalapeno=litchi=mushroom=nut=pear=soy=tomato=vanilla kiwi=orange=quinoa=radish banana=fig date eggplant 0
47 cherry=date=hazelnut=litchi=nut=pear=quinoa=radish=tomato=vanilla grape=orange banana=fig=jalapeno=kiwi=mushroom apple=soy eggplant 0
65 fig=hazelnut=jalapeno=litchi=mushroom=nut=orange=pear=radish=soy=tomato=vanilla apple=kiwi banana=cherry=quinoa eggplant date=grape 0
9 banana=cherry=eggplant=hazelnut=jalapeno=pear=quinoa=radish=soy=tomato=vanilla litchi=mushroom=nut=orange apple=fig=grape date kiwi 0
47 fig=grape=mushroom=orange=pear=quinoa=soy=tomato=vanilla cherry=hazelnut=jalapeno=radish eggplant=nut litchi apple=kiwi banana=date 0
65 banana=cherry=hazelnut=jalapeno=kiwi=litchi=mushroom=orange=quinoa=soy=vanilla fig=grape=radish=tomato eggplant=nut=pear date apple 0
0
"Apple"
"Banana"
"Cherry"
"Date"
"Eggplant"
"Fig"
"Grape"
"Hazelnut"
"Jalapeno"
"Kiwi"
"Litchi"
"Mushroom"
"Nut"
"Orange"
"Pear"
"Quinoa"
"Radish"
"Soy"
"Tomato"
"Vanilla"
"Vegetable contest"
"#
        );
    }

    #[test]
    fn test_blt_hypergeometric() {
        let mut buf = Vec::new();
        let mut rng = ChaChaRng::seed_from_u64(42);
        write_blt_hypergeometric(
            &mut rng,
            &mut buf,
            "Vegetable contest",
            &VEGETABLES,
            /* ballot_count = */ 7,
        )
        .unwrap();

        assert_eq!(
            std::str::from_utf8(&buf).unwrap(),
            r#"20 10
[nick apple banana cherry date eggplant fig grape hazelnut jalapeno kiwi litchi mushroom nut orange pear quinoa radish soy tomato vanilla]
51 banana apple=cherry=grape eggplant date fig=jalapeno=mushroom=soy kiwi hazelnut=quinoa litchi orange=pear nut tomato vanilla radish 0
46 banana apple date cherry=fig eggplant=mushroom=nut grape hazelnut=jalapeno=kiwi=orange=quinoa radish=soy pear=vanilla litchi tomato 0
33 cherry jalapeno banana=date apple=grape eggplant nut fig=litchi=pear hazelnut=mushroom=quinoa kiwi=tomato orange vanilla soy radish 0
16 banana date apple=eggplant=fig=jalapeno grape=kiwi pear litchi=mushroom cherry hazelnut=nut vanilla orange=quinoa radish soy tomato 0
2 banana cherry=eggplant apple=hazelnut=jalapeno date nut=radish grape=orange fig=pear kiwi=litchi=tomato quinoa=soy vanilla mushroom 0
20 apple banana=litchi cherry=nut fig=grape=hazelnut=kiwi date=eggplant jalapeno radish pear=tomato mushroom quinoa orange=soy vanilla 0
44 banana apple cherry=grape date=nut eggplant=fig hazelnut=jalapeno pear kiwi=orange radish litchi=mushroom vanilla tomato quinoa=soy 0
0
"Apple"
"Banana"
"Cherry"
"Date"
"Eggplant"
"Fig"
"Grape"
"Hazelnut"
"Jalapeno"
"Kiwi"
"Litchi"
"Mushroom"
"Nut"
"Orange"
"Pear"
"Quinoa"
"Radish"
"Soy"
"Tomato"
"Vanilla"
"Vegetable contest"
"#
        );
    }

    #[test]
    fn test_blt_mixed() {
        let mut buf = Vec::new();
        let mut rng = ChaChaRng::seed_from_u64(42);
        write_blt_mixed(
            &mut rng,
            &mut buf,
            "Vegetable contest",
            &VEGETABLES,
            /* ballot_count = */ 20,
        )
        .unwrap();

        assert_eq!(
            std::str::from_utf8(&buf).unwrap(),
            r#"20 10
[nick apple banana cherry date eggplant fig grape hazelnut jalapeno kiwi litchi mushroom nut orange pear quinoa radish soy tomato vanilla]
1 apple=banana=cherry=date 0
1 banana=date=eggplant hazelnut fig=jalapeno=kiwi 0
1 apple=banana=cherry=date=eggplant=fig=grape=hazelnut=kiwi=litchi=mushroom=nut=orange=pear=vanilla radish 0
1 eggplant 0
1 date 0
1 cherry banana date apple jalapeno hazelnut kiwi eggplant grape fig=mushroom pear litchi=orange quinoa=soy tomato 0
1 apple banana fig litchi cherry eggplant kiwi date grape jalapeno nut 0
1 date apple banana eggplant=hazelnut=kiwi=litchi fig 0
1 cherry apple 0
1 fig apple=date cherry=eggplant hazelnut=mushroom litchi banana nut=pear vanilla soy grape=kiwi=orange=tomato jalapeno=quinoa=radish 0
1 apple=banana=cherry date=eggplant=fig=jalapeno=kiwi=mushroom litchi grape=hazelnut=nut=pear=radish=tomato orange=quinoa=soy=vanilla 0
1 apple eggplant banana grape 0
1 banana=date=eggplant 0
1 banana=date apple=cherry=eggplant=fig=hazelnut=litchi grape=nut jalapeno=kiwi=mushroom=orange=pear=quinoa=radish=soy=tomato=vanilla 0
1 apple fig 0
1 apple banana=date fig cherry=grape=hazelnut mushroom eggplant litchi=orange kiwi=vanilla jalapeno quinoa pear 0
1 banana=cherry 0
1 banana apple date 0
1 apple cherry=eggplant=fig banana 0
1 apple=jalapeno=kiwi date grape 0
0
"Apple"
"Banana"
"Cherry"
"Date"
"Eggplant"
"Fig"
"Grape"
"Hazelnut"
"Jalapeno"
"Kiwi"
"Litchi"
"Mushroom"
"Nut"
"Orange"
"Pear"
"Quinoa"
"Radish"
"Soy"
"Tomato"
"Vanilla"
"Vegetable contest"
"#
        );
    }
}
