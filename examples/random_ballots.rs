//! Script to generate random ballot files.

use rand::distributions::{Distribution, Uniform};
use rand::seq::index::sample;
use rand::thread_rng;
use rand_distr::Hypergeometric;
use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufWriter, Result, Write};

static VEGETABLES: [&str; 20] = [
    "apple", "banana", "cherry", "date", "eggplant", "fig", "grape", "hazelnut", "jalapeno",
    "kiwi", "litchi", "mushroom", "nut", "orange", "pear", "quinoa", "radish", "soy", "tomato",
    "vanilla",
];

fn main() -> Result<()> {
    let file = File::create("rand_2x10.blt")?;
    write_blt_2x10(&mut BufWriter::new(file), 1000)?;

    let file = File::create("rand_5x4.blt")?;
    write_blt_5x4(&mut BufWriter::new(file), 1000)?;

    let file = File::create("rand_hypergeometric.blt")?;
    write_blt_hypergeometric(&mut BufWriter::new(file), 1000)?;

    Ok(())
}

fn write_blt_2x10(output: &mut impl Write, ballot_count: usize) -> Result<()> {
    write_header(output)?;
    generate_2x10(output, ballot_count)?;
    write_footer(output)?;
    Ok(())
}

fn write_blt_5x4(output: &mut impl Write, ballot_count: usize) -> Result<()> {
    write_header(output)?;
    generate_5x4(output, ballot_count)?;
    write_footer(output)?;
    Ok(())
}

fn write_blt_hypergeometric(output: &mut impl Write, ballot_count: usize) -> Result<()> {
    write_header(output)?;
    generate_hypergeometric(output, ballot_count)?;
    write_footer(output)?;
    Ok(())
}

fn write_header(output: &mut impl Write) -> Result<()> {
    writeln!(output, "20 10")?;
    write!(output, "[nick")?;
    for v in VEGETABLES {
        write!(output, " {v}")?;
    }
    writeln!(output, "]")?;
    Ok(())
}

fn write_footer(output: &mut impl Write) -> Result<()> {
    for v in VEGETABLES {
        let mut name = v.to_string().into_bytes();
        if let Some(x) = name.first_mut() {
            *x = x.to_ascii_uppercase();
        }
        let name = String::from_utf8(name).unwrap();
        writeln!(output, "\"{name}\"")?;
    }
    writeln!(output, "\"Vegetable contest\"")?;
    Ok(())
}

fn generate_2x10(output: &mut impl Write, ballot_count: usize) -> Result<()> {
    generate_tuples(output, ballot_count, 2)
}

fn generate_5x4(output: &mut impl Write, ballot_count: usize) -> Result<()> {
    generate_tuples(output, ballot_count, 5)
}

fn generate_hypergeometric(output: &mut impl Write, ballot_count: usize) -> Result<()> {
    let distributions = (0..20)
        .map(|i| Hypergeometric::new(100, 50, 20 + i).unwrap())
        .collect::<Vec<_>>();
    generate_distributions(output, ballot_count, &distributions)
}

fn generate_tuples(output: &mut impl Write, ballot_count: usize, tuple_size: usize) -> Result<()> {
    let mut rng = thread_rng();
    let count_dist = Uniform::from(1..100);
    for _ in 0..ballot_count {
        let count = count_dist.sample(&mut rng);
        write!(output, "{count} ")?;

        let order = sample(&mut rng, 20, 20);
        for (i, index) in order.iter().enumerate() {
            if i > 0 {
                if i % tuple_size == 0 {
                    write!(output, " ")?;
                } else {
                    write!(output, "=")?;
                }
            }
            write!(output, "{}", VEGETABLES[index])?;
        }

        writeln!(output, " 0")?;
    }
    writeln!(output, "0")?;
    Ok(())
}

fn generate_distributions<D: Distribution<u64>>(
    output: &mut impl Write,
    ballot_count: usize,
    distributions: &[D],
) -> Result<()> {
    let mut rng = thread_rng();
    let count_dist = Uniform::from(1..100);

    for _ in 0..ballot_count {
        let count = count_dist.sample(&mut rng);
        write!(output, "{count} ")?;

        let mut order: BTreeMap<u64, Vec<usize>> = BTreeMap::new();
        for (i, d) in distributions.iter().enumerate() {
            let value = d.sample(&mut rng);
            order.entry(value).or_insert_with(Vec::new).push(i);
        }

        for (i, rank) in order.values().enumerate() {
            if i > 0 {
                write!(output, " ")?;
            }
            for (j, &index) in rank.iter().enumerate() {
                if j > 0 {
                    write!(output, "=")?;
                }
                write!(output, "{}", VEGETABLES[index])?;
            }
        }

        writeln!(output, " 0")?;
    }

    writeln!(output, "0")?;
    Ok(())
}
