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

use super::{Rational, RationalRef};
use num::traits::{One, Zero};
use num::{BigInt, BigRational};
use std::cmp::Ordering;
use std::fmt::{Debug, Display};
use std::iter::{Product, Sum};
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Sub, SubAssign};

#[derive(Clone, Debug, PartialEq)]
struct Denom {
    primes: [u8; Self::NUM_PRIMES],
    remainder: Option<BigInt>,
}

impl Denom {
    const NUM_PRIMES: usize = 24;
    const PRIMES: [u64; Self::NUM_PRIMES] = [
        2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89,
    ];

    fn one() -> Self {
        Denom {
            primes: [0; Self::NUM_PRIMES],
            remainder: None,
        }
    }

    fn precision() -> Self {
        let mut primes = [0; Denom::NUM_PRIMES];
        // 1_000_000 = (2*5)^6
        primes[0] = 6; // prime = 2
        primes[2] = 6; // prime = 5
        Denom {
            primes,
            remainder: None,
        }
    }

    fn to_bigint(&self) -> BigInt {
        let mut result = match &self.remainder {
            Some(x) => x.clone(),
            None => BigInt::one(),
        };
        let mut tmp = 1u64;
        for (i, &count) in self.primes.iter().enumerate() {
            let p = Self::PRIMES[i];
            for _ in 0..count {
                match tmp.checked_mul(p) {
                    Some(prod) => tmp = prod,
                    None => {
                        result *= tmp;
                        tmp = p;
                    }
                }
            }
        }
        result * tmp
    }

    const fn decompose_small(mut x: u64) -> Self {
        let mut primes = [0; Self::NUM_PRIMES];
        // TODO: use a for loop once supported in `const fn` context.
        let mut i = 0;
        while x > 1 && i < Self::NUM_PRIMES {
            let p = Self::PRIMES[i];
            while x % p == 0 {
                x /= p;
                primes[i] += 1;
            }
            i += 1;
        }

        if x != 1 {
            panic!("Failed to decompose small integer into small prime factors.");
        }
        Denom {
            primes,
            remainder: None,
        }
    }

    // TODO: Use std::array::from_fn when available in const contexts.
    const DECOMPOSED: [Self; 90] = [
        Self::decompose_small(1),
        Self::decompose_small(2),
        Self::decompose_small(3),
        Self::decompose_small(4),
        Self::decompose_small(5),
        Self::decompose_small(6),
        Self::decompose_small(7),
        Self::decompose_small(8),
        Self::decompose_small(9),
        Self::decompose_small(10),
        Self::decompose_small(11),
        Self::decompose_small(12),
        Self::decompose_small(13),
        Self::decompose_small(14),
        Self::decompose_small(15),
        Self::decompose_small(16),
        Self::decompose_small(17),
        Self::decompose_small(18),
        Self::decompose_small(19),
        Self::decompose_small(20),
        Self::decompose_small(21),
        Self::decompose_small(22),
        Self::decompose_small(23),
        Self::decompose_small(24),
        Self::decompose_small(25),
        Self::decompose_small(26),
        Self::decompose_small(27),
        Self::decompose_small(28),
        Self::decompose_small(29),
        Self::decompose_small(30),
        Self::decompose_small(31),
        Self::decompose_small(32),
        Self::decompose_small(33),
        Self::decompose_small(34),
        Self::decompose_small(35),
        Self::decompose_small(36),
        Self::decompose_small(37),
        Self::decompose_small(38),
        Self::decompose_small(39),
        Self::decompose_small(40),
        Self::decompose_small(41),
        Self::decompose_small(42),
        Self::decompose_small(43),
        Self::decompose_small(44),
        Self::decompose_small(45),
        Self::decompose_small(46),
        Self::decompose_small(47),
        Self::decompose_small(48),
        Self::decompose_small(49),
        Self::decompose_small(50),
        Self::decompose_small(51),
        Self::decompose_small(52),
        Self::decompose_small(53),
        Self::decompose_small(54),
        Self::decompose_small(55),
        Self::decompose_small(56),
        Self::decompose_small(57),
        Self::decompose_small(58),
        Self::decompose_small(59),
        Self::decompose_small(60),
        Self::decompose_small(61),
        Self::decompose_small(62),
        Self::decompose_small(63),
        Self::decompose_small(64),
        Self::decompose_small(65),
        Self::decompose_small(66),
        Self::decompose_small(67),
        Self::decompose_small(68),
        Self::decompose_small(69),
        Self::decompose_small(70),
        Self::decompose_small(71),
        Self::decompose_small(72),
        Self::decompose_small(73),
        Self::decompose_small(74),
        Self::decompose_small(75),
        Self::decompose_small(76),
        Self::decompose_small(77),
        Self::decompose_small(78),
        Self::decompose_small(79),
        Self::decompose_small(80),
        Self::decompose_small(81),
        Self::decompose_small(82),
        Self::decompose_small(83),
        Self::decompose_small(84),
        Self::decompose_small(85),
        Self::decompose_small(86),
        Self::decompose_small(87),
        Self::decompose_small(88),
        Self::decompose_small(89),
        Self::decompose_small(90),
    ];

    fn decompose(x: BigInt) -> Self {
        if x > BigInt::zero() && x <= BigInt::from(90) {
            return Self::DECOMPOSED[TryInto::<usize>::try_into(x).unwrap() - 1].clone();
        }
        Self::decompose_now(x)
    }

    fn decompose_now(mut x: BigInt) -> Self {
        let mut primes = [0; Self::NUM_PRIMES];
        'outer: for (i, &p) in Self::PRIMES.iter().enumerate() {
            while (&x % p).is_zero() {
                x /= p;
                primes[i] += 1;
                if x.is_one() {
                    break 'outer;
                }
            }
        }

        if x.is_one() {
            Denom {
                primes,
                remainder: None,
            }
        } else {
            Denom {
                primes,
                remainder: Some(x),
            }
        }
    }

    /// Returns the least common multiple of two denominators, adjusting the
    /// numerators accordingly.
    fn normalize(lnum: &mut BigInt, rnum: &mut BigInt, ldenom: &Self, rdenom: &Self) -> Self {
        let mut primes = [0; Self::NUM_PRIMES];
        let mut ltmp = 1u64;
        let mut rtmp = 1u64;
        for (i, &p) in Self::PRIMES.iter().enumerate() {
            let lcount = ldenom.primes[i];
            let rcount = rdenom.primes[i];
            match lcount.cmp(&rcount) {
                Ordering::Equal => {
                    primes[i] = lcount;
                }
                Ordering::Less => {
                    Self::accum_pow(lnum, &mut ltmp, p, rcount - lcount);
                    primes[i] = rcount;
                }
                Ordering::Greater => {
                    Self::accum_pow(rnum, &mut rtmp, p, lcount - rcount);
                    primes[i] = lcount;
                }
            }
        }

        *lnum *= ltmp;
        *rnum *= rtmp;
        let remainder = match (&ldenom.remainder, &rdenom.remainder) {
            (None, None) => None,
            (None, Some(r)) => {
                *lnum *= r;
                Some(r.clone())
            }
            (Some(l), None) => {
                *rnum *= l;
                Some(l.clone())
            }
            (Some(l), Some(r)) => {
                if l == r {
                    Some(l.clone())
                } else {
                    *lnum *= r;
                    *rnum *= l;
                    Some(l * r)
                }
            }
        };
        Denom { primes, remainder }
    }

    /// Computes `prime.pow(exponent)` and multiplies it into the accumulated
    /// `(numerator, tmp)`.
    fn accum_pow(numerator: &mut BigInt, tmp: &mut u64, prime: u64, exponent: u8) {
        for _ in 0..exponent {
            match tmp.checked_mul(prime) {
                Some(prod) => *tmp = prod,
                None => {
                    *numerator *= *tmp;
                    *tmp = prime;
                }
            }
        }
    }
}

impl Mul<&'_ Denom> for &'_ Denom {
    type Output = Denom;
    #[allow(clippy::needless_range_loop)]
    fn mul(self, rhs: &'_ Denom) -> Denom {
        let mut primes = [0; Denom::NUM_PRIMES];
        for i in 0..Denom::NUM_PRIMES {
            primes[i] = self.primes[i].checked_add(rhs.primes[i]).unwrap();
        }
        let remainder = match (&self.remainder, &rhs.remainder) {
            (None, None) => None,
            (None, Some(r)) => Some(r.clone()),
            (Some(l), None) => Some(l.clone()),
            (Some(l), Some(r)) => Some(l * r),
        };
        Denom { primes, remainder }
    }
}

/// A [`BigRational`] that approximates to some precision in
/// [`Rational::div_up_as_keep_factor()`]. The other operations behave exactly
/// as [`BigRational`].
#[derive(Clone, Debug)]
pub struct ApproxRational {
    num: BigInt,
    denom: Denom,
}

impl PartialEq for ApproxRational {
    fn eq(&self, rhs: &Self) -> bool {
        &self.num * rhs.denom.to_bigint() == &rhs.num * self.denom.to_bigint()
    }
}

impl Eq for ApproxRational {}

impl PartialOrd for ApproxRational {
    fn partial_cmp(&self, rhs: &Self) -> Option<Ordering> {
        Some(self.cmp(rhs))
    }
}

impl Ord for ApproxRational {
    fn cmp(&self, rhs: &Self) -> Ordering {
        (&self.num * rhs.denom.to_bigint()).cmp(&(&rhs.num * self.denom.to_bigint()))
    }
}

impl ApproxRational {
    fn reduced(&self) -> BigRational {
        BigRational::new(self.num.clone(), self.denom.to_bigint())
    }
}

impl Display for ApproxRational {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> Result<(), std::fmt::Error> {
        Display::fmt(&self.reduced(), f)
    }
}

impl Zero for ApproxRational {
    fn zero() -> Self {
        ApproxRational {
            num: BigInt::zero(),
            denom: Denom::one(),
        }
    }
    fn is_zero(&self) -> bool {
        self.num.is_zero()
    }
}
impl One for ApproxRational {
    fn one() -> Self {
        ApproxRational {
            num: BigInt::one(),
            denom: Denom::one(),
        }
    }
}

impl Add for ApproxRational {
    type Output = Self;
    fn add(mut self, mut rhs: Self) -> Self {
        let denom = Denom::normalize(&mut self.num, &mut rhs.num, &self.denom, &rhs.denom);
        ApproxRational {
            num: self.num + rhs.num,
            denom,
        }
    }
}
impl Sub for ApproxRational {
    type Output = Self;
    fn sub(mut self, mut rhs: Self) -> Self {
        let denom = Denom::normalize(&mut self.num, &mut rhs.num, &self.denom, &rhs.denom);
        ApproxRational {
            num: self.num - rhs.num,
            denom,
        }
    }
}
impl Mul for ApproxRational {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        ApproxRational {
            num: self.num * rhs.num,
            denom: &self.denom * &rhs.denom,
        }
    }
}
impl Mul<BigInt> for ApproxRational {
    type Output = Self;
    fn mul(self, rhs: BigInt) -> Self {
        ApproxRational {
            num: self.num * rhs,
            denom: self.denom,
        }
    }
}
impl Div<BigInt> for ApproxRational {
    type Output = Self;
    #[allow(clippy::suspicious_arithmetic_impl)]
    fn div(self, rhs: BigInt) -> Self {
        ApproxRational {
            num: self.num,
            denom: &self.denom * &Denom::decompose(rhs),
        }
    }
}

impl Add<&'_ Self> for ApproxRational {
    type Output = Self;
    fn add(mut self, rhs: &'_ Self) -> Self {
        let mut rnum = rhs.num.clone();
        let denom = Denom::normalize(&mut self.num, &mut rnum, &self.denom, &rhs.denom);
        ApproxRational {
            num: self.num + rnum,
            denom,
        }
    }
}
impl Sub<&'_ Self> for ApproxRational {
    type Output = Self;
    fn sub(mut self, rhs: &'_ Self) -> Self {
        let mut rnum = rhs.num.clone();
        let denom = Denom::normalize(&mut self.num, &mut rnum, &self.denom, &rhs.denom);
        ApproxRational {
            num: self.num - rnum,
            denom,
        }
    }
}
impl Mul<&'_ Self> for ApproxRational {
    type Output = Self;
    fn mul(self, rhs: &'_ Self) -> Self {
        ApproxRational {
            num: self.num * &rhs.num,
            denom: &self.denom * &rhs.denom,
        }
    }
}

impl Add<&'_ ApproxRational> for &'_ ApproxRational {
    type Output = ApproxRational;
    fn add(self, rhs: &'_ ApproxRational) -> ApproxRational {
        let mut lnum = self.num.clone();
        let mut rnum = rhs.num.clone();
        let denom = Denom::normalize(&mut lnum, &mut rnum, &self.denom, &rhs.denom);
        ApproxRational {
            num: lnum + rnum,
            denom,
        }
    }
}
impl Sub<&'_ ApproxRational> for &'_ ApproxRational {
    type Output = ApproxRational;
    fn sub(self, rhs: &'_ ApproxRational) -> ApproxRational {
        let mut lnum = self.num.clone();
        let mut rnum = rhs.num.clone();
        let denom = Denom::normalize(&mut lnum, &mut rnum, &self.denom, &rhs.denom);
        ApproxRational {
            num: lnum - rnum,
            denom,
        }
    }
}
impl Mul<&'_ ApproxRational> for &'_ ApproxRational {
    type Output = ApproxRational;
    fn mul(self, rhs: &'_ ApproxRational) -> ApproxRational {
        ApproxRational {
            num: &self.num * &rhs.num,
            denom: &self.denom * &rhs.denom,
        }
    }
}
impl Mul<&'_ BigInt> for &'_ ApproxRational {
    type Output = ApproxRational;
    fn mul(self, rhs: &'_ BigInt) -> ApproxRational {
        ApproxRational {
            num: &self.num * rhs,
            denom: self.denom.clone(),
        }
    }
}
impl Div<&'_ BigInt> for &'_ ApproxRational {
    type Output = ApproxRational;
    #[allow(clippy::suspicious_arithmetic_impl)]
    fn div(self, rhs: &'_ BigInt) -> ApproxRational {
        ApproxRational {
            num: self.num.clone(),
            denom: &self.denom * &Denom::decompose(rhs.clone()),
        }
    }
}

impl AddAssign for ApproxRational {
    fn add_assign(&mut self, mut rhs: Self) {
        self.denom = Denom::normalize(&mut self.num, &mut rhs.num, &self.denom, &rhs.denom);
        self.num += rhs.num;
    }
}
impl SubAssign for ApproxRational {
    fn sub_assign(&mut self, mut rhs: Self) {
        self.denom = Denom::normalize(&mut self.num, &mut rhs.num, &self.denom, &rhs.denom);
        self.num -= rhs.num;
    }
}
impl MulAssign for ApproxRational {
    fn mul_assign(&mut self, rhs: Self) {
        self.denom = &self.denom * &rhs.denom;
        self.num *= rhs.num;
    }
}

impl AddAssign<&'_ Self> for ApproxRational {
    fn add_assign(&mut self, rhs: &'_ Self) {
        let mut rnum = rhs.num.clone();
        self.denom = Denom::normalize(&mut self.num, &mut rnum, &self.denom, &rhs.denom);
        self.num += &rnum;
    }
}
impl SubAssign<&'_ Self> for ApproxRational {
    fn sub_assign(&mut self, rhs: &'_ Self) {
        let mut rnum = rhs.num.clone();
        self.denom = Denom::normalize(&mut self.num, &mut rnum, &self.denom, &rhs.denom);
        self.num -= &rnum;
    }
}
impl MulAssign<&'_ Self> for ApproxRational {
    fn mul_assign(&mut self, rhs: &'_ Self) {
        self.denom = &self.denom * &rhs.denom;
        self.num *= &rhs.num;
    }
}
impl DivAssign<&'_ BigInt> for ApproxRational {
    #[allow(clippy::suspicious_op_assign_impl)]
    fn div_assign(&mut self, rhs: &'_ BigInt) {
        self.denom = &self.denom * &Denom::decompose(rhs.clone());
    }
}

impl Sum for ApproxRational {
    fn sum<I>(iter: I) -> Self
    where
        I: Iterator<Item = Self>,
    {
        iter.fold(Self::zero(), |acc, x| acc + x)
    }
}

impl Product for ApproxRational {
    fn product<I>(iter: I) -> Self
    where
        I: Iterator<Item = Self>,
    {
        iter.fold(Self::one(), |acc, x| acc * x)
    }
}

impl<'a> Sum<&'a Self> for ApproxRational {
    fn sum<I>(iter: I) -> Self
    where
        I: Iterator<Item = &'a Self>,
    {
        iter.fold(Self::zero(), |acc, x| acc + x)
    }
}

impl RationalRef<&BigInt, ApproxRational> for &ApproxRational {}

impl Rational<BigInt> for ApproxRational {
    fn from_int(i: BigInt) -> Self {
        ApproxRational {
            num: i,
            denom: Denom::one(),
        }
    }

    fn ratio_i(num: BigInt, denom: BigInt) -> Self {
        ApproxRational {
            num,
            denom: Denom::decompose(denom),
        }
    }

    fn to_f64(&self) -> f64 {
        Rational::to_f64(&self.reduced())
    }

    fn epsilon() -> Self {
        Self::zero()
    }

    fn is_exact() -> bool {
        true
    }

    fn description() -> &'static str {
        "exact rational arithmetic with rounding of keep factors (6 decimal places)"
    }

    fn div_up_as_keep_factor(&self, rhs: &Self) -> Self {
        let precision = BigInt::from(1_000_000);

        let ldenom = self.denom.to_bigint();
        let rdenom = rhs.denom.to_bigint();

        let num = &self.num * precision * rdenom;
        let denom = ldenom * &rhs.num;

        Self {
            num: (num + &denom - 1) / denom,
            denom: Denom::precision(),
        }
    }

    #[cfg(test)]
    fn get_positive_test_values() -> Vec<Self> {
        let mut result = Vec::new();
        for i in 0..=30 {
            result.push(Self::ratio(1 << i, 1));
        }
        for i in 0..=30 {
            result.push(Self::ratio(1, 1 << i));
        }
        for i in 0..=30 {
            result.push(Self::ratio(0x7FFF_FFFF - (1 << i), 1));
        }
        for i in 0..=30 {
            result.push(Self::ratio(1, 0x7FFF_FFFF - (1 << i)));
        }
        result
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::{big_numeric_tests, numeric_benchmarks, numeric_tests};

    fn make_ratio(num: i64, denom: i64) -> ApproxRational {
        ApproxRational::ratio_i(BigInt::from(num), BigInt::from(denom))
    }

    numeric_tests!(
        BigInt,
        ApproxRational,
        test_values_are_positive,
        test_is_exact,
        test_ratio,
        test_ratio_invert,
        test_is_zero,
        test_zero_is_add_neutral,
        test_add_is_commutative,
        test_opposite,
        test_sub_self,
        test_add_sub,
        test_sub_add,
        test_one_is_mul_neutral,
        test_mul_is_commutative,
        test_mul_up_is_commutative,
        test_mul_up_integers,
        test_mul_up_wrt_mul,
        test_one_is_div_up_neutral => fail(r"assertion `left == right` failed: div_up(a, 1) != a for 1/128
  left: ApproxRational { num: 7813, denom: Denom { primes: [6, 0, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], remainder: None } }
 right: ApproxRational { num: 1, denom: Denom { primes: [7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], remainder: None } }"),
        test_div_up_self,
        test_mul_div_up => fail(r"assertion `left == right` failed: div_up(a * b, b) != a for 1/128, 1
  left: ApproxRational { num: 7813, denom: Denom { primes: [6, 0, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], remainder: None } }
 right: ApproxRational { num: 1, denom: Denom { primes: [7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], remainder: None } }"),
        test_mul_by_int,
        test_mul_div_by_int,
        test_references,
        test_assign,
    );

    big_numeric_tests!(
        BigInt,
        ApproxRational,
        Some(100_000),
        test_add_is_associative,
        test_mul_is_associative,
        test_mul_is_distributive,
        test_mul_by_int_is_associative,
        test_mul_by_int_is_distributive,
        test_div_by_int_is_associative,
        test_div_by_int_is_distributive,
        test_sum,
        test_product,
    );

    numeric_benchmarks!(
        BigInt,
        ApproxRational,
        bench_add,
        bench_sub,
        bench_mul,
        bench_div_up,
    );

    #[test]
    fn test_description() {
        assert_eq!(
            ApproxRational::description(),
            "exact rational arithmetic with rounding of keep factors (6 decimal places)"
        );
    }

    #[test]
    fn test_display() {
        assert_eq!(format!("{}", ApproxRational::zero()), "0");
        assert_eq!(format!("{}", ApproxRational::one()), "1");
        assert_eq!(format!("{}", make_ratio(0, 1)), "0");
        assert_eq!(format!("{}", make_ratio(1, 1)), "1");
        assert_eq!(format!("{}", make_ratio(1, 2)), "1/2");
        assert_eq!(format!("{}", make_ratio(1, 23456789)), "1/23456789");
        assert_eq!(format!("{}", make_ratio(123456789, 1)), "123456789");
        assert_eq!(format!("{}", make_ratio(-1, 1)), "-1");
        assert_eq!(format!("{}", make_ratio(-1, 2)), "-1/2");
        assert_eq!(format!("{}", make_ratio(2, 2)), "1");
        assert_eq!(format!("{}", make_ratio(60, 14)), "30/7");
    }

    #[test]
    fn test_display_test_values() {
        #[rustfmt::skip]
        let expected_displays = [
            "1", "2", "4", "8", "16", "32", "64", "128", "256", "512", "1024", "2048", "4096",
            "8192", "16384", "32768", "65536", "131072", "262144", "524288", "1048576", "2097152",
            "4194304", "8388608", "16777216", "33554432", "67108864", "134217728", "268435456",
            "536870912", "1073741824",
            "1", "1/2", "1/4", "1/8", "1/16", "1/32", "1/64", "1/128", "1/256", "1/512", "1/1024",
            "1/2048", "1/4096", "1/8192", "1/16384", "1/32768", "1/65536", "1/131072", "1/262144",
            "1/524288", "1/1048576", "1/2097152", "1/4194304", "1/8388608", "1/16777216",
            "1/33554432", "1/67108864", "1/134217728", "1/268435456", "1/536870912", "1/1073741824",
            "2147483646", "2147483645", "2147483643", "2147483639", "2147483631", "2147483615",
            "2147483583", "2147483519", "2147483391", "2147483135", "2147482623", "2147481599",
            "2147479551", "2147475455", "2147467263", "2147450879", "2147418111", "2147352575",
            "2147221503", "2146959359", "2146435071", "2145386495", "2143289343", "2139095039",
            "2130706431", "2113929215", "2080374783", "2013265919", "1879048191", "1610612735",
            "1073741823",
            "1/2147483646", "1/2147483645", "1/2147483643", "1/2147483639", "1/2147483631",
            "1/2147483615", "1/2147483583", "1/2147483519", "1/2147483391", "1/2147483135",
            "1/2147482623", "1/2147481599", "1/2147479551", "1/2147475455", "1/2147467263",
            "1/2147450879", "1/2147418111", "1/2147352575", "1/2147221503", "1/2146959359",
            "1/2146435071", "1/2145386495", "1/2143289343", "1/2139095039", "1/2130706431",
            "1/2113929215", "1/2080374783", "1/2013265919", "1/1879048191", "1/1610612735",
            "1/1073741823",
        ];
        let actual_displays: Vec<String> = ApproxRational::get_positive_test_values()
            .iter()
            .map(|x| format!("{x}"))
            .collect();
        assert_eq!(actual_displays, expected_displays);
    }

    mod denom {
        use super::*;

        #[test]
        fn test_decompose_small() {
            for (i, x) in Denom::DECOMPOSED.iter().enumerate() {
                assert_eq!(x, &Denom::decompose_small(i as u64 + 1));
                assert_eq!(x, &Denom::decompose(BigInt::from(i + 1)));
                assert_eq!(x, &Denom::decompose_now(BigInt::from(i + 1)));
            }
        }

        #[test]
        #[should_panic(expected = "Failed to decompose small integer into small prime factors.")]
        fn test_decompose_small_out_of_range() {
            assert_eq!(Denom::decompose_small(97).to_bigint(), BigInt::from(97));
        }

        #[test]
        fn test_decompose_is_correct() {
            for i in 1..=1000 {
                let bigi = BigInt::from(i);
                let x = Denom::decompose(bigi.clone());
                let mut recomposed = x.remainder.unwrap_or_else(BigInt::one);
                for (i, &prime) in Denom::PRIMES.iter().enumerate() {
                    for _ in 0..x.primes[i] {
                        recomposed *= prime;
                    }
                }
                assert_eq!(recomposed, bigi);
            }
        }

        #[test]
        fn test_decompose_known_values() {
            assert_eq!(
                Denom::decompose(BigInt::from(128)),
                Denom {
                    primes: [
                        7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
                    ],
                    remainder: None,
                }
            );
            assert_eq!(
                Denom::decompose(BigInt::from(89)),
                Denom {
                    primes: [
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1
                    ],
                    remainder: None,
                }
            );
            assert_eq!(
                Denom::decompose(BigInt::from(97)),
                Denom {
                    primes: [
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
                    ],
                    remainder: Some(BigInt::from(97)),
                }
            );
            assert_eq!(
                Denom::decompose(BigInt::from(97000)),
                Denom {
                    primes: [
                        3, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
                    ],
                    remainder: Some(BigInt::from(97)),
                }
            );
        }

        #[test]
        fn test_to_bigint() {
            for i in 1..=1000 {
                let bigi = BigInt::from(i);
                let x = Denom::decompose(bigi.clone());
                assert_eq!(x.to_bigint(), bigi);
            }
        }

        #[test]
        fn test_product() {
            let values = (100..200)
                .map(|i| Denom::decompose(BigInt::from(i)))
                .collect::<Vec<_>>();
            for (i, x) in values.iter().enumerate().map(|(i, x)| (i + 100, x)) {
                for (j, y) in values.iter().enumerate().map(|(j, y)| (j + 100, y)) {
                    let z = x * y;
                    assert_eq!(z, Denom::decompose(BigInt::from(i * j)));
                    for k in 0..Denom::NUM_PRIMES {
                        assert_eq!(z.primes[k], x.primes[k] + y.primes[k]);
                    }
                }
            }
        }

        #[test]
        fn test_normalize() {
            let values = (100..200)
                .map(|i| Denom::decompose(BigInt::from(i)))
                .collect::<Vec<_>>();
            for x in &values {
                for y in &values {
                    let mut xnum = BigInt::one();
                    let mut ynum = BigInt::one();
                    let lcm = Denom::normalize(&mut xnum, &mut ynum, x, y);
                    let lcm_bigint = lcm.to_bigint();

                    assert_eq!(xnum * x.to_bigint(), lcm_bigint);
                    assert_eq!(ynum * y.to_bigint(), lcm_bigint);
                    for k in 0..Denom::NUM_PRIMES {
                        assert_eq!(lcm.primes[k], std::cmp::max(x.primes[k], y.primes[k]));
                    }
                }
            }
        }
    }
}
