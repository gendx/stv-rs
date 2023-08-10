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

//! Module for fixed-point arithmetic, defined in terms of *decimal* places.
//! For now, it only implements 9 decimal places (i.e. with a factor `10^-9`).
//! This implementation is backed by a [`BigInt`].

use super::{Rational, RationalRef};
use num::bigint::Sign;
use num::traits::{One, Zero};
use num::{BigInt, BigRational, Integer, Signed};
use std::fmt::{Debug, Display};
use std::iter::{Product, Sum};
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Sub, SubAssign};

/// A fixed-point decimal arithmetic for 9 decimal places. This type represents
/// a number `x` by the integer `x * 10^9`, backed by a [`BigInt`].
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd)]
pub struct BigFixedDecimal9(BigInt);

impl BigFixedDecimal9 {
    const FACTOR: i64 = 1_000_000_000;
}

impl Display for BigFixedDecimal9 {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> Result<(), std::fmt::Error> {
        let sign = match self.0.sign() {
            Sign::Plus | Sign::NoSign => "",
            Sign::Minus => "-",
        };
        let (i, rem) = self.0.abs().div_rem(&BigInt::from(Self::FACTOR));
        write!(f, "{sign}{i}.{rem:09}")
    }
}

impl Zero for BigFixedDecimal9 {
    fn zero() -> Self {
        BigFixedDecimal9(BigInt::zero())
    }
    fn is_zero(&self) -> bool {
        self.0.is_zero()
    }
}
impl One for BigFixedDecimal9 {
    fn one() -> Self {
        BigFixedDecimal9(BigInt::from(Self::FACTOR))
    }
}

impl Add for BigFixedDecimal9 {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        BigFixedDecimal9(self.0.add(rhs.0))
    }
}
impl Sub for BigFixedDecimal9 {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        BigFixedDecimal9(self.0.sub(rhs.0))
    }
}
impl Mul for BigFixedDecimal9 {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        let ratio = BigRational::new(self.0.mul(rhs.0), BigInt::from(Self::FACTOR));
        BigFixedDecimal9(ratio.floor().numer().clone())
    }
}
impl Mul<BigInt> for BigFixedDecimal9 {
    type Output = Self;
    fn mul(self, rhs: BigInt) -> Self {
        BigFixedDecimal9(self.0.mul(rhs))
    }
}
impl Div for BigFixedDecimal9 {
    type Output = Self;
    #[allow(clippy::suspicious_arithmetic_impl)]
    fn div(self, rhs: Self) -> Self {
        let ratio = BigRational::new(self.0 * BigInt::from(Self::FACTOR), rhs.0);
        BigFixedDecimal9(ratio.floor().numer().clone())
    }
}
impl Div<BigInt> for BigFixedDecimal9 {
    type Output = Self;
    fn div(self, rhs: BigInt) -> Self {
        let ratio = BigRational::new(self.0, rhs);
        BigFixedDecimal9(ratio.floor().numer().clone())
    }
}

impl Add<&'_ Self> for BigFixedDecimal9 {
    type Output = Self;
    fn add(self, rhs: &'_ Self) -> Self {
        BigFixedDecimal9(self.0.add(&rhs.0))
    }
}
impl Sub<&'_ Self> for BigFixedDecimal9 {
    type Output = Self;
    fn sub(self, rhs: &'_ Self) -> Self {
        BigFixedDecimal9(self.0.sub(&rhs.0))
    }
}
impl Mul<&'_ Self> for BigFixedDecimal9 {
    type Output = Self;
    fn mul(self, rhs: &'_ Self) -> Self {
        let ratio = BigRational::new(self.0.mul(&rhs.0), BigInt::from(Self::FACTOR));
        BigFixedDecimal9(ratio.floor().numer().clone())
    }
}
impl Div<&'_ Self> for BigFixedDecimal9 {
    type Output = Self;
    #[allow(clippy::suspicious_arithmetic_impl)]
    fn div(self, rhs: &'_ Self) -> Self {
        let ratio = BigRational::new(self.0 * BigInt::from(Self::FACTOR), rhs.0.clone());
        BigFixedDecimal9(ratio.floor().numer().clone())
    }
}

impl Add<&'_ BigFixedDecimal9> for &'_ BigFixedDecimal9 {
    type Output = BigFixedDecimal9;
    fn add(self, rhs: &'_ BigFixedDecimal9) -> BigFixedDecimal9 {
        BigFixedDecimal9((&self.0).add(&rhs.0))
    }
}
impl Sub<&'_ BigFixedDecimal9> for &'_ BigFixedDecimal9 {
    type Output = BigFixedDecimal9;
    fn sub(self, rhs: &'_ BigFixedDecimal9) -> BigFixedDecimal9 {
        BigFixedDecimal9((&self.0).sub(&rhs.0))
    }
}
impl Mul<&'_ BigFixedDecimal9> for &'_ BigFixedDecimal9 {
    type Output = BigFixedDecimal9;
    fn mul(self, rhs: &'_ BigFixedDecimal9) -> BigFixedDecimal9 {
        let ratio = BigRational::new(
            (&self.0).mul(&rhs.0),
            BigInt::from(BigFixedDecimal9::FACTOR),
        );
        BigFixedDecimal9(ratio.floor().numer().clone())
    }
}
impl Mul<&'_ BigInt> for &'_ BigFixedDecimal9 {
    type Output = BigFixedDecimal9;
    fn mul(self, rhs: &'_ BigInt) -> BigFixedDecimal9 {
        BigFixedDecimal9((&self.0).mul(rhs))
    }
}
impl Div<&'_ BigFixedDecimal9> for &'_ BigFixedDecimal9 {
    type Output = BigFixedDecimal9;
    #[allow(clippy::suspicious_arithmetic_impl)]
    fn div(self, rhs: &'_ BigFixedDecimal9) -> BigFixedDecimal9 {
        let ratio = BigRational::new(
            self.0.clone() * BigInt::from(BigFixedDecimal9::FACTOR),
            rhs.0.clone(),
        );
        BigFixedDecimal9(ratio.floor().numer().clone())
    }
}
impl Div<&'_ BigInt> for &'_ BigFixedDecimal9 {
    type Output = BigFixedDecimal9;
    fn div(self, rhs: &'_ BigInt) -> BigFixedDecimal9 {
        let ratio = BigRational::new(self.0.clone(), rhs.clone());
        BigFixedDecimal9(ratio.floor().numer().clone())
    }
}

impl AddAssign for BigFixedDecimal9 {
    fn add_assign(&mut self, rhs: Self) {
        self.0.add_assign(rhs.0)
    }
}
impl SubAssign for BigFixedDecimal9 {
    fn sub_assign(&mut self, rhs: Self) {
        self.0.sub_assign(rhs.0)
    }
}
impl MulAssign for BigFixedDecimal9 {
    fn mul_assign(&mut self, rhs: Self) {
        let ratio = BigRational::new(self.0.clone().mul(rhs.0), BigInt::from(Self::FACTOR));
        self.0 = ratio.floor().numer().clone();
    }
}

impl AddAssign<&'_ Self> for BigFixedDecimal9 {
    fn add_assign(&mut self, rhs: &'_ Self) {
        self.0.add_assign(&rhs.0)
    }
}
impl SubAssign<&'_ Self> for BigFixedDecimal9 {
    fn sub_assign(&mut self, rhs: &'_ Self) {
        self.0.sub_assign(&rhs.0)
    }
}
impl MulAssign<&'_ Self> for BigFixedDecimal9 {
    fn mul_assign(&mut self, rhs: &'_ Self) {
        let ratio = BigRational::new(self.0.clone().mul(&rhs.0), BigInt::from(Self::FACTOR));
        self.0 = ratio.floor().numer().clone();
    }
}
impl DivAssign<&'_ BigInt> for BigFixedDecimal9 {
    fn div_assign(&mut self, rhs: &'_ BigInt) {
        let ratio = BigRational::new(self.0.clone(), rhs.clone());
        self.0 = ratio.floor().numer().clone();
    }
}

impl Sum for BigFixedDecimal9 {
    fn sum<I>(iter: I) -> Self
    where
        I: Iterator<Item = Self>,
    {
        BigFixedDecimal9(iter.map(|item| item.0).sum())
    }
}

impl Product for BigFixedDecimal9 {
    fn product<I>(iter: I) -> Self
    where
        I: Iterator<Item = Self>,
    {
        iter.fold(Self::one(), |acc, x| acc * x)
    }
}

impl<'a> Sum<&'a Self> for BigFixedDecimal9 {
    fn sum<I>(iter: I) -> Self
    where
        I: Iterator<Item = &'a Self>,
    {
        BigFixedDecimal9(iter.map(|item| &item.0).sum())
    }
}

impl RationalRef<&BigInt, BigFixedDecimal9> for &BigFixedDecimal9 {}

impl Rational<BigInt> for BigFixedDecimal9 {
    fn from_int(i: BigInt) -> Self {
        BigFixedDecimal9(i * BigInt::from(Self::FACTOR))
    }

    fn ratio_i(num: BigInt, denom: BigInt) -> Self {
        BigFixedDecimal9((num * BigInt::from(Self::FACTOR)) / denom)
    }

    fn to_f64(&self) -> f64 {
        Rational::to_f64(&BigRational::new(
            self.0.clone(),
            BigInt::from(Self::FACTOR),
        ))
    }

    fn epsilon() -> Self {
        BigFixedDecimal9(BigInt::from(1))
    }

    fn is_exact() -> bool {
        false
    }

    fn description() -> &'static str {
        "fixed-point decimal arithmetic (9 places)"
    }

    fn mul_up(&self, rhs: &Self) -> Self {
        let ratio = BigRational::new(&self.0 * &rhs.0, BigInt::from(Self::FACTOR));
        BigFixedDecimal9(ratio.ceil().numer().clone())
    }

    fn div_up(&self, rhs: &Self) -> Self {
        let ratio = BigRational::new(&self.0 * BigInt::from(Self::FACTOR), rhs.0.clone());
        BigFixedDecimal9(ratio.ceil().numer().clone())
    }

    #[cfg(test)]
    fn get_positive_test_values() -> Vec<Self> {
        let mut result = Vec::new();
        for i in 0..=62 {
            result.push(BigFixedDecimal9(BigInt::from(1i64 << i)));
        }
        for i in 0..=62 {
            result.push(BigFixedDecimal9(BigInt::from(
                0x7FFF_FFFF_FFFF_FFFF_i64 - (1 << i),
            )));
        }
        result
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::{big_numeric_tests, numeric_benchmarks, numeric_tests};

    numeric_tests!(
        BigInt,
        BigFixedDecimal9,
        test_values_are_positive,
        test_is_exact,
        test_ceil_precision,
        test_ratio,
        test_ratio_invert => fail(r"assertion `left == right` failed: R::ratio(1, a) * a != 1 for 3
  left: BigFixedDecimal9(999999999)
 right: BigFixedDecimal9(1000000000)"),
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
        test_invert => fail(r"assertion `left == right` failed: 1/(1/a) != a for 2.147483648
  left: BigFixedDecimal9(2147483649)
 right: BigFixedDecimal9(2147483648)"),
        test_div_self,
        test_div_up_self,
        test_div_up_wrt_div,
        test_mul_div => fail(r"assertion `left == right` failed: (a * b) / b != a for 0.000000001, 0.000000001
  left: BigFixedDecimal9(0)
 right: BigFixedDecimal9(1)"),
        test_div_mul => fail(r"assertion `left == right` failed: (a / b) * b != a for 0.000000001, 0.000001024
  left: BigFixedDecimal9(0)
 right: BigFixedDecimal9(1)"),
        test_mul_by_int,
        test_div_by_int,
        test_references,
        test_assign,
    );

    big_numeric_tests!(
        BigInt,
        BigFixedDecimal9,
        None,
        test_add_is_associative,
        test_mul_is_associative => fail(r"assertion `left == right` failed: (a * b) * c != a * (b * c) for 0.000000001, 0.000000001, 1152921504.606846976
  left: BigFixedDecimal9(0)
 right: BigFixedDecimal9(1)"),
        test_mul_is_distributive => fail(r"assertion `left == right` failed: a * (b + c) != (a * b) + (a * c) for 0.000000001, 0.033554432, 281474.976710656
  left: BigFixedDecimal9(281475)
 right: BigFixedDecimal9(281474)"),
        test_mul_by_int_is_associative,
        test_mul_by_int_is_distributive,
        test_div_by_int_is_associative,
        test_div_by_int_is_distributive => fail(r"assertion `left == right` failed: (a + b) / c != (a / c) + (b / c) for 0.000000001, 0.000000001, 2
  left: BigFixedDecimal9(1)
 right: BigFixedDecimal9(0)"),
        test_sum,
        test_product,
    );

    numeric_benchmarks!(
        BigInt,
        BigFixedDecimal9,
        bench_add,
        bench_sub,
        bench_mul,
        bench_div,
    );

    #[test]
    fn test_description() {
        assert_eq!(
            BigFixedDecimal9::description(),
            "fixed-point decimal arithmetic (9 places)"
        );
    }

    #[test]
    fn test_display() {
        assert_eq!(format!("{}", BigFixedDecimal9::zero()), "0.000000000");
        assert_eq!(format!("{}", BigFixedDecimal9::one()), "1.000000000");
        assert_eq!(
            format!("{}", BigFixedDecimal9(BigInt::from(0))),
            "0.000000000"
        );
        assert_eq!(
            format!("{}", BigFixedDecimal9(BigInt::from(1))),
            "0.000000001"
        );
        assert_eq!(
            format!("{}", BigFixedDecimal9(BigInt::from(1_000_000_000))),
            "1.000000000"
        );
        assert_eq!(
            format!("{}", BigFixedDecimal9(BigInt::from(1_234_567_890))),
            "1.234567890"
        );
        assert_eq!(
            format!("{}", BigFixedDecimal9(BigInt::from(-1))),
            "-0.000000001"
        );
        assert_eq!(
            format!("{}", BigFixedDecimal9(BigInt::from(-1_000_000_000))),
            "-1.000000000"
        );
        assert_eq!(
            format!("{}", BigFixedDecimal9(BigInt::from(10).pow(20))),
            "100000000000.000000000"
        );
    }

    #[test]
    fn test_display_test_values() {
        #[rustfmt::skip]
        let expected_displays = [
            "0.000000001", "0.000000002", "0.000000004", "0.000000008",
            "0.000000016", "0.000000032", "0.000000064", "0.000000128",
            "0.000000256", "0.000000512", "0.000001024", "0.000002048",
            "0.000004096", "0.000008192", "0.000016384", "0.000032768",
            "0.000065536", "0.000131072", "0.000262144", "0.000524288",
            "0.001048576", "0.002097152", "0.004194304", "0.008388608",
            "0.016777216", "0.033554432", "0.067108864", "0.134217728",
            "0.268435456", "0.536870912", "1.073741824", "2.147483648",
            "4.294967296", "8.589934592", "17.179869184", "34.359738368",
            "68.719476736", "137.438953472", "274.877906944", "549.755813888",
            "1099.511627776", "2199.023255552", "4398.046511104", "8796.093022208",
            "17592.186044416", "35184.372088832", "70368.744177664", "140737.488355328",
            "281474.976710656", "562949.953421312", "1125899.906842624", "2251799.813685248",
            "4503599.627370496", "9007199.254740992", "18014398.509481984", "36028797.018963968",
            "72057594.037927936", "144115188.075855872", "288230376.151711744",
            "576460752.303423488", "1152921504.606846976", "2305843009.213693952",
            "4611686018.427387904",
            "9223372036.854775806", "9223372036.854775805", "9223372036.854775803",
            "9223372036.854775799", "9223372036.854775791", "9223372036.854775775",
            "9223372036.854775743", "9223372036.854775679", "9223372036.854775551",
            "9223372036.854775295", "9223372036.854774783", "9223372036.854773759",
            "9223372036.854771711", "9223372036.854767615", "9223372036.854759423",
            "9223372036.854743039", "9223372036.854710271", "9223372036.854644735",
            "9223372036.854513663", "9223372036.854251519", "9223372036.853727231",
            "9223372036.852678655", "9223372036.850581503", "9223372036.846387199",
            "9223372036.837998591", "9223372036.821221375", "9223372036.787666943",
            "9223372036.720558079", "9223372036.586340351", "9223372036.317904895",
            "9223372035.781033983", "9223372034.707292159", "9223372032.559808511",
            "9223372028.264841215", "9223372019.674906623", "9223372002.495037439",
            "9223371968.135299071", "9223371899.415822335", "9223371761.976868863",
            "9223371487.098961919", "9223370937.343148031", "9223369837.831520255",
            "9223367638.808264703", "9223363240.761753599", "9223354444.668731391",
            "9223336852.482686975", "9223301668.110598143", "9223231299.366420479",
            "9223090561.878065151", "9222809086.901354495", "9222246136.947933183",
            "9221120237.041090559", "9218868437.227405311", "9214364837.600034815",
            "9205357638.345293823", "9187343239.835811839", "9151314442.816847871",
            "9079256848.778919935", "8935141660.703064063", "8646911284.551352319",
            "8070450532.247928831", "6917529027.641081855", "4611686018.427387903"
        ];
        let actual_displays: Vec<String> = BigFixedDecimal9::get_positive_test_values()
            .iter()
            .map(|x| format!("{x}"))
            .collect();
        assert_eq!(actual_displays, expected_displays);
    }

    /// Check that [`BigFixedDecimal9`] correctly handles inputs that would
    /// overflow with [`FixedDecimal9`], which is backed by [`i64`].
    #[test]
    fn test_i64_overflow() {
        // The intermediate result of 10^19 is just between 2^63 and 2^64.
        assert_eq!(
            BigFixedDecimal9::from_int(5.into()) * BigFixedDecimal9::from_int(2.into()),
            BigFixedDecimal9::from_int(10.into())
        );
        // The intermediate result is above 2^64.
        assert_eq!(
            BigFixedDecimal9::from_int(1_000.into()) * BigFixedDecimal9::from_int(1_000.into()),
            BigFixedDecimal9::from_int(1_000_000.into())
        );
        // The final result exceeds 2^64.
        assert_eq!(
            BigFixedDecimal9::from_int(1_000_000.into())
                * BigFixedDecimal9::from_int(1_000_000.into()),
            BigFixedDecimal9::from_int(1_000_000_000_000_i64.into())
        );
    }
}
