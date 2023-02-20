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
//! This implementation is backed by [`i64`], and will panic in case of overflow
//! if the `checked_i64` feature is enabled.

use super::Rational;
use log::{trace, warn};
use num::traits::{One, Zero};
use num::Integer;
use std::fmt::{Debug, Display};
use std::iter::{Product, Sum};
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Sub, SubAssign};

/// A fixed-point decimal arithmetic for 9 decimal places. This type represents
/// a number `x` by the integer `x * 10^9`, backed by a [`i64`].
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd)]
pub struct FixedDecimal9(i64);

#[cfg(test)]
impl FixedDecimal9 {
    pub(crate) fn new(x: i64) -> Self {
        FixedDecimal9(x)
    }
}

impl Display for FixedDecimal9 {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> Result<(), std::fmt::Error> {
        let sign = if self.0 < 0 { "-" } else { "" };
        let (i, rem) = self.0.abs().div_rem(&i64::from(1_000_000_000));
        write!(f, "{sign}{i}.{rem:09}")
    }
}

impl Zero for FixedDecimal9 {
    fn zero() -> Self {
        FixedDecimal9(i64::zero())
    }
    fn is_zero(&self) -> bool {
        self.0.is_zero()
    }
}
impl One for FixedDecimal9 {
    fn one() -> Self {
        FixedDecimal9(1_000_000_000)
    }
}

impl Add for FixedDecimal9 {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        #[cfg(feature = "checked_i64")]
        return FixedDecimal9(self.0.checked_add(rhs.0).unwrap());
        #[cfg(not(feature = "checked_i64"))]
        return FixedDecimal9(self.0 + rhs.0);
    }
}
impl Sub for FixedDecimal9 {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        #[cfg(feature = "checked_i64")]
        return FixedDecimal9(self.0.checked_sub(rhs.0).unwrap());
        #[cfg(not(feature = "checked_i64"))]
        return FixedDecimal9(self.0 - rhs.0);
    }
}
impl Mul for FixedDecimal9 {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        #[cfg(feature = "checked_i64")]
        return FixedDecimal9(
            ((self.0 as i128 * rhs.0 as i128) / 1_000_000_000)
                .try_into()
                .unwrap(),
        );
        #[cfg(not(feature = "checked_i64"))]
        return FixedDecimal9(((self.0 as i128 * rhs.0 as i128) / 1_000_000_000) as i64);
    }
}
impl Mul<i64> for FixedDecimal9 {
    type Output = Self;
    fn mul(self, rhs: i64) -> Self {
        #[cfg(feature = "checked_i64")]
        return FixedDecimal9(self.0.checked_mul(rhs).unwrap());
        #[cfg(not(feature = "checked_i64"))]
        return FixedDecimal9(self.0 * rhs);
    }
}
impl Div for FixedDecimal9 {
    type Output = Self;
    #[allow(clippy::suspicious_arithmetic_impl)]
    fn div(self, rhs: Self) -> Self {
        #[cfg(feature = "checked_i64")]
        return FixedDecimal9(
            (self.0 as i128 * 1_000_000_000)
                .checked_div(rhs.0 as i128)
                .unwrap()
                .try_into()
                .unwrap(),
        );
        #[cfg(not(feature = "checked_i64"))]
        return FixedDecimal9(((self.0 as i128 * 1_000_000_000) / rhs.0 as i128) as i64);
    }
}
impl Div<i64> for FixedDecimal9 {
    type Output = Self;
    fn div(self, rhs: i64) -> Self {
        #[cfg(feature = "checked_i64")]
        return FixedDecimal9(self.0.checked_div(rhs).unwrap());
        #[cfg(not(feature = "checked_i64"))]
        return FixedDecimal9(self.0 / rhs);
    }
}

impl Add<&'_ Self> for FixedDecimal9 {
    type Output = Self;
    fn add(self, rhs: &'_ Self) -> Self {
        #[cfg(feature = "checked_i64")]
        return FixedDecimal9(self.0.checked_add(rhs.0).unwrap());
        #[cfg(not(feature = "checked_i64"))]
        return FixedDecimal9(self.0 + rhs.0);
    }
}
impl Sub<&'_ Self> for FixedDecimal9 {
    type Output = Self;
    fn sub(self, rhs: &'_ Self) -> Self {
        #[cfg(feature = "checked_i64")]
        return FixedDecimal9(self.0.checked_sub(rhs.0).unwrap());
        #[cfg(not(feature = "checked_i64"))]
        return FixedDecimal9(self.0 - rhs.0);
    }
}
impl Mul<&'_ Self> for FixedDecimal9 {
    type Output = Self;
    fn mul(self, rhs: &'_ Self) -> Self {
        #[cfg(feature = "checked_i64")]
        return FixedDecimal9(
            ((self.0 as i128 * rhs.0 as i128) / 1_000_000_000)
                .try_into()
                .unwrap(),
        );
        #[cfg(not(feature = "checked_i64"))]
        return FixedDecimal9(((self.0 as i128 * rhs.0 as i128) / 1_000_000_000) as i64);
    }
}
impl Div<&'_ Self> for FixedDecimal9 {
    type Output = Self;
    #[allow(clippy::suspicious_arithmetic_impl)]
    fn div(self, rhs: &'_ Self) -> Self {
        #[cfg(feature = "checked_i64")]
        return FixedDecimal9(
            (self.0 as i128 * 1_000_000_000)
                .checked_div(rhs.0 as i128)
                .unwrap()
                .try_into()
                .unwrap(),
        );
        #[cfg(not(feature = "checked_i64"))]
        return FixedDecimal9(((self.0 as i128 * 1_000_000_000) / rhs.0 as i128) as i64);
    }
}

impl Add<&'_ FixedDecimal9> for &'_ FixedDecimal9 {
    type Output = FixedDecimal9;
    fn add(self, rhs: &'_ FixedDecimal9) -> FixedDecimal9 {
        #[cfg(feature = "checked_i64")]
        return FixedDecimal9(self.0.checked_add(rhs.0).unwrap());
        #[cfg(not(feature = "checked_i64"))]
        return FixedDecimal9(self.0 + rhs.0);
    }
}
impl Sub<&'_ FixedDecimal9> for &'_ FixedDecimal9 {
    type Output = FixedDecimal9;
    fn sub(self, rhs: &'_ FixedDecimal9) -> FixedDecimal9 {
        #[cfg(feature = "checked_i64")]
        return FixedDecimal9(self.0.checked_sub(rhs.0).unwrap());
        #[cfg(not(feature = "checked_i64"))]
        return FixedDecimal9(self.0 - rhs.0);
    }
}
impl Mul<&'_ FixedDecimal9> for &'_ FixedDecimal9 {
    type Output = FixedDecimal9;
    fn mul(self, rhs: &'_ FixedDecimal9) -> FixedDecimal9 {
        #[cfg(feature = "checked_i64")]
        return FixedDecimal9(
            ((self.0 as i128 * rhs.0 as i128) / 1_000_000_000)
                .try_into()
                .unwrap(),
        );
        #[cfg(not(feature = "checked_i64"))]
        return FixedDecimal9(((self.0 as i128 * rhs.0 as i128) / 1_000_000_000) as i64);
    }
}
impl Mul<&'_ i64> for &'_ FixedDecimal9 {
    type Output = FixedDecimal9;
    fn mul(self, rhs: &'_ i64) -> FixedDecimal9 {
        #[cfg(feature = "checked_i64")]
        return FixedDecimal9(self.0.checked_mul(*rhs).unwrap());
        #[cfg(not(feature = "checked_i64"))]
        return FixedDecimal9(self.0 * rhs);
    }
}
impl Div<&'_ FixedDecimal9> for &'_ FixedDecimal9 {
    type Output = FixedDecimal9;
    #[allow(clippy::suspicious_arithmetic_impl)]
    fn div(self, rhs: &'_ FixedDecimal9) -> FixedDecimal9 {
        #[cfg(feature = "checked_i64")]
        return FixedDecimal9(
            (self.0 as i128 * 1_000_000_000)
                .checked_div(rhs.0 as i128)
                .unwrap()
                .try_into()
                .unwrap(),
        );
        #[cfg(not(feature = "checked_i64"))]
        return FixedDecimal9(((self.0 as i128 * 1_000_000_000) / rhs.0 as i128) as i64);
    }
}
impl Div<&'_ i64> for &'_ FixedDecimal9 {
    type Output = FixedDecimal9;
    fn div(self, rhs: &'_ i64) -> FixedDecimal9 {
        #[cfg(feature = "checked_i64")]
        return FixedDecimal9(self.0.checked_div(*rhs).unwrap());
        #[cfg(not(feature = "checked_i64"))]
        return FixedDecimal9(self.0 / rhs);
    }
}

impl AddAssign for FixedDecimal9 {
    fn add_assign(&mut self, rhs: Self) {
        #[cfg(feature = "checked_i64")]
        {
            self.0 = self.0.checked_add(rhs.0).unwrap();
        }
        #[cfg(not(feature = "checked_i64"))]
        {
            self.0 += rhs.0;
        }
    }
}
impl SubAssign for FixedDecimal9 {
    fn sub_assign(&mut self, rhs: Self) {
        #[cfg(feature = "checked_i64")]
        {
            self.0 = self.0.checked_sub(rhs.0).unwrap();
        }
        #[cfg(not(feature = "checked_i64"))]
        {
            self.0 -= rhs.0;
        }
    }
}
impl MulAssign for FixedDecimal9 {
    fn mul_assign(&mut self, rhs: Self) {
        #[cfg(feature = "checked_i64")]
        {
            self.0 = ((self.0 as i128 * rhs.0 as i128) / 1_000_000_000)
                .try_into()
                .unwrap();
        }
        #[cfg(not(feature = "checked_i64"))]
        {
            self.0 = ((self.0 as i128 * rhs.0 as i128) / 1_000_000_000) as i64;
        }
    }
}

impl AddAssign<&'_ Self> for FixedDecimal9 {
    fn add_assign(&mut self, rhs: &'_ Self) {
        #[cfg(feature = "checked_i64")]
        {
            self.0 = self.0.checked_add(rhs.0).unwrap();
        }
        #[cfg(not(feature = "checked_i64"))]
        {
            self.0 += rhs.0;
        }
    }
}
impl SubAssign<&'_ Self> for FixedDecimal9 {
    fn sub_assign(&mut self, rhs: &'_ Self) {
        #[cfg(feature = "checked_i64")]
        {
            self.0 = self.0.checked_sub(rhs.0).unwrap();
        }
        #[cfg(not(feature = "checked_i64"))]
        {
            self.0 -= rhs.0;
        }
    }
}
impl MulAssign<&'_ Self> for FixedDecimal9 {
    fn mul_assign(&mut self, rhs: &'_ Self) {
        #[cfg(feature = "checked_i64")]
        {
            self.0 = ((self.0 as i128 * rhs.0 as i128) / 1_000_000_000)
                .try_into()
                .unwrap();
        }
        #[cfg(not(feature = "checked_i64"))]
        {
            self.0 = ((self.0 as i128 * rhs.0 as i128) / 1_000_000_000) as i64;
        }
    }
}
impl DivAssign<&'_ i64> for FixedDecimal9 {
    fn div_assign(&mut self, rhs: &'_ i64) {
        #[cfg(feature = "checked_i64")]
        {
            self.0 = self.0.checked_div(*rhs).unwrap();
        }
        #[cfg(not(feature = "checked_i64"))]
        {
            self.0 /= rhs;
        }
    }
}

impl Sum for FixedDecimal9 {
    fn sum<I>(iter: I) -> Self
    where
        I: Iterator<Item = Self>,
    {
        FixedDecimal9(iter.map(|item| item.0).sum())
    }
}

impl Product for FixedDecimal9 {
    fn product<I>(iter: I) -> Self
    where
        I: Iterator<Item = Self>,
    {
        iter.fold(Self::one(), |acc, x| acc * x)
    }
}

impl<'a> Sum<&'a Self> for FixedDecimal9 {
    fn sum<I>(iter: I) -> Self
    where
        I: Iterator<Item = &'a Self>,
    {
        FixedDecimal9(iter.map(|item| &item.0).sum())
    }
}

impl Rational<i64> for FixedDecimal9 {
    fn from_int(i: i64) -> Self {
        #[cfg(feature = "checked_i64")]
        return FixedDecimal9(i.checked_mul(1_000_000_000).unwrap());
        #[cfg(not(feature = "checked_i64"))]
        return FixedDecimal9(i * 1_000_000_000);
    }

    fn ratio_i(num: i64, denom: i64) -> Self {
        #[cfg(feature = "checked_i64")]
        return FixedDecimal9(
            (num as i128 * 1_000_000_000)
                .checked_div(denom as i128)
                .unwrap()
                .try_into()
                .unwrap(),
        );
        #[cfg(not(feature = "checked_i64"))]
        return FixedDecimal9(((num as i128 * 1_000_000_000) / denom as i128) as i64);
    }

    fn to_f64(&self) -> f64 {
        self.0 as f64 / 1_000_000_000_f64
    }

    fn assert_eq(a: Self, b: Self, msg: &str) {
        if a.0 != b.0 {
            let error_eps: i64 = (a.0 - b.0).abs();
            if error_eps <= 10 {
                trace!("{msg}: Failed comparison {a} != {b} (error = {error_eps} * eps)");
            } else if error_eps <= 10_000_000 {
                warn!("{msg}: Failed comparison {a} != {b} (error = {error_eps} * eps)");
            } else {
                panic!("{msg}: Failed comparison {a} != {b} (error = {error_eps} * eps)");
            }
        }
    }

    fn epsilon() -> Self {
        FixedDecimal9(1)
    }

    fn is_exact() -> bool {
        false
    }

    fn description() -> &'static str {
        "fixed-point decimal arithmetic (9 places)"
    }

    fn mul_up(&self, rhs: &Self) -> Self {
        #[cfg(feature = "checked_i64")]
        return FixedDecimal9(
            ((self.0 as i128 * rhs.0 as i128)
                .checked_add(999_999_999)
                .unwrap()
                / 1_000_000_000)
                .try_into()
                .unwrap(),
        );
        #[cfg(not(feature = "checked_i64"))]
        return FixedDecimal9(
            ((self.0 as i128 * rhs.0 as i128 + 999_999_999) / 1_000_000_000) as i64,
        );
    }

    fn div_up(&self, rhs: &Self) -> Self {
        #[cfg(feature = "checked_i64")]
        return FixedDecimal9(
            (self.0 as i128 * 1_000_000_000 + rhs.0 as i128 - 1)
                .checked_div(rhs.0 as i128)
                .unwrap()
                .try_into()
                .unwrap(),
        );
        #[cfg(not(feature = "checked_i64"))]
        return FixedDecimal9(
            ((self.0 as i128 * 1_000_000_000 + rhs.0 as i128 - 1) / rhs.0 as i128) as i64,
        );
    }

    #[cfg(test)]
    fn get_positive_test_values() -> Vec<Self> {
        let mut result = Vec::new();
        for i in 0..=30 {
            result.push(FixedDecimal9(1 << i));
        }
        for i in 0..=30 {
            result.push(FixedDecimal9(0x7FFF_FFFF - (1 << i)));
        }
        result
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::{big_numeric_tests, numeric_benchmarks, numeric_tests};

    numeric_tests!(
        i64,
        FixedDecimal9,
        test_values_are_positive,
        test_is_exact,
        test_ceil_precision,
        test_ratio,
        test_ratio_invert => fail(r"assertion failed: `(left == right)`
  left: `FixedDecimal9(999999999)`,
 right: `FixedDecimal9(1000000000)`: R::ratio(1, a) * a != 1 for 3"),
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
        test_invert => fail(r"assertion failed: `(left == right)`
  left: `FixedDecimal9(2147483649)`,
 right: `FixedDecimal9(2147483646)`: 1/(1/a) != a for 2.147483646"),
        test_div_self,
        test_div_up_self,
        test_div_up_wrt_div,
        test_mul_div => fail(r"assertion failed: `(left == right)`
  left: `FixedDecimal9(0)`,
 right: `FixedDecimal9(1)`: (a * b) / b != a for 0.000000001, 0.000000001"),
        test_div_mul => fail(r"assertion failed: `(left == right)`
  left: `FixedDecimal9(0)`,
 right: `FixedDecimal9(1)`: (a / b) * b != a for 0.000000001, 0.000001024"),
        test_mul_by_int,
        test_div_by_int,
        test_references,
        test_assign,
    );

    big_numeric_tests!(
        i64,
        FixedDecimal9,
        None,
        test_add_is_associative,
        test_mul_is_associative => fail(r"assertion failed: `(left == right)`
  left: `FixedDecimal9(0)`,
 right: `FixedDecimal9(1)`: (a * b) * c != a * (b * c) for 0.000000001, 0.536870912, 2.147483646"),
        test_mul_is_distributive => fail(r"assertion failed: `(left == right)`
  left: `FixedDecimal9(2)`,
 right: `FixedDecimal9(1)`: a * (b + c) != (a * b) + (a * c) for 0.000000001, 0.134217728, 1.879048191"),
        test_mul_by_int_is_associative,
        test_mul_by_int_is_distributive,
        test_div_by_int_is_associative,
        test_div_by_int_is_distributive => fail(r"assertion failed: `(left == right)`
  left: `FixedDecimal9(1)`,
 right: `FixedDecimal9(0)`: (a + b) / c != (a / c) + (b / c) for 0.000000001, 0.000000001, 2"),
        test_sum,
        test_product,
    );

    numeric_benchmarks!(
        i64,
        FixedDecimal9,
        bench_add,
        bench_sub,
        bench_mul,
        bench_div,
    );

    #[test]
    fn test_description() {
        assert_eq!(
            FixedDecimal9::description(),
            "fixed-point decimal arithmetic (9 places)"
        );
    }

    #[test]
    fn test_display() {
        assert_eq!(format!("{}", FixedDecimal9::zero()), "0.000000000");
        assert_eq!(format!("{}", FixedDecimal9::one()), "1.000000000");
        assert_eq!(format!("{}", FixedDecimal9(0)), "0.000000000");
        assert_eq!(format!("{}", FixedDecimal9(1)), "0.000000001");
        assert_eq!(format!("{}", FixedDecimal9(1_000_000_000)), "1.000000000");
        assert_eq!(format!("{}", FixedDecimal9(1_234_567_890)), "1.234567890");
        assert_eq!(format!("{}", FixedDecimal9(-1)), "-0.000000001");
        assert_eq!(format!("{}", FixedDecimal9(-1_000_000_000)), "-1.000000000");
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
            "0.268435456", "0.536870912", "1.073741824",
            "2.147483646", "2.147483645", "2.147483643", "2.147483639",
            "2.147483631", "2.147483615", "2.147483583", "2.147483519",
            "2.147483391", "2.147483135", "2.147482623", "2.147481599",
            "2.147479551", "2.147475455", "2.147467263", "2.147450879",
            "2.147418111", "2.147352575", "2.147221503", "2.146959359",
            "2.146435071", "2.145386495", "2.143289343", "2.139095039",
            "2.130706431", "2.113929215", "2.080374783", "2.013265919",
            "1.879048191", "1.610612735", "1.073741823",
        ];
        let actual_displays: Vec<String> = FixedDecimal9::get_positive_test_values()
            .iter()
            .map(|x| format!("{x}"))
            .collect();
        assert_eq!(actual_displays, expected_displays);
    }
}
