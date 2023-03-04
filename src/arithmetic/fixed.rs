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
//! The implementation is generic over N decimal places (i.e. with a factor
//! `10^-N`). This implementation is backed by [`i64`], and will panic in case
//! of overflow if the `checked_i64` feature is enabled.

use super::{Integer, IntegerRef, Rational, RationalRef};
use num::traits::{One, Zero};
use num::Integer as NumInteger;
use std::fmt::{Debug, Display};
use std::iter::{Product, Sum};
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Sub, SubAssign};

/// An integer wrapping a [`i64`], performing arithmetic overflow checks if the
/// `checked_i64` feature is enabled.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd)]
pub struct Integer64(i64);

impl Debug for Integer64 {
    #[inline(always)]
    fn fmt(&self, f: &mut std::fmt::Formatter) -> Result<(), std::fmt::Error> {
        write!(f, "{}", self.0)
    }
}

#[cfg(test)]
impl Display for Integer64 {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> Result<(), std::fmt::Error> {
        write!(f, "{}", self.0)
    }
}

impl Zero for Integer64 {
    #[inline(always)]
    fn zero() -> Self {
        Integer64(i64::zero())
    }
    #[inline(always)]
    fn is_zero(&self) -> bool {
        self.0.is_zero()
    }
}
impl One for Integer64 {
    #[inline(always)]
    fn one() -> Self {
        Integer64(1)
    }
}

impl Add for Integer64 {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        #[cfg(feature = "checked_i64")]
        return Integer64(self.0.checked_add(rhs.0).unwrap());
        #[cfg(not(feature = "checked_i64"))]
        return Integer64(self.0 + rhs.0);
    }
}
impl Sub for Integer64 {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        #[cfg(feature = "checked_i64")]
        return Integer64(self.0.checked_sub(rhs.0).unwrap());
        #[cfg(not(feature = "checked_i64"))]
        return Integer64(self.0 - rhs.0);
    }
}
impl Mul for Integer64 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        #[cfg(feature = "checked_i64")]
        return Integer64(self.0.checked_mul(rhs.0).unwrap());
        #[cfg(not(feature = "checked_i64"))]
        return Integer64(self.0 * rhs.0);
    }
}

impl Add<&'_ Integer64> for &'_ Integer64 {
    type Output = Integer64;
    #[inline(always)]
    fn add(self, rhs: &'_ Integer64) -> Integer64 {
        *self + *rhs
    }
}
impl Sub<&'_ Integer64> for &'_ Integer64 {
    type Output = Integer64;
    #[inline(always)]
    fn sub(self, rhs: &'_ Integer64) -> Integer64 {
        *self - *rhs
    }
}
impl Mul<&'_ Integer64> for &'_ Integer64 {
    type Output = Integer64;
    #[inline(always)]
    fn mul(self, rhs: &'_ Integer64) -> Integer64 {
        *self * *rhs
    }
}

impl Product for Integer64 {
    #[inline(always)]
    fn product<I>(iter: I) -> Self
    where
        I: Iterator<Item = Self>,
    {
        iter.fold(Self::one(), |acc, x| acc * x)
    }
}

impl Integer for Integer64 {
    #[inline(always)]
    fn from_usize(i: usize) -> Self {
        #[cfg(feature = "checked_i64")]
        return Integer64(i.try_into().unwrap());
        #[cfg(not(feature = "checked_i64"))]
        return Integer64(i as i64);
    }

    #[cfg(test)]
    fn get_positive_test_values() -> Vec<Self> {
        let mut result = Vec::new();
        for i in 0..=30 {
            result.push(Integer64(1 << i));
            result.push(Integer64(0x7FFF_FFFF ^ (1 << i)));
        }
        result
    }
}

impl IntegerRef<Integer64> for &Integer64 {}

/// A fixed-point decimal arithmetic for N decimal places. This type represents
/// a number `x` by the integer `x * 10^N`, backed by a [`i64`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd)]
pub struct FixedDecimal<const N: u32>(i64);

/// A fixed-point decimal arithmetic for 9 decimal places. See [`FixedDecimal`].
pub type FixedDecimal9 = FixedDecimal<9>;

impl<const N: u32> FixedDecimal<N> {
    const FACTOR: i64 = 10_i64.pow(N);
    const FACTOR_I128: i128 = Self::FACTOR as i128;
}

#[cfg(test)]
impl<const N: u32> FixedDecimal<N> {
    #[inline(always)]
    pub(crate) fn new(x: i64) -> Self {
        FixedDecimal(x)
    }

    fn from_i64(x: i64) -> Self {
        FixedDecimal::from_int(Integer64(x))
    }
}

impl<const N: u32> Display for FixedDecimal<N> {
    #[inline(always)]
    fn fmt(&self, f: &mut std::fmt::Formatter) -> Result<(), std::fmt::Error> {
        let sign = if self.0 < 0 { "-" } else { "" };
        let (i, rem) = self.0.abs().div_rem(&Self::FACTOR);
        write!(f, "{sign}{i}.{rem:0width$}", width = N as usize)
    }
}

impl<const N: u32> Zero for FixedDecimal<N> {
    #[inline(always)]
    fn zero() -> Self {
        FixedDecimal(i64::zero())
    }
    #[inline(always)]
    fn is_zero(&self) -> bool {
        self.0.is_zero()
    }
}
impl<const N: u32> One for FixedDecimal<N> {
    #[inline(always)]
    fn one() -> Self {
        FixedDecimal(Self::FACTOR)
    }
}

impl<const N: u32> Add for FixedDecimal<N> {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        #[cfg(feature = "checked_i64")]
        return FixedDecimal(self.0.checked_add(rhs.0).unwrap());
        #[cfg(not(feature = "checked_i64"))]
        return FixedDecimal(self.0 + rhs.0);
    }
}
impl<const N: u32> Sub for FixedDecimal<N> {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        #[cfg(feature = "checked_i64")]
        return FixedDecimal(self.0.checked_sub(rhs.0).unwrap());
        #[cfg(not(feature = "checked_i64"))]
        return FixedDecimal(self.0 - rhs.0);
    }
}
impl<const N: u32> Mul for FixedDecimal<N> {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        FixedDecimal(match self.0.checked_mul(rhs.0) {
            Some(product) => product / Self::FACTOR,
            None => {
                let result: i128 = (self.0 as i128 * rhs.0 as i128) / Self::FACTOR_I128;
                #[cfg(feature = "checked_i64")]
                let result: i64 = result.try_into().unwrap();
                #[cfg(not(feature = "checked_i64"))]
                let result: i64 = result as i64;
                result
            }
        })
    }
}
impl<const N: u32> Mul<Integer64> for FixedDecimal<N> {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Integer64) -> Self {
        #[cfg(feature = "checked_i64")]
        return FixedDecimal(self.0.checked_mul(rhs.0).unwrap());
        #[cfg(not(feature = "checked_i64"))]
        return FixedDecimal(self.0 * rhs.0);
    }
}
impl<const N: u32> Div<Integer64> for FixedDecimal<N> {
    type Output = Self;
    #[inline(always)]
    fn div(self, rhs: Integer64) -> Self {
        #[cfg(feature = "checked_i64")]
        return FixedDecimal(self.0.checked_div(rhs.0).unwrap());
        #[cfg(not(feature = "checked_i64"))]
        return FixedDecimal(self.0 / rhs.0);
    }
}

impl<const N: u32> Add<&'_ Self> for FixedDecimal<N> {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: &'_ Self) -> Self {
        self + *rhs
    }
}
impl<const N: u32> Sub<&'_ Self> for FixedDecimal<N> {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: &'_ Self) -> Self {
        self - *rhs
    }
}
impl<const N: u32> Mul<&'_ Self> for FixedDecimal<N> {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: &'_ Self) -> Self {
        self * *rhs
    }
}

impl<const N: u32> Add<&'_ FixedDecimal<N>> for &'_ FixedDecimal<N> {
    type Output = FixedDecimal<N>;
    #[inline(always)]
    fn add(self, rhs: &'_ FixedDecimal<N>) -> FixedDecimal<N> {
        *self + *rhs
    }
}
impl<const N: u32> Sub<&'_ FixedDecimal<N>> for &'_ FixedDecimal<N> {
    type Output = FixedDecimal<N>;
    #[inline(always)]
    fn sub(self, rhs: &'_ FixedDecimal<N>) -> FixedDecimal<N> {
        *self - *rhs
    }
}
impl<const N: u32> Mul<&'_ FixedDecimal<N>> for &'_ FixedDecimal<N> {
    type Output = FixedDecimal<N>;
    #[inline(always)]
    fn mul(self, rhs: &'_ FixedDecimal<N>) -> FixedDecimal<N> {
        *self * *rhs
    }
}
impl<const N: u32> Mul<&'_ Integer64> for &'_ FixedDecimal<N> {
    type Output = FixedDecimal<N>;
    #[inline(always)]
    fn mul(self, rhs: &'_ Integer64) -> FixedDecimal<N> {
        *self * *rhs
    }
}
impl<const N: u32> Div<&'_ Integer64> for &'_ FixedDecimal<N> {
    type Output = FixedDecimal<N>;
    #[inline(always)]
    fn div(self, rhs: &'_ Integer64) -> FixedDecimal<N> {
        *self / *rhs
    }
}

impl<const N: u32> AddAssign for FixedDecimal<N> {
    #[inline(always)]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs
    }
}
impl<const N: u32> SubAssign for FixedDecimal<N> {
    #[inline(always)]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs
    }
}
impl<const N: u32> MulAssign for FixedDecimal<N> {
    #[inline(always)]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs
    }
}

impl<const N: u32> AddAssign<&'_ Self> for FixedDecimal<N> {
    #[inline(always)]
    fn add_assign(&mut self, rhs: &'_ Self) {
        *self = *self + *rhs
    }
}
impl<const N: u32> SubAssign<&'_ Self> for FixedDecimal<N> {
    #[inline(always)]
    fn sub_assign(&mut self, rhs: &'_ Self) {
        *self = *self - *rhs
    }
}
impl<const N: u32> MulAssign<&'_ Self> for FixedDecimal<N> {
    #[inline(always)]
    fn mul_assign(&mut self, rhs: &'_ Self) {
        *self = *self * *rhs
    }
}
impl<const N: u32> DivAssign<&'_ Integer64> for FixedDecimal<N> {
    #[inline(always)]
    fn div_assign(&mut self, rhs: &'_ Integer64) {
        *self = *self / *rhs
    }
}

impl<const N: u32> Sum for FixedDecimal<N> {
    #[inline(always)]
    fn sum<I>(iter: I) -> Self
    where
        I: Iterator<Item = Self>,
    {
        FixedDecimal(iter.map(|item| item.0).sum())
    }
}

impl<const N: u32> Product for FixedDecimal<N> {
    #[inline(always)]
    fn product<I>(iter: I) -> Self
    where
        I: Iterator<Item = Self>,
    {
        iter.fold(Self::one(), |acc, x| acc * x)
    }
}

impl<'a, const N: u32> Sum<&'a Self> for FixedDecimal<N> {
    #[inline(always)]
    fn sum<I>(iter: I) -> Self
    where
        I: Iterator<Item = &'a Self>,
    {
        FixedDecimal(iter.map(|item| &item.0).sum())
    }
}

impl<const N: u32> RationalRef<&Integer64, FixedDecimal<N>> for &FixedDecimal<N> {}

impl<const N: u32> Rational<Integer64> for FixedDecimal<N> {
    #[inline(always)]
    fn from_int(i: Integer64) -> Self {
        #[cfg(feature = "checked_i64")]
        return FixedDecimal(i.0.checked_mul(Self::FACTOR).unwrap());
        #[cfg(not(feature = "checked_i64"))]
        return FixedDecimal(i.0 * Self::FACTOR);
    }

    #[inline(always)]
    fn ratio_i(num: Integer64, denom: Integer64) -> Self {
        FixedDecimal(match num.0.checked_mul(Self::FACTOR) {
            Some(product) => {
                #[cfg(feature = "checked_i64")]
                let result: i64 = product.checked_div(denom.0).unwrap();
                #[cfg(not(feature = "checked_i64"))]
                let result: i64 = product / denom.0;
                result
            }
            None => {
                let product: i128 = num.0 as i128 * Self::FACTOR_I128;
                #[cfg(feature = "checked_i64")]
                let result: i64 = product
                    .checked_div(denom.0 as i128)
                    .unwrap()
                    .try_into()
                    .unwrap();
                #[cfg(not(feature = "checked_i64"))]
                let result: i64 = (product / denom.0 as i128) as i64;
                result
            }
        })
    }

    #[inline(always)]
    fn to_f64(&self) -> f64 {
        self.0 as f64 / Self::FACTOR as f64
    }

    #[inline(always)]
    fn epsilon() -> Self {
        FixedDecimal(1)
    }

    #[inline(always)]
    fn is_exact() -> bool {
        false
    }

    #[inline(always)]
    fn description() -> String {
        format!("fixed-point decimal arithmetic ({N} places)")
    }

    #[inline(always)]
    fn mul_up(&self, rhs: &Self) -> Self {
        FixedDecimal(
            match self
                .0
                .checked_mul(rhs.0)
                .and_then(|product| product.checked_add(Self::FACTOR - 1))
            {
                Some(value) => value / Self::FACTOR,
                None => {
                    let product: i128 = self.0 as i128 * rhs.0 as i128;
                    #[cfg(feature = "checked_i64")]
                    let result: i64 = (product.checked_add(Self::FACTOR_I128 - 1).unwrap()
                        / Self::FACTOR_I128)
                        .try_into()
                        .unwrap();
                    #[cfg(not(feature = "checked_i64"))]
                    let result: i64 =
                        ((product + Self::FACTOR_I128 - 1) / Self::FACTOR_I128) as i64;
                    result
                }
            },
        )
    }

    #[inline(always)]
    fn div_up_as_keep_factor(&self, rhs: &Self) -> Self {
        FixedDecimal(
            match self
                .0
                .checked_mul(Self::FACTOR)
                .and_then(|product| product.checked_add(rhs.0 - 1))
            {
                Some(value) => {
                    #[cfg(feature = "checked_i64")]
                    let result: i64 = value.checked_div(rhs.0).unwrap();
                    #[cfg(not(feature = "checked_i64"))]
                    let result: i64 = value / rhs.0;
                    result
                }
                None => {
                    let value: i128 = self.0 as i128 * Self::FACTOR_I128 + rhs.0 as i128 - 1;
                    #[cfg(feature = "checked_i64")]
                    let result: i64 = value
                        .checked_div(rhs.0 as i128)
                        .unwrap()
                        .try_into()
                        .unwrap();
                    #[cfg(not(feature = "checked_i64"))]
                    let result: i64 = (value / rhs.0 as i128) as i64;
                    result
                }
            },
        )
    }

    #[cfg(test)]
    fn get_positive_test_values() -> Vec<Self> {
        let mut result = Vec::new();
        for i in 0..=30 {
            result.push(FixedDecimal(1 << i));
        }
        for i in 0..=30 {
            result.push(FixedDecimal(0x7FFF_FFFF - (1 << i)));
        }
        result
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::{
        big_integer_tests, big_numeric_tests, integer_tests, numeric_benchmarks, numeric_tests,
    };

    integer_tests!(
        Integer64,
        testi_values_are_positive,
        testi_is_zero,
        testi_zero_is_add_neutral,
        testi_add_is_commutative,
        testi_opposite,
        testi_sub_self,
        testi_add_sub,
        testi_sub_add,
        testi_one_is_mul_neutral,
        testi_mul_is_commutative,
    );

    #[cfg(not(any(feature = "checked_i64", overflow_checks)))]
    big_integer_tests!(
        Integer64,
        None,
        testi_add_is_associative,
        testi_mul_is_associative,
        testi_mul_is_distributive,
    );
    #[cfg(feature = "checked_i64")]
    big_integer_tests!(
        Integer64,
        None,
        testi_add_is_associative,
        testi_mul_is_associative => fail(r"called `Option::unwrap()` on a `None` value"),
        testi_mul_is_distributive,
        testi_product => fail(r"called `Option::unwrap()` on a `None` value"),
    );
    #[cfg(all(not(feature = "checked_i64"), overflow_checks))]
    big_integer_tests!(
        Integer64,
        None,
        testi_add_is_associative,
        testi_mul_is_associative => fail(r"attempt to multiply with overflow"),
        testi_mul_is_distributive,
        testi_product => fail(r"attempt to multiply with overflow"),
    );

    numeric_tests!(
        Integer64,
        FixedDecimal9,
        test_values_are_positive,
        test_is_exact,
        test_ratio,
        test_ratio_invert => fail(r"assertion `left == right` failed: R::ratio(1, a) * a != 1 for 3
  left: FixedDecimal(999999999)
 right: FixedDecimal(1000000000)"),
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
        test_one_is_div_up_neutral,
        test_div_up_self,
        test_mul_div_up => fail(r"assertion `left == right` failed: div_up(a * b, b) != a for 0.000000001, 0.000000001
  left: FixedDecimal(0)
 right: FixedDecimal(1)"),
        test_mul_by_int,
        test_mul_div_by_int,
        test_references,
        test_assign,
    );

    big_numeric_tests!(
        Integer64,
        FixedDecimal9,
        None,
        test_add_is_associative,
        test_mul_is_associative => fail(r"assertion `left == right` failed: (a * b) * c != a * (b * c) for 0.000000001, 0.536870912, 2.147483646
  left: FixedDecimal(0)
 right: FixedDecimal(1)"),
        test_mul_is_distributive => fail(r"assertion `left == right` failed: a * (b + c) != (a * b) + (a * c) for 0.000000001, 0.134217728, 1.879048191
  left: FixedDecimal(2)
 right: FixedDecimal(1)"),
        test_mul_by_int_is_associative,
        test_mul_by_int_is_distributive,
        test_div_by_int_is_associative,
        test_div_by_int_is_distributive => fail(r"assertion `left == right` failed: (a + b) / c != (a / c) + (b / c) for 0.000000001, 0.000000001, 2
  left: FixedDecimal(1)
 right: FixedDecimal(0)"),
        test_sum,
        test_product,
    );

    numeric_benchmarks!(
        Integer64,
        FixedDecimal9,
        bench_add,
        bench_sub,
        bench_mul,
        bench_div_up,
    );

    #[test]
    fn test_description() {
        assert_eq!(
            FixedDecimal::<8>::description(),
            "fixed-point decimal arithmetic (8 places)"
        );
        assert_eq!(
            FixedDecimal::<9>::description(),
            "fixed-point decimal arithmetic (9 places)"
        );
    }

    #[test]
    fn test_display() {
        assert_eq!(format!("{}", FixedDecimal::<9>::zero()), "0.000000000");
        assert_eq!(format!("{}", FixedDecimal::<9>::one()), "1.000000000");
        assert_eq!(format!("{}", FixedDecimal::<9>(0)), "0.000000000");
        assert_eq!(format!("{}", FixedDecimal::<9>(1)), "0.000000001");
        assert_eq!(
            format!("{}", FixedDecimal::<9>(1_000_000_000)),
            "1.000000000"
        );
        assert_eq!(
            format!("{}", FixedDecimal::<9>(1_234_567_890)),
            "1.234567890"
        );
        assert_eq!(format!("{}", FixedDecimal::<9>(-1)), "-0.000000001");
        assert_eq!(
            format!("{}", FixedDecimal::<9>(-1_000_000_000)),
            "-1.000000000"
        );
    }

    #[test]
    fn test_display_test_values_8() {
        #[rustfmt::skip]
        let expected_displays = [
            "0.00000001", "0.00000002", "0.00000004", "0.00000008",
            "0.00000016", "0.00000032", "0.00000064", "0.00000128",
            "0.00000256", "0.00000512", "0.00001024", "0.00002048",
            "0.00004096", "0.00008192", "0.00016384", "0.00032768",
            "0.00065536", "0.00131072", "0.00262144", "0.00524288",
            "0.01048576", "0.02097152", "0.04194304", "0.08388608",
            "0.16777216", "0.33554432", "0.67108864", "1.34217728",
            "2.68435456", "5.36870912", "10.73741824",
            "21.47483646", "21.47483645", "21.47483643", "21.47483639",
            "21.47483631", "21.47483615", "21.47483583", "21.47483519",
            "21.47483391", "21.47483135", "21.47482623", "21.47481599",
            "21.47479551", "21.47475455", "21.47467263", "21.47450879",
            "21.47418111", "21.47352575", "21.47221503", "21.46959359",
            "21.46435071", "21.45386495", "21.43289343", "21.39095039",
            "21.30706431", "21.13929215", "20.80374783", "20.13265919",
            "18.79048191", "16.10612735", "10.73741823",
        ];
        let actual_displays: Vec<String> = FixedDecimal::<8>::get_positive_test_values()
            .iter()
            .map(|x| format!("{x}"))
            .collect();
        assert_eq!(actual_displays, expected_displays);
    }

    #[test]
    fn test_display_test_values_9() {
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
        let actual_displays: Vec<String> = FixedDecimal::<9>::get_positive_test_values()
            .iter()
            .map(|x| format!("{x}"))
            .collect();
        assert_eq!(actual_displays, expected_displays);
    }

    #[test]
    fn test_intermediate_overflow() {
        // Even though the intermediate product overflows a i64, the result doesn't and
        // is correct.
        assert_eq!(
            FixedDecimal9::from_i64(1_000) * FixedDecimal9::from_i64(1_000),
            FixedDecimal9::from_i64(1_000_000)
        );
        // The intermediate result of 10^19 is just between 2^63 and 2^64. Therefore it
        // overflows i64 as well.
        assert_eq!(
            FixedDecimal9::from_i64(5) * FixedDecimal9::from_i64(2),
            FixedDecimal9::from_i64(10)
        );
        // The intermediate product of 10^19 overflows a i64.
        assert_eq!(
            FixedDecimal9::ratio(10_000_000_000, 2),
            FixedDecimal9::from_i64(5_000_000_000)
        );

        // Same check for MulAssign.
        let mut x = FixedDecimal9::from_i64(1_000);
        x *= FixedDecimal9::from_i64(1_000);
        assert_eq!(x, FixedDecimal9::from_i64(1_000_000));
    }

    #[cfg(feature = "checked_i64")]
    #[test]
    #[should_panic(expected = "called `Result::unwrap()` on an `Err` value: TryFromIntError(())")]
    fn test_mul_overflow() {
        // The result overflows an i64, which is caught by checked_i64.
        let _ = FixedDecimal9::from_i64(1_000_000) * FixedDecimal9::from_i64(1_000_000);
    }

    #[cfg(feature = "checked_i64")]
    #[test]
    #[should_panic(expected = "called `Result::unwrap()` on an `Err` value: TryFromIntError(())")]
    fn test_mul_assign_overflow() {
        // The result overflows an i64, which is caught by checked_i64.
        let mut x = FixedDecimal9::from_i64(1_000_000);
        x *= FixedDecimal9::from_i64(1_000_000);
    }

    #[cfg(not(feature = "checked_i64"))]
    #[test]
    fn test_overflow() {
        // When checked_i64 is disabled, overflow leads to incorrect values.
        assert_eq!(
            FixedDecimal9::from_i64(1_000_000) * FixedDecimal9::from_i64(1_000_000),
            FixedDecimal::<9>(3_875_820_019_684_212_736)
        );

        let mut x = FixedDecimal9::from_i64(1_000_000);
        x *= FixedDecimal9::from_i64(1_000_000);
        assert_eq!(x, FixedDecimal::<9>(3_875_820_019_684_212_736));
    }
}
