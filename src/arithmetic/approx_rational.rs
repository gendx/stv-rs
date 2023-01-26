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

use super::Rational;
use num::traits::{One, Zero};
use num::{BigInt, BigRational};
use std::fmt::{Debug, Display};
use std::iter::{Product, Sum};
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Sub, SubAssign};

/// A [`BigRational`] that approximates to some precision in
/// [`Rational::ceil_precision()`]. The other operations behave exactly as
/// [`BigRational`].
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd)]
pub struct ApproxRational(BigRational);

impl Display for ApproxRational {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> Result<(), std::fmt::Error> {
        Display::fmt(&self.0, f)
    }
}

impl Zero for ApproxRational {
    fn zero() -> Self {
        ApproxRational(BigRational::zero())
    }
    fn is_zero(&self) -> bool {
        self.0.is_zero()
    }
}
impl One for ApproxRational {
    fn one() -> Self {
        ApproxRational(BigRational::one())
    }
}

impl Add for ApproxRational {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        ApproxRational(self.0.add(rhs.0))
    }
}
impl Sub for ApproxRational {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        ApproxRational(self.0.sub(rhs.0))
    }
}
impl Mul for ApproxRational {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        ApproxRational(self.0.mul(rhs.0))
    }
}
impl Mul<BigInt> for ApproxRational {
    type Output = Self;
    fn mul(self, rhs: BigInt) -> Self {
        ApproxRational(self.0.mul(rhs))
    }
}
impl Div for ApproxRational {
    type Output = Self;
    fn div(self, rhs: Self) -> Self {
        ApproxRational(self.0.div(rhs.0))
    }
}
impl Div<BigInt> for ApproxRational {
    type Output = Self;
    fn div(self, rhs: BigInt) -> Self {
        ApproxRational(self.0.div(rhs))
    }
}

impl Add<&'_ Self> for ApproxRational {
    type Output = Self;
    fn add(self, rhs: &'_ Self) -> Self {
        ApproxRational(self.0.add(&rhs.0))
    }
}
impl Sub<&'_ Self> for ApproxRational {
    type Output = Self;
    fn sub(self, rhs: &'_ Self) -> Self {
        ApproxRational(self.0.sub(&rhs.0))
    }
}
impl Mul<&'_ Self> for ApproxRational {
    type Output = Self;
    fn mul(self, rhs: &'_ Self) -> Self {
        ApproxRational(self.0.mul(&rhs.0))
    }
}
impl Div<&'_ Self> for ApproxRational {
    type Output = Self;
    fn div(self, rhs: &'_ Self) -> Self {
        ApproxRational(self.0.div(&rhs.0))
    }
}

impl Add<&'_ ApproxRational> for &'_ ApproxRational {
    type Output = ApproxRational;
    fn add(self, rhs: &'_ ApproxRational) -> ApproxRational {
        ApproxRational((&self.0).add(&rhs.0))
    }
}
impl Sub<&'_ ApproxRational> for &'_ ApproxRational {
    type Output = ApproxRational;
    fn sub(self, rhs: &'_ ApproxRational) -> ApproxRational {
        ApproxRational((&self.0).sub(&rhs.0))
    }
}
impl Mul<&'_ ApproxRational> for &'_ ApproxRational {
    type Output = ApproxRational;
    fn mul(self, rhs: &'_ ApproxRational) -> ApproxRational {
        ApproxRational((&self.0).mul(&rhs.0))
    }
}
impl Mul<&'_ BigInt> for &'_ ApproxRational {
    type Output = ApproxRational;
    fn mul(self, rhs: &'_ BigInt) -> ApproxRational {
        ApproxRational((&self.0).mul(rhs))
    }
}
impl Div<&'_ ApproxRational> for &'_ ApproxRational {
    type Output = ApproxRational;
    fn div(self, rhs: &'_ ApproxRational) -> ApproxRational {
        ApproxRational((&self.0).div(&rhs.0))
    }
}
impl Div<&'_ BigInt> for &'_ ApproxRational {
    type Output = ApproxRational;
    fn div(self, rhs: &'_ BigInt) -> ApproxRational {
        ApproxRational((&self.0).div(rhs))
    }
}

impl AddAssign for ApproxRational {
    fn add_assign(&mut self, rhs: Self) {
        self.0.add_assign(rhs.0)
    }
}
impl SubAssign for ApproxRational {
    fn sub_assign(&mut self, rhs: Self) {
        self.0.sub_assign(rhs.0)
    }
}
impl MulAssign for ApproxRational {
    fn mul_assign(&mut self, rhs: Self) {
        self.0.mul_assign(rhs.0)
    }
}

impl AddAssign<&'_ Self> for ApproxRational {
    fn add_assign(&mut self, rhs: &'_ Self) {
        self.0.add_assign(&rhs.0)
    }
}
impl SubAssign<&'_ Self> for ApproxRational {
    fn sub_assign(&mut self, rhs: &'_ Self) {
        self.0.sub_assign(&rhs.0)
    }
}
impl MulAssign<&'_ Self> for ApproxRational {
    fn mul_assign(&mut self, rhs: &'_ Self) {
        self.0.mul_assign(&rhs.0)
    }
}
impl DivAssign<&'_ BigInt> for ApproxRational {
    fn div_assign(&mut self, rhs: &'_ BigInt) {
        self.0.div_assign(rhs)
    }
}

impl Sum for ApproxRational {
    fn sum<I>(iter: I) -> Self
    where
        I: Iterator<Item = Self>,
    {
        ApproxRational(iter.map(|item| item.0).sum())
    }
}

impl Product for ApproxRational {
    fn product<I>(iter: I) -> Self
    where
        I: Iterator<Item = Self>,
    {
        ApproxRational(iter.map(|item| item.0).product())
    }
}

impl<'a> Sum<&'a Self> for ApproxRational {
    fn sum<I>(iter: I) -> Self
    where
        I: Iterator<Item = &'a Self>,
    {
        ApproxRational(iter.map(|item| &item.0).sum())
    }
}

impl Rational<BigInt> for ApproxRational {
    fn from_int(i: BigInt) -> Self {
        ApproxRational(BigRational::from_int(i))
    }

    fn to_f64(&self) -> f64 {
        Rational::to_f64(&self.0)
    }

    fn assert_eq(a: Self, b: Self, msg: &str) {
        Rational::assert_eq(a.0, b.0, msg);
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

    fn ceil_precision(&mut self) {
        let precision = BigInt::from(1_000_000);

        self.0 *= &precision;
        let numer = self.0.ceil().numer().clone();
        self.0 = BigRational::new(numer, precision);
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::{big_numeric_tests, numeric_tests};

    fn make_ratio(num: i64, denom: i64) -> ApproxRational {
        ApproxRational(BigRational::new(BigInt::from(num), BigInt::from(denom)))
    }

    fn get_positive_test_values() -> Vec<ApproxRational> {
        let mut result = Vec::new();
        for i in 0..=30 {
            result.push(make_ratio(1 << i, 1));
        }
        for i in 0..=30 {
            result.push(make_ratio(1, 1 << i));
        }
        for i in 0..=30 {
            result.push(make_ratio(0x7FFF_FFFF - (1 << i), 1));
        }
        for i in 0..=30 {
            result.push(make_ratio(1, 0x7FFF_FFFF - (1 << i)));
        }
        result
    }

    numeric_tests!(
        test_values_are_positive,
        test_is_zero,
        test_zero_is_add_neutral,
        test_add_is_commutative,
        test_opposite,
        test_sub_self,
        test_add_sub,
        test_sub_add,
        test_one_is_mul_neutral,
        test_mul_is_commutative,
        test_invert,
        test_div_self,
        test_mul_div,
        test_div_mul,
        test_references,
        test_assign,
    );

    big_numeric_tests!(
        Some(100_000),
        test_add_is_associative,
        test_mul_is_associative,
        test_mul_is_distributive,
        test_mul_by_int_is_associative,
        test_mul_by_int_is_distributive,
        test_sum,
        test_product,
    );

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
        let actual_displays: Vec<String> = get_positive_test_values()
            .iter()
            .map(|x| format!("{x}"))
            .collect();
        assert_eq!(actual_displays, expected_displays);
    }
}
