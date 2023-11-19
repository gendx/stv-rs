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

//! Module providing exact arithmetic, implementing the [`Integer`] trait for
//! [`i64`] and [`BigInt`], and the [`Rational`] trait for [`Ratio<I>`].

use super::{Integer, IntegerRef, Rational, RationalRef};
use num::bigint::ToBigInt;
use num::rational::Ratio;
use num::traits::{NumAssign, ToPrimitive, Zero};
use num::BigInt;
use std::fmt::{Debug, Display};

#[cfg(test)]
impl Integer for i64 {
    #[inline(always)]
    fn from_usize(i: usize) -> Self {
        i as i64
    }

    fn get_positive_test_values() -> Vec<Self> {
        let mut result = Vec::new();
        for i in 0..=30 {
            result.push(1 << i);
            result.push(0x7FFF_FFFF ^ (1 << i));
        }
        result
    }
}

#[cfg(test)]
impl IntegerRef<i64> for &i64 {}

impl Integer for BigInt {
    fn from_usize(i: usize) -> Self {
        BigInt::from(i)
    }

    #[cfg(test)]
    fn get_positive_test_values() -> Vec<Self> {
        let mut result = Vec::new();
        for i in 0..64 {
            result.push(BigInt::from(1u64 << i));
            result.push(BigInt::from(!(1u64 << i)));
        }
        result
    }
}

impl IntegerRef<BigInt> for &BigInt {}

impl<I> RationalRef<&I, Ratio<I>> for &Ratio<I> where I: Clone + num::Integer {}

impl<I> Rational<I> for Ratio<I>
where
    I: Display + Debug,
    I: num::Integer,
    I: ToPrimitive + ToBigInt,
    I: Integer + NumAssign,
    for<'a> &'a I: IntegerRef<I>,
{
    fn from_int(i: I) -> Self {
        Ratio::from_integer(i)
    }

    fn ratio_i(num: I, denom: I) -> Self {
        Ratio::new(num, denom)
    }

    fn to_f64(&self) -> f64 {
        ToPrimitive::to_f64(self).unwrap()
    }

    fn epsilon() -> Self {
        Self::zero()
    }

    fn is_exact() -> bool {
        true
    }

    fn description() -> &'static str {
        "exact rational arithmetic"
    }

    fn div_up_as_keep_factor(&self, rhs: &Self) -> Self {
        self / rhs
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
            result.push(Self::ratio(0x7FFF_FFFF ^ (1 << i), 1));
        }
        for i in 0..=30 {
            result.push(Self::ratio(1, 0x7FFF_FFFF ^ (1 << i)));
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
    use num::traits::One;
    use num::BigRational;

    fn make_ratio(num: i64, denom: i64) -> BigRational {
        BigRational::ratio_i(BigInt::from(num), BigInt::from(denom))
    }

    integer_tests!(
        BigInt,
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

    big_integer_tests!(
        BigInt,
        Some(100_000),
        testi_add_is_associative,
        testi_mul_is_associative,
        testi_mul_is_distributive,
        testi_product,
    );

    numeric_tests!(
        BigInt,
        BigRational,
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
        test_one_is_div_up_neutral,
        test_div_up_self,
        test_mul_div_up,
        test_mul_by_int,
        test_mul_div_by_int,
        test_references,
        test_assign,
    );

    big_numeric_tests!(
        BigInt,
        BigRational,
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
        BigRational,
        bench_add,
        bench_sub,
        bench_mul,
        bench_div_up,
    );

    #[test]
    fn test_description() {
        assert_eq!(BigRational::description(), "exact rational arithmetic");
    }

    #[test]
    fn test_display() {
        assert_eq!(format!("{}", BigRational::zero()), "0");
        assert_eq!(format!("{}", BigRational::one()), "1");
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
        let actual_displays: Vec<String> = BigRational::get_positive_test_values()
            .iter()
            .map(|x| format!("{x}"))
            .collect();
        assert_eq!(actual_displays, expected_displays);
    }
}
