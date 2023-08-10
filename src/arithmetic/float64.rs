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

//! Module implementing the [`Integer`] and [`Rational`] traits for [`f64`].

use super::{Integer, IntegerRef, Rational, RationalRef};
use log::trace;
use num::traits::Zero;

impl Integer for f64 {
    fn from_usize(i: usize) -> Self {
        i as f64
    }
}

impl IntegerRef<f64> for &f64 {}

impl RationalRef<&f64, f64> for &f64 {}

impl Rational<f64> for f64 {
    fn from_int(i: f64) -> Self {
        i
    }

    fn ratio_i(num: f64, denom: f64) -> Self {
        num / denom
    }

    fn to_f64(&self) -> f64 {
        *self
    }

    fn assert_eq(a: Self, b: Self, msg: &str) {
        if a != b {
            let error = 2f64 * (a - b).abs() / (a.abs() + b.abs());
            let error_eps = (error / f64::EPSILON).round() as usize;
            if error_eps <= 1000 {
                trace!("{msg}: Failed comparison {a} != {b} (error = {error_eps} * eps)");
            } else {
                panic!("{msg}: Failed comparison {a} != {b} (error = {error_eps} * eps)");
            }
        }
    }

    fn epsilon() -> Self {
        Self::zero()
    }

    fn is_exact() -> bool {
        true
    }

    fn description() -> &'static str {
        "64-bit floating-point arithmetic"
    }

    #[cfg(test)]
    fn get_positive_test_values() -> Vec<Self> {
        let mut result = Vec::new();
        for i in 0..=30 {
            result.push((1 << i) as f64);
        }
        for i in 0..=30 {
            result.push((0x7FFF_FFFF - (1 << i)) as f64);
        }
        for i in 0..=30 {
            result.push(1.0 / (1 << i) as f64);
        }
        for i in 0..=30 {
            result.push(1.0 / (0x7FFF_FFFF - (1 << i)) as f64);
        }
        result
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::util::log_tester::ThreadLocalLogger;
    use crate::{big_numeric_tests, numeric_benchmarks, numeric_tests};
    use log::Level::Trace;

    numeric_tests!(
        f64,
        f64,
        test_values_are_positive,
        test_is_exact,
        test_ceil_precision,
        test_ratio,
        test_ratio_invert => fail(r"assertion `left == right` failed: R::ratio(1, a) * a != 1 for 49
  left: 0.9999999999999999
 right: 1.0"),
        test_is_zero,
        test_zero_is_add_neutral,
        test_add_is_commutative,
        test_opposite,
        test_sub_self,
        test_add_sub => fail(r"assertion `left == right` failed: (a + b) - b != a for 1, 0.0000000004656613430357376
  left: 0.9999999999999999
 right: 1.0"),
        test_sub_add => fail(r"assertion `left == right` failed: (a - b) + b != a for 0.00000011920928955078125, 2147483646
  left: 0.0
 right: 1.1920928955078125e-7"),
        test_one_is_mul_neutral,
        test_mul_is_commutative,
        test_mul_up_is_commutative,
        test_mul_up_integers,
        test_mul_up_wrt_mul,
        test_invert => fail(r"assertion `left == right` failed: 1/(1/a) != a for 2147483631
  left: 2147483631.0000002
 right: 2147483631.0"),
        test_div_self,
        test_div_up_self,
        test_div_up_wrt_div,
        test_mul_div => fail(r"assertion `left == right` failed: (a * b) / b != a for 2147483646, 0.000000000465661315280157
  left: 2147483646.0000002
 right: 2147483646.0"),
        test_div_mul => fail(r"assertion `left == right` failed: (a / b) * b != a for 1, 2147483631
  left: 0.9999999999999999
 right: 1.0"),
        test_mul_by_int,
        test_div_by_int,
        test_references,
        test_assign,
    );

    big_numeric_tests!(
        f64,
        f64,
        None,
        test_add_is_associative => fail(r"assertion `left == right` failed: (a + b) + c != a + (b + c) for 1, 1, 0.0000000004656615095692907
  left: 2.0000000004656617
 right: 2.0000000004656613"),
        test_mul_is_associative => fail(r"assertion `left == right` failed: (a * b) * c != a * (b * c) for 2147483646, 2147483646, 2147483583
  left: 9.903519996076708e27
 right: 9.903519996076707e27"),
        test_mul_is_distributive => fail(r"assertion `left == right` failed: a * (b + c) != (a * b) + (a * c) for 2147483646, 1, 2147483519
  left: 4.6116857392545137e18
 right: 4.611685739254514e18"),
        test_mul_by_int_is_associative => fail(r"assertion `left == right` failed: (a * b) * c != a * (b * c) for 0.0000000004656612944634737, 3, 3
  left: 4.190951650171264e-9
 right: 4.190951650171263e-9"),
        test_mul_by_int_is_distributive => fail(r"assertion `left == right` failed: (a + b) * c != (a * c) + (b * c) for 1, 0.0000000004656613985469087, 3
  left: 3.0000000013969848
 right: 3.0000000013969843"),
        test_div_by_int_is_associative => fail(r"assertion `left == right` failed: (a / b) / c != a / (b * c) for 1, 3, 11
  left: 0.0303030303030303
 right: 0.030303030303030304"),
        test_div_by_int_is_distributive => fail(r"assertion `left == right` failed: (a + b) / c != (a / c) + (b / c) for 1, 2, 5
  left: 0.6
 right: 0.6000000000000001"),
        test_sum,
        test_product,
    );

    numeric_benchmarks!(f64, f64, bench_add, bench_sub, bench_mul, bench_div,);

    #[test]
    fn test_description() {
        assert_eq!(f64::description(), "64-bit floating-point arithmetic");
    }

    #[test]
    fn test_assert_eq() {
        let logger = ThreadLocalLogger::start();
        f64::assert_eq(0.0, 0.0, "Error message");
        logger.check_target_logs("stv_rs::arithmetic::float64", []);

        let logger = ThreadLocalLogger::start();
        f64::assert_eq(1.0, 1.0 + f64::EPSILON, "Error message");
        logger.check_target_logs(
            "stv_rs::arithmetic::float64",
            [(
                Trace,
                "Error message: Failed comparison 1 != 1.0000000000000002 (error = 1 * eps)",
            )],
        );

        let logger = ThreadLocalLogger::start();
        f64::assert_eq(1.0, 1.0 + 999.0 * f64::EPSILON, "Error message");
        logger.check_target_logs(
            "stv_rs::arithmetic::float64",
            [(
                Trace,
                "Error message: Failed comparison 1 != 1.0000000000002218 (error = 999 * eps)",
            )],
        );

        let logger = ThreadLocalLogger::start();
        f64::assert_eq(1.0, 1.0 + 1000.0 * f64::EPSILON, "Error message");
        logger.check_target_logs(
            "stv_rs::arithmetic::float64",
            [(
                Trace,
                "Error message: Failed comparison 1 != 1.000000000000222 (error = 1000 * eps)",
            )],
        );
    }

    #[test]
    #[should_panic(
        expected = "Error message: Failed comparison 1 != 1.0000000000002223 (error = 1001 * eps)"
    )]
    fn test_assert_ne() {
        f64::assert_eq(1.0, 1.0 + 1001.0 * f64::EPSILON, "Error message");
    }

    #[test]
    fn test_display_test_values() {
        #[rustfmt::skip]
        let expected_displays = [
          "1", "2", "4", "8", "16", "32", "64", "128", "256", "512", "1024", "2048", "4096", "8192",
          "16384", "32768", "65536", "131072", "262144", "524288", "1048576", "2097152", "4194304",
          "8388608", "16777216", "33554432", "67108864", "134217728", "268435456", "536870912",
          "1073741824", "2147483646", "2147483645", "2147483643", "2147483639", "2147483631",
          "2147483615", "2147483583", "2147483519", "2147483391", "2147483135", "2147482623",
          "2147481599", "2147479551", "2147475455", "2147467263", "2147450879", "2147418111",
          "2147352575", "2147221503", "2146959359", "2146435071", "2145386495", "2143289343",
          "2139095039", "2130706431", "2113929215", "2080374783", "2013265919", "1879048191",
          "1610612735", "1073741823",
          "1", "0.5", "0.25", "0.125", "0.0625", "0.03125", "0.015625", "0.0078125", "0.00390625",
          "0.001953125", "0.0009765625", "0.00048828125", "0.000244140625", "0.0001220703125",
          "0.00006103515625", "0.000030517578125", "0.0000152587890625", "0.00000762939453125",
          "0.000003814697265625", "0.0000019073486328125", "0.00000095367431640625",
          "0.000000476837158203125", "0.0000002384185791015625", "0.00000011920928955078125",
          "0.00000005960464477539063", "0.000000029802322387695313", "0.000000014901161193847656",
          "0.000000007450580596923828", "0.000000003725290298461914", "0.000000001862645149230957",
          "0.0000000009313225746154785", "0.0000000004656612877414201",
          "0.0000000004656612879582606", "0.0000000004656612883919414",
          "0.0000000004656612892593032", "0.0000000004656612909940266",
          "0.0000000004656612944634737", "0.0000000004656613014023679",
          "0.000000000465661315280157", "0.0000000004656613430357376",
          "0.0000000004656613985469087", "0.0000000004656615095692907",
          "0.0000000004656617316142135", "0.0000000004656621757046943",
          "0.000000000465663063888197", "0.0000000004656648402653671",
          "0.0000000004656683930603658", "0.0000000004656754988130022",
          "0.0000000004656897109688659", "0.0000000004657181378832345",
          "0.0000000004657750021247607", "0.0000000004658887722767739",
          "0.0000000004661164793992049", "0.000000000466572562060278",
          "0.0000000004674874102216082", "0.0000000004693279118375177",
          "0.0000000004730527365364029", "0.0000000004806826193874318",
          "0.0000000004967053733749714", "0.0000000005321843286349222",
          "0.0000000006208817167958132", "0.0000000009313225754828403"
        ];
        let actual_displays: Vec<String> = f64::get_positive_test_values()
            .iter()
            .map(|x| format!("{x}"))
            .collect();
        assert_eq!(actual_displays, expected_displays);
    }
}
