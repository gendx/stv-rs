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

mod test {
    use crate::{big_numeric_tests, numeric_benchmarks, numeric_tests};
    use num::rational::Ratio;

    numeric_tests!(
        i64,
        Ratio<i64>,
        test_values_are_positive,
        test_is_exact,
        test_ceil_precision,
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
        test_invert,
        test_div_self,
        test_div_up_self,
        test_div_up_wrt_div,
        test_mul_div,
        test_div_mul,
        test_mul_by_int,
        test_div_by_int,
        test_references,
        test_assign,
    );

    big_numeric_tests!(
        i64,
        Ratio<i64>,
        None,
        test_add_is_associative => fail(r"assertion failed: `(left == right)`
  left: `Ratio { numer: -6148914815790568570, denom: 1537228657060915911 }`,
 right: `Ratio { numer: 4099276419306327682, denom: 512409552353638637 }`: (a + b) + c != a + (b + c) for 8, 1/2147483643, 1/2147483631"),
        test_mul_is_associative => fail(r"denominator == 0"),
        test_mul_is_distributive => fail(r"assertion failed: `(left == right)`
  left: `Ratio { numer: -4294967291, denom: 42949672936 }`,
 right: `Ratio { numer: 5, denom: 21474836468 }`: a * (b + c) != (a * b) + (a * c) for 1/4, 1/2147483646, 1/2147483645"),
        test_mul_by_int_is_associative,
        test_mul_by_int_is_distributive,
        test_div_by_int_is_associative,
        test_div_by_int_is_distributive => fail(r"assertion failed: `(left == right)`
  left: `Ratio { numer: -738197503, denom: 3026418953955049472 }`,
 right: `Ratio { numer: 2146451207, denom: 9079256861865148416 }`: (a + b) / c != (a / c) + (b / c) for 1/67108864, 1/2147483645, 65"),
        test_sum,
        test_product => fail(r"denominator == 0"),
    );

    numeric_benchmarks!(i64, Ratio<i64>, bench_add, bench_sub, bench_mul, bench_div,);
}
