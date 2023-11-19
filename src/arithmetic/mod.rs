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

//! Module providing traits to abstract over the arithmetic needed for STV, and
//! various implementations of this arithmetic.

mod approx_rational;
mod exact;
mod fixed;
mod fixed_big;
mod float64;
#[cfg(test)]
mod ratio_i64;

pub use approx_rational::ApproxRational;
pub use fixed::{FixedDecimal9, Integer64};
pub use fixed_big::BigFixedDecimal9;

use num::traits::{One, Zero};
use std::fmt::{Debug, Display};
use std::iter::{Product, Sum};
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Sub, SubAssign};

/// Trait representing integer arithmetic. This is a lighter version of the
/// `Integer` trait provided by the [`num` crate](https://crates.io/crates/num),
/// here we only consider the arithmetic operations needed for STV.
pub trait Integer:
    Clone
    + Debug
    + PartialEq
    + PartialOrd
    + Zero
    + One
    + Product
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
where
    for<'a> &'a Self: IntegerRef<Self>,
{
    /// Obtains an integer from a primitive `usize` integer.
    fn from_usize(i: usize) -> Self;

    #[cfg(test)]
    fn get_positive_test_values() -> Vec<Self>;
}

/// Trait representing rational numbers w.r.t. an [`Integer`] type. Here we only
/// consider the arithmetic operations needed for STV.
pub trait Rational<I>:
    Clone
    + Display
    + Debug
    + PartialEq
    + PartialOrd
    + Zero
    + One
    + AddAssign
    + SubAssign
    + MulAssign
    + Sum
    + Product
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Mul<I, Output = Self>
    + Div<I, Output = Self>
    + for<'a> AddAssign<&'a Self>
    + for<'a> SubAssign<&'a Self>
    + for<'a> MulAssign<&'a Self>
    + for<'a> DivAssign<&'a I>
    + for<'a> Sum<&'a Self>
    + for<'a> Add<&'a Self, Output = Self>
    + for<'a> Sub<&'a Self, Output = Self>
    + for<'a> Mul<&'a Self, Output = Self>
where
    I: Integer,
    for<'a> &'a I: IntegerRef<I>,
    for<'a> &'a Self: RationalRef<&'a I, Self>,
{
    /// Obtains a number equal to the given integer.
    fn from_int(i: I) -> Self;

    /// Obtains a number equal to the given integer.
    fn from_usize(i: usize) -> Self {
        Self::from_int(I::from_usize(i))
    }

    /// Obtains a number equal to the ratio between the given numerator and
    /// denominator.
    fn ratio_i(num: I, denom: I) -> Self;

    /// Obtains a number equal to the ratio between the given numerator and
    /// denominator.
    fn ratio(num: usize, denom: usize) -> Self {
        Self::ratio_i(I::from_usize(num), I::from_usize(denom))
    }

    /// Converts a number into its floating-point approximation. This can be
    /// useful to print approximation of large numbers.
    fn to_f64(&self) -> f64;

    /// Allows to customize equality assertion to inexact types such as [`f64`].
    #[track_caller]
    fn assert_eq(a: Self, b: Self, msg: &str) {
        assert_eq!(a, b, "{msg}");
    }

    /// Minimal representable value for non-exact arithmetic.
    fn epsilon() -> Self;

    /// Whether this type represents exact arithmetic, i.e. [`Self::epsilon()`]
    /// is zero.
    fn is_exact() -> bool;

    /// Description of the implemented arithmetic, e.g. "64-bit floating point
    /// arithmetic".
    fn description() -> &'static str;

    /// Multiplication, rounding up for types that perform a rounding on this
    /// operation.
    fn mul_up(&self, rhs: &Self) -> Self {
        self * rhs
    }

    /// Performs a division to obtain a keep factor. The division is rounded up
    /// for types that cannot compute the result precisely. Additionally,
    /// types may further round the result, which can be useful with exact
    /// arithmetic, to avoid complexity explosion of rational numbers.
    fn div_up_as_keep_factor(&self, rhs: &Self) -> Self;

    #[cfg(test)]
    fn get_positive_test_values() -> Vec<Self>;
}

/// Helper trait that integer references implement.
pub trait IntegerRef<Output>:
    Sized + Add<Self, Output = Output> + Sub<Self, Output = Output> + Mul<Self, Output = Output>
{
}

/// Helper trait that rational references implement.
pub trait RationalRef<RefI, Output>:
    Sized
    + Add<Self, Output = Output>
    + Sub<Self, Output = Output>
    + Mul<Self, Output = Output>
    + Mul<RefI, Output = Output>
    + Div<RefI, Output = Output>
{
}

#[cfg(test)]
pub(crate) mod test {
    use super::*;
    use ::test::Bencher;
    use rand::distributions::{Distribution, Uniform};
    use rand::seq::SliceRandom;
    use rand::thread_rng;
    use std::hint::black_box;
    use std::marker::PhantomData;

    #[macro_export]
    macro_rules! integer_tests {
        ( $typei:ty, ) => {};
        ( $typei:ty, $case:ident, $( $others:tt )* ) => {
            #[test]
            fn $case() {
                $crate::arithmetic::test::IntegerTests::<$typei>::$case();
            }

            integer_tests!($typei, $($others)*);
        };
        ( $typei:ty, $case:ident => fail($msg:expr), $( $others:tt )* ) => {
            #[test]
            #[should_panic(expected = $msg)]
            fn $case() {
                $crate::arithmetic::test::IntegerTests::<$typei>::$case();
            }

            integer_tests!($typei, $($others)*);
        };
    }

    #[macro_export]
    macro_rules! big_integer_tests {
        ( $typei:ty, $num_samples:expr, ) => {};
        ( $typei:ty, $num_samples:expr, $case:ident, $( $others:tt )* ) => {
            #[test]
            fn $case() {
                $crate::arithmetic::test::IntegerTests::<$typei>::$case($num_samples);
            }

            big_integer_tests!($typei, $num_samples, $($others)*);
        };
        ( $typei:ty, $num_samples:expr, $case:ident => fail($msg:expr), $( $others:tt )* ) => {
            #[test]
            #[should_panic(expected = $msg)]
            fn $case() {
                $crate::arithmetic::test::IntegerTests::<$typei>::$case($num_samples);
            }

            big_integer_tests!($typei, $num_samples, $($others)*);
        };
    }

    #[macro_export]
    macro_rules! numeric_tests {
        ( $typei:ty, $typer:ty, ) => {};
        ( $typei:ty, $typer:ty, $case:ident, $( $others:tt )* ) => {
            #[test]
            fn $case() {
                $crate::arithmetic::test::NumericTests::<$typei, $typer>::$case();
            }

            numeric_tests!($typei, $typer, $($others)*);
        };
        ( $typei:ty, $typer:ty, $case:ident => fail($msg:expr), $( $others:tt )* ) => {
            #[test]
            #[should_panic(expected = $msg)]
            fn $case() {
                $crate::arithmetic::test::NumericTests::<$typei, $typer>::$case();
            }

            numeric_tests!($typei, $typer, $($others)*);
        };
    }

    #[macro_export]
    macro_rules! big_numeric_tests {
        ( $typei:ty, $typer:ty, $num_samples:expr, ) => {};
        ( $typei:ty, $typer:ty, $num_samples:expr, $case:ident, $( $others:tt )* ) => {
            #[test]
            fn $case() {
                $crate::arithmetic::test::NumericTests::<$typei, $typer>::$case($num_samples);
            }

            big_numeric_tests!($typei, $typer, $num_samples, $($others)*);
        };
        ( $typei:ty, $typer:ty, $num_samples:expr, $case:ident => fail($msg:expr), $( $others:tt )* ) => {
            #[test]
            #[should_panic(expected = $msg)]
            fn $case() {
                $crate::arithmetic::test::NumericTests::<$typei, $typer>::$case($num_samples);
            }

            big_numeric_tests!($typei, $typer, $num_samples, $($others)*);
        };
    }

    #[macro_export]
    macro_rules! numeric_benchmarks {
        ( $typei:ty, $typer:ty, ) => {};
        ( $typei:ty, $typer:ty, $case:ident, $( $others:tt )* ) => {
            #[bench]
            fn $case(b: &mut ::test::Bencher) {
                $crate::arithmetic::test::NumericTests::<$typei, $typer>::$case(b);
            }

            numeric_benchmarks!($typei, $typer, $($others)*);
        };
    }

    pub struct IntegerTests<I> {
        _phantomi: PhantomData<I>,
    }

    impl<I> IntegerTests<I>
    where
        I: Integer + Display,
        for<'a> &'a I: IntegerRef<I>,
    {
        pub fn testi_values_are_positive() {
            let test_values = I::get_positive_test_values();
            Self::loopi_check1(&test_values, |a| {
                assert!(*a > I::zero(), "{a} is not positive");
            });
        }

        pub fn testi_is_zero() {
            let test_values = I::get_positive_test_values();
            assert!(I::zero().is_zero());
            assert!(!I::one().is_zero());
            Self::loopi_check1(&test_values, |a| {
                assert!(!a.is_zero(), "{a} is zero");
            });
        }

        pub fn testi_zero_is_add_neutral() {
            let test_values = I::get_positive_test_values();
            Self::loopi_check1(&test_values, |a| {
                assert_eq!(a + &I::zero(), *a, "a + 0 != a for {a}");
                assert_eq!(&I::zero() + a, *a, "0 + a != a for {a}");
                assert_eq!(a - &I::zero(), *a, "a - 0 != a for {a}");
            })
        }

        #[allow(clippy::eq_op)]
        pub fn testi_add_is_commutative() {
            let test_values = I::get_positive_test_values();
            Self::loopi_check2(&test_values, |a, b| {
                assert_eq!(a + b, b + a, "a + b != b + a for {a}, {b}");
            })
        }

        pub fn testi_add_is_associative(num_samples: Option<usize>) {
            let test_values = I::get_positive_test_values();
            Self::loopi_check3(&test_values, num_samples, |a, b, c| {
                assert_eq!(
                    &(a + b) + c,
                    a + &(b + c),
                    "(a + b) + c != a + (b + c) for {a}, {b}, {c}"
                );
            })
        }

        pub fn testi_opposite() {
            let test_values = I::get_positive_test_values();
            Self::loopi_check1(&test_values, |a| {
                assert_eq!(
                    I::zero() - (I::zero() - a.clone()),
                    *a,
                    "-(-a) != a for {a}"
                );
            });
        }

        #[allow(clippy::eq_op)]
        pub fn testi_sub_self() {
            let test_values = I::get_positive_test_values();
            Self::loopi_check1(&test_values, |a| {
                assert_eq!(a - a, I::zero(), "a - a != 0 for {a}");
            });
        }

        pub fn testi_add_sub() {
            let test_values = I::get_positive_test_values();
            Self::loopi_check2(&test_values, |a, b| {
                assert_eq!(&(a + b) - b, *a, "(a + b) - b != a for {a}, {b}");
            });
        }

        pub fn testi_sub_add() {
            let test_values = I::get_positive_test_values();
            Self::loopi_check2(&test_values, |a, b| {
                assert_eq!(&(a - b) + b, *a, "(a - b) + b != a for {a}, {b}");
            });
        }

        pub fn testi_one_is_mul_neutral() {
            let test_values = I::get_positive_test_values();
            Self::loopi_check1(&test_values, |a| {
                assert_eq!(a * &I::one(), *a, "a * 1 != a for {a}");
                assert_eq!(&I::one() * a, *a, "1 * a != a for {a}");
            })
        }

        pub fn testi_mul_is_commutative() {
            let test_values = I::get_positive_test_values();
            Self::loopi_check2(&test_values, |a, b| {
                assert_eq!(a * b, b * a, "a * b != b * a for {a}, {b}");
            })
        }

        pub fn testi_mul_is_associative(num_samples: Option<usize>) {
            let test_values = I::get_positive_test_values();
            Self::loopi_check3(&test_values, num_samples, |a, b, c| {
                assert_eq!(
                    &(a * b) * c,
                    a * &(b * c),
                    "(a * b) * c != a * (b * c) for {a}, {b}, {c}"
                );
            })
        }

        pub fn testi_mul_is_distributive(num_samples: Option<usize>) {
            let test_values = I::get_positive_test_values();
            Self::loopi_check3(&test_values, num_samples, |a, b, c| {
                assert_eq!(
                    a * &(b + c),
                    &(a * b) + &(a * c),
                    "a * (b + c) != (a * b) + (a * c) for {a}, {b}, {c}"
                );
            })
        }

        pub fn testi_product(num_samples: Option<usize>) {
            let test_values = I::get_positive_test_values();
            Self::loopi_check3(&test_values, num_samples, |a, b, c| {
                assert_eq!(
                    [a.clone(), b.clone(), c.clone()].into_iter().product::<I>(),
                    &(a * b) * c,
                    "[a, b, c].product() != a * b * c for {a}, {b}, {c}"
                );
            });
        }

        fn loopi_check1(test_values: &[I], f: impl Fn(&I)) {
            for a in test_values {
                f(a);
            }
        }

        fn loopi_check2(test_values: &[I], f: impl Fn(&I, &I)) {
            for a in test_values {
                for b in test_values {
                    f(a, b);
                }
            }
        }

        fn loopi_check3(test_values: &[I], num_samples: Option<usize>, f: impl Fn(&I, &I, &I)) {
            match num_samples {
                None => {
                    // Exhaustive check.
                    for a in test_values {
                        for b in test_values {
                            for c in test_values {
                                f(a, b, c);
                            }
                        }
                    }
                }
                Some(n) => {
                    // Randomly sample values rather than conducting an exhaustive O(n^3) search on
                    // the test values.
                    let mut rng = thread_rng();

                    for _ in 0..n {
                        let a = test_values.choose(&mut rng).unwrap();
                        let b = test_values.choose(&mut rng).unwrap();
                        let c = test_values.choose(&mut rng).unwrap();
                        f(a, b, c);
                    }
                }
            }
        }
    }

    pub struct NumericTests<I, R> {
        _phantomi: PhantomData<I>,
        _phantomr: PhantomData<R>,
    }

    impl<I, R> NumericTests<I, R>
    where
        I: Integer + Display,
        for<'a> &'a I: IntegerRef<I>,
        R: Rational<I>,
        for<'a> &'a R: RationalRef<&'a I, R>,
    {
        pub fn test_values_are_positive() {
            let test_values = R::get_positive_test_values();
            Self::loop_check1(&test_values, |a| {
                assert!(*a > R::zero(), "{a} is not positive");
            });
        }

        pub fn test_is_exact() {
            assert!(
                R::is_exact() == R::epsilon().is_zero(),
                "epsilon is {} but is_exact() returns {}",
                R::epsilon(),
                R::is_exact()
            );
        }

        pub fn test_ratio() {
            Self::loop_check_i1(|a| {
                assert_eq!(
                    R::ratio_i(a.clone(), I::from_usize(1)),
                    R::from_int(a.clone()),
                    "R::ratio(a, 1) != a for {a}"
                );
            });
        }

        pub fn test_ratio_invert() {
            Self::loop_check_i1(|a| {
                if !a.is_zero() {
                    assert_eq!(
                        R::ratio_i(I::from_usize(1), a.clone()) * R::from_int(a.clone()),
                        R::one(),
                        "R::ratio(1, a) * a != 1 for {a}"
                    );
                }
            });
        }

        pub fn test_is_zero() {
            let test_values = R::get_positive_test_values();
            assert!(R::zero().is_zero());
            assert!(!R::one().is_zero());
            Self::loop_check1(&test_values, |a| {
                assert!(!a.is_zero(), "{a} is zero");
            });
        }

        pub fn test_zero_is_add_neutral() {
            let test_values = R::get_positive_test_values();
            Self::loop_check1(&test_values, |a| {
                assert_eq!(a + &R::zero(), *a, "a + 0 != a for {a}");
                assert_eq!(&R::zero() + a, *a, "0 + a != a for {a}");
                assert_eq!(a - &R::zero(), *a, "a - 0 != a for {a}");
            })
        }

        #[allow(clippy::eq_op)]
        pub fn test_add_is_commutative() {
            let test_values = R::get_positive_test_values();
            Self::loop_check2(&test_values, |a, b| {
                assert_eq!(a + b, b + a, "a + b != b + a for {a}, {b}");
            })
        }

        pub fn test_add_is_associative(num_samples: Option<usize>) {
            let test_values = R::get_positive_test_values();
            Self::loop_check3(&test_values, num_samples, |a, b, c| {
                assert_eq!(
                    &(a + b) + c,
                    a + &(b + c),
                    "(a + b) + c != a + (b + c) for {a}, {b}, {c}"
                );
            })
        }

        pub fn test_opposite() {
            let test_values = R::get_positive_test_values();
            Self::loop_check1(&test_values, |a| {
                assert_eq!(R::zero() - (R::zero() - a), *a, "-(-a) != a for {a}");
            });
        }

        #[allow(clippy::eq_op)]
        pub fn test_sub_self() {
            let test_values = R::get_positive_test_values();
            Self::loop_check1(&test_values, |a| {
                assert_eq!(a - a, R::zero(), "a - a != 0 for {a}");
            });
        }

        pub fn test_add_sub() {
            let test_values = R::get_positive_test_values();
            Self::loop_check2(&test_values, |a, b| {
                assert_eq!(&(a + b) - b, *a, "(a + b) - b != a for {a}, {b}");
            });
        }

        pub fn test_sub_add() {
            let test_values = R::get_positive_test_values();
            Self::loop_check2(&test_values, |a, b| {
                assert_eq!(&(a - b) + b, *a, "(a - b) + b != a for {a}, {b}");
            });
        }

        pub fn test_one_is_mul_neutral() {
            let test_values = R::get_positive_test_values();
            Self::loop_check1(&test_values, |a| {
                assert_eq!(a * &R::one(), *a, "a * 1 != a for {a}");
                assert_eq!(&R::one() * a, *a, "1 * a != a for {a}");
            })
        }

        pub fn test_mul_is_commutative() {
            let test_values = R::get_positive_test_values();
            Self::loop_check2(&test_values, |a, b| {
                assert_eq!(a * b, b * a, "a * b != b * a for {a}, {b}");
            })
        }

        pub fn test_mul_is_associative(num_samples: Option<usize>) {
            let test_values = R::get_positive_test_values();
            Self::loop_check3(&test_values, num_samples, |a, b, c| {
                assert_eq!(
                    &(a * b) * c,
                    a * &(b * c),
                    "(a * b) * c != a * (b * c) for {a}, {b}, {c}"
                );
            })
        }

        pub fn test_mul_is_distributive(num_samples: Option<usize>) {
            let test_values = R::get_positive_test_values();
            Self::loop_check3(&test_values, num_samples, |a, b, c| {
                assert_eq!(
                    a * &(b + c),
                    &(a * b) + &(a * c),
                    "a * (b + c) != (a * b) + (a * c) for {a}, {b}, {c}"
                );
            })
        }

        pub fn test_mul_up_is_commutative() {
            let test_values = R::get_positive_test_values();
            Self::loop_check2(&test_values, |a, b| {
                assert_eq!(
                    R::mul_up(a, b),
                    R::mul_up(b, a),
                    "mul_up(a, b) != mul_up(b, a) for {a}, {b}"
                );
            })
        }

        pub fn test_mul_up_integers() {
            Self::loop_check_i2(|a, b| {
                assert_eq!(
                    R::mul_up(&R::from_int(a.clone()), &R::from_int(b.clone())),
                    R::from_int(a.clone() * b.clone()),
                    "mul_up(a, b) != a * b for {a}, {b}"
                );
            })
        }

        pub fn test_mul_up_wrt_mul() {
            let test_values = R::get_positive_test_values();
            Self::loop_check2(&test_values, |a, b| {
                let mul = a * b;
                let mul_up = R::mul_up(a, b);
                assert!(mul_up >= mul, "mul_up(a, b) < a * b for {a}, {b}");
                assert!(
                    mul_up <= mul + R::epsilon(),
                    "mul_up(a, b) > a * b + epsilon for {a}, {b}"
                );
            })
        }

        pub fn test_one_is_div_up_neutral() {
            let test_values = R::get_positive_test_values();
            Self::loop_check1(&test_values, |a| {
                assert_eq!(
                    R::div_up_as_keep_factor(a, &R::one()),
                    *a,
                    "div_up(a, 1) != a for {a}"
                );
            })
        }

        pub fn test_div_up_self() {
            let test_values = R::get_positive_test_values();
            Self::loop_check1(&test_values, |a| {
                assert_eq!(
                    R::div_up_as_keep_factor(a, a),
                    R::one(),
                    "div_up(a, a) != 1 for {a}"
                );
            });
        }

        pub fn test_mul_div_up() {
            let test_values = R::get_positive_test_values();
            Self::loop_check2(&test_values, |a, b| {
                assert_eq!(
                    R::div_up_as_keep_factor(&(a * b), b),
                    *a,
                    "div_up(a * b, b) != a for {a}, {b}"
                );
            });
        }

        pub fn test_mul_by_int() {
            let test_values = R::get_positive_test_values();
            Self::loop_check1_i1(&test_values, |a, b| {
                assert_eq!(
                    a * &b,
                    a * &R::from_int(b.clone()),
                    "a * int(b) != a * b for {a}, {b}"
                );
            })
        }

        pub fn test_mul_div_by_int() {
            let test_values = R::get_positive_test_values();
            Self::loop_check1_i1(&test_values, |a, b| {
                if !b.is_zero() {
                    assert_eq!(&(a * &b) / &b, *a, "a * int(b) / int(b) != a for {a}, {b}");
                }
            })
        }

        pub fn test_mul_by_int_is_associative(num_samples: Option<usize>) {
            let test_values = R::get_positive_test_values();
            Self::loop_check1_i2(&test_values, num_samples, |a, b, c| {
                assert_eq!(
                    &(a * &b) * &c,
                    a * &(b.clone() * c.clone()),
                    "(a * b) * c != a * (b * c) for {a}, {b}, {c}"
                );
            })
        }

        pub fn test_mul_by_int_is_distributive(num_samples: Option<usize>) {
            let test_values = R::get_positive_test_values();
            Self::loop_check2_i1(&test_values, num_samples, |a: &R, b: &R, c: I| {
                assert_eq!(
                    &(a + b) * &c,
                    &(a * &c) + &(b * &c),
                    "(a + b) * c != (a * c) + (b * c) for {a}, {b}, {c}"
                );
            })
        }

        pub fn test_div_by_int_is_associative(num_samples: Option<usize>) {
            let test_values = R::get_positive_test_values();
            Self::loop_check1_i2(&test_values, num_samples, |a, b, c| {
                if !b.is_zero() && !c.is_zero() {
                    assert_eq!(
                        &(a / &b) / &c,
                        a / &(b.clone() * c.clone()),
                        "(a / b) / c != a / (b * c) for {a}, {b}, {c}"
                    );
                }
            })
        }

        pub fn test_div_by_int_is_distributive(num_samples: Option<usize>) {
            let test_values = R::get_positive_test_values();
            Self::loop_check2_i1(&test_values, num_samples, |a: &R, b: &R, c: I| {
                if !c.is_zero() {
                    assert_eq!(
                        &(a + b) / &c,
                        &(a / &c) + &(b / &c),
                        "(a + b) / c != (a / c) + (b / c) for {a}, {b}, {c}"
                    );
                }
            })
        }

        pub fn test_references() {
            let test_values = R::get_positive_test_values();

            Self::loop_check2(&test_values, |a, b| {
                assert_eq!(
                    a + b,
                    a.clone() + b.clone(),
                    "&a + &b != a + b for {a}, {b}"
                );
                assert_eq!(
                    a - b,
                    a.clone() - b.clone(),
                    "&a - &b != a - b for {a}, {b}"
                );
                assert_eq!(
                    a * b,
                    a.clone() * b.clone(),
                    "&a * &b != a * b for {a}, {b}"
                );
                assert_eq!(a + b, a.clone() + b, "&a + &b != a + &b for {a}, {b}");
                assert_eq!(a - b, a.clone() - b, "&a - &b != a - &b for {a}, {b}");
                assert_eq!(a * b, a.clone() * b, "&a * &b != a * &b for {a}, {b}");
            });

            Self::loop_check1_i1(&test_values, |a, b| {
                if !b.is_zero() {
                    assert_eq!(
                        a * &b,
                        a.clone() * b.clone(),
                        "&a * &b != a * b for {a}, {b}"
                    );
                    assert_eq!(
                        a / &b,
                        a.clone() / b.clone(),
                        "&a / &b != a / b for {a}, {b}"
                    );
                }
            });
        }

        pub fn test_assign() {
            let test_values = R::get_positive_test_values();

            Self::loop_check2(&test_values, |a, b| {
                let mut c = a.clone();
                c += b.clone();
                assert_eq!(c, a + b, "a += b != a + b for {a}, {b}");
                let mut c = a.clone();
                c -= b.clone();
                assert_eq!(c, a - b, "a -= b != a - b for {a}, {b}");
                let mut c = a.clone();
                c *= b.clone();
                assert_eq!(c, a * b, "a *= b != a * b for {a}, {b}");

                let mut c = a.clone();
                c += b;
                assert_eq!(c, a + b, "a += &b != a + b for {a}, {b}");
                let mut c = a.clone();
                c -= b;
                assert_eq!(c, a - b, "a -= &b != a - b for {a}, {b}");
                let mut c = a.clone();
                c *= b;
                assert_eq!(c, a * b, "a *= &b != a * b for {a}, {b}");
            });

            Self::loop_check1_i1(&test_values, |a, b| {
                if !b.is_zero() {
                    let mut c = a.clone();
                    c /= &b;
                    assert_eq!(c, a / &b, "a /= &b != a / &b for {a}, {b}");
                }
            });
        }

        pub fn test_sum(num_samples: Option<usize>) {
            let test_values = R::get_positive_test_values();
            Self::loop_check3(&test_values, num_samples, |a, b, c| {
                assert_eq!(
                    [a.clone(), b.clone(), c.clone()].into_iter().sum::<R>(),
                    &(a + b) + c,
                    "[a, b, c].sum() != a + b + c for {a}, {b}, {c}"
                );
                assert_eq!(
                    [a, b, c].into_iter().sum::<R>(),
                    &(a + b) + c,
                    "[&a, &b, &c].sum() != a + b + c for {a}, {b}, {c}"
                );
            });
        }

        pub fn test_product(num_samples: Option<usize>) {
            let test_values = R::get_positive_test_values();
            Self::loop_check3(&test_values, num_samples, |a, b, c| {
                assert_eq!(
                    [a.clone(), b.clone(), c.clone()].into_iter().product::<R>(),
                    &(a * b) * c,
                    "[a, b, c].product() != a * b * c for {a}, {b}, {c}"
                );
            });
        }

        pub fn bench_add(bencher: &mut Bencher) {
            let a = R::zero();
            let b = R::one();
            bencher.iter(|| black_box(&a) + black_box(&b));
        }

        pub fn bench_sub(bencher: &mut Bencher) {
            let a = R::zero();
            let b = R::one();
            bencher.iter(|| black_box(&a) - black_box(&b));
        }

        pub fn bench_mul(bencher: &mut Bencher) {
            let a = R::zero();
            let b = R::one();
            bencher.iter(|| black_box(&a) * black_box(&b));
        }

        pub fn bench_div_up(bencher: &mut Bencher) {
            let a = R::zero();
            let b = R::one();
            bencher.iter(|| R::div_up_as_keep_factor(black_box(&a), black_box(&b)));
        }

        fn loop_check1(test_values: &[R], f: impl Fn(&R)) {
            for a in test_values {
                f(a);
            }
        }

        fn loop_check_i1(f: impl Fn(I)) {
            for aa in 0..100 {
                let a = I::from_usize(aa);
                f(a);
            }
        }

        fn loop_check2(test_values: &[R], f: impl Fn(&R, &R)) {
            for a in test_values {
                for b in test_values {
                    f(a, b);
                }
            }
        }

        fn loop_check1_i1(test_values: &[R], f: impl Fn(&R, I)) {
            for a in test_values {
                for bb in 0..100 {
                    let b = I::from_usize(bb);
                    f(a, b);
                }
            }
        }

        fn loop_check_i2(f: impl Fn(I, I)) {
            for aa in 0..100 {
                for bb in 0..100 {
                    let a = I::from_usize(aa);
                    let b = I::from_usize(bb);
                    f(a, b);
                }
            }
        }

        fn loop_check3(test_values: &[R], num_samples: Option<usize>, f: impl Fn(&R, &R, &R)) {
            match num_samples {
                None => {
                    // Exhaustive check.
                    for a in test_values {
                        for b in test_values {
                            for c in test_values {
                                f(a, b, c);
                            }
                        }
                    }
                }
                Some(n) => {
                    // Randomly sample values rather than conducting an exhaustive O(n^3) search on
                    // the test values.
                    let mut rng = thread_rng();

                    for _ in 0..n {
                        let a = test_values.choose(&mut rng).unwrap();
                        let b = test_values.choose(&mut rng).unwrap();
                        let c = test_values.choose(&mut rng).unwrap();
                        f(a, b, c);
                    }
                }
            }
        }

        fn loop_check2_i1(test_values: &[R], num_samples: Option<usize>, f: impl Fn(&R, &R, I)) {
            match num_samples {
                None => {
                    // Exhaustive check.
                    for a in test_values {
                        for b in test_values {
                            for cc in 0..100 {
                                let c = I::from_usize(cc);
                                f(a, b, c);
                            }
                        }
                    }
                }
                Some(n) => {
                    // Randomly sample values rather than conducting an exhaustive O(n^3) search on
                    // the test values.
                    let integer_distribution = Uniform::from(0..100);
                    let mut rng = thread_rng();

                    for _ in 0..n {
                        let a = test_values.choose(&mut rng).unwrap();
                        let b = test_values.choose(&mut rng).unwrap();
                        let c = I::from_usize(integer_distribution.sample(&mut rng));
                        f(a, b, c);
                    }
                }
            }
        }

        fn loop_check1_i2(test_values: &[R], num_samples: Option<usize>, f: impl Fn(&R, I, I)) {
            match num_samples {
                None => {
                    // Exhaustive check.
                    for a in test_values {
                        for bb in 0..100 {
                            for cc in 0..100 {
                                let b = I::from_usize(bb);
                                let c = I::from_usize(cc);
                                f(a, b, c);
                            }
                        }
                    }
                }
                Some(n) => {
                    // Randomly sample values rather than conducting an exhaustive O(n^3) search on
                    // the test values.
                    let integer_distribution = Uniform::from(0..100);
                    let mut rng = thread_rng();

                    for _ in 0..n {
                        let a = test_values.choose(&mut rng).unwrap();
                        let b = I::from_usize(integer_distribution.sample(&mut rng));
                        let c = I::from_usize(integer_distribution.sample(&mut rng));
                        f(a, b, c);
                    }
                }
            }
        }
    }
}
