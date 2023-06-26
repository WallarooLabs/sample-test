//! Test utilities for [`sample_std::Sample`].
//!
//! It is a direct port of [`quickcheck::QuickCheck`], with some key differences:
//!
//! - We use the [`Debug`] impl of tuples whose parts impl [`Debug`]. This means we
//!   can create a single general [`Testable::shrink`] definition.
//! - We use an iterative shrinking process instead of a recursive one (see
//!   [`Testable::shrink`]). This allows us to halt after a fixed number of
//!   shrinking steps, which sidesteps accidental infinite shrinking
//!   implementations and avoids the potential for stack overflows.
use std::cmp;
use std::env;
use std::fmt::Debug;
use std::panic;

use sample_std::{Random, Sample};

use crate::tester::Status::{Discard, Fail, Pass};
use crate::{error, info, trace};

/// The main [SampleTest] type for setting configuration and running sample-based testing.
pub struct SampleTest {
    tests: u64,
    max_tests: u64,
    min_tests_passed: u64,
    gen: Random,
}

fn st_tests() -> u64 {
    let default = 100;
    match env::var("SAMPLE_TEST_TESTS") {
        Ok(val) => val.parse().unwrap_or(default),
        Err(_) => default,
    }
}

fn st_max_tests() -> u64 {
    let default = 10_000;
    match env::var("SAMPLE_TEST_MAX_TESTS") {
        Ok(val) => val.parse().unwrap_or(default),
        Err(_) => default,
    }
}

fn st_min_tests_passed() -> u64 {
    let default = 0;
    match env::var("SAMPLE_TEST_MIN_TESTS_PASSED") {
        Ok(val) => val.parse().unwrap_or(default),
        Err(_) => default,
    }
}

impl SampleTest {
    /// Creates a new [SampleTest] value.
    ///
    /// This can be used to run [SampleTest] on things that implement [Testable].
    /// You may also adjust the configuration, such as the number of tests to
    /// run.
    ///
    /// By default, the maximum number of passed tests is set to `100`, the max
    /// number of overall tests is set to `10000` and the generator is created
    /// with a size of `100`.
    pub fn new() -> SampleTest {
        let gen = Random::new();
        let tests = st_tests();
        let max_tests = cmp::max(tests, st_max_tests());
        let min_tests_passed = st_min_tests_passed();

        SampleTest {
            tests,
            max_tests,
            min_tests_passed,
            gen,
        }
    }

    /// Set the number of tests to run.
    ///
    /// This actually refers to the maximum number of *passed* tests that
    /// can occur. Namely, if a test causes a failure, future testing on that
    /// property stops. Additionally, if tests are discarded, there may be
    /// fewer than `tests` passed.
    pub fn tests(mut self, tests: u64) -> SampleTest {
        self.tests = tests;
        self
    }

    /// Set the maximum number of tests to run.
    ///
    /// The number of invocations of a property will never exceed this number.
    /// This is necessary to cap the number of tests because [SampleTest]
    /// properties can discard tests.
    pub fn max_tests(mut self, max_tests: u64) -> SampleTest {
        self.max_tests = max_tests;
        self
    }

    /// Set the minimum number of tests that needs to pass.
    ///
    /// This actually refers to the minimum number of *valid* *passed* tests
    /// that needs to pass for the property to be considered successful.
    pub fn min_tests_passed(mut self, min_tests_passed: u64) -> SampleTest {
        self.min_tests_passed = min_tests_passed;
        self
    }

    /// Tests a property and returns the result.
    ///
    /// The result returned is either the number of tests passed or a witness
    /// of failure.
    ///
    /// (If you're using Rust's unit testing infrastructure, then you'll
    /// want to use the `sample_test` method, which will `panic!` on failure.)
    pub fn sample_test_count<S, A>(&mut self, s: S, f: A) -> Result<u64, TestResult>
    where
        A: Testable<S>,
        S: Sample,
        S::Output: Clone + Debug,
    {
        let mut n_tests_passed = 0;
        for _ in 0..self.max_tests {
            if n_tests_passed >= self.tests {
                break;
            }
            match f.test_once(&s, &mut self.gen) {
                TestResult { status: Pass, .. } => n_tests_passed += 1,
                TestResult {
                    status: Discard, ..
                } => continue,
                r @ TestResult { status: Fail, .. } => return Err(r),
            }
        }
        Ok(n_tests_passed)
    }

    /// Tests a property and calls `panic!` on failure.
    ///
    /// The `panic!` message will include a (hopefully) minimal witness of
    /// failure.
    ///
    /// It is appropriate to use this method with Rust's unit testing
    /// infrastructure.
    ///
    /// Note that if the environment variable `RUST_LOG` is set to enable
    /// `info` level log messages for the `sample_test` crate, then this will
    /// include output on how many [SampleTest] tests were passed.
    ///
    /// # Example
    ///
    /// ```rust
    /// use sample_test::{SampleTest};
    /// use sample_std::VecSampler;
    ///
    /// fn prop_reverse_reverse() {
    ///     fn revrev(xs: Vec<usize>) -> bool {
    ///         let rev: Vec<_> = xs.clone().into_iter().rev().collect();
    ///         let revrev: Vec<_> = rev.into_iter().rev().collect();
    ///         xs == revrev
    ///     }
    ///     let sampler = (VecSampler { length: (0..20), el: (0..100usize) },);
    ///     SampleTest::new().sample_test(sampler, revrev as fn(Vec<usize>) -> bool);
    /// }
    /// ```
    pub fn sample_test<S, A>(&mut self, s: S, f: A)
    where
        A: Testable<S>,
        S: Sample,
        S::Output: Clone + Debug,
    {
        // Ignore log init failures, implying it has already been done.
        let _ = crate::env_logger_init();

        let n_tests_passed = match self.sample_test_count(s, f) {
            Ok(n_tests_passed) => n_tests_passed,
            Err(result) => panic!("{}", result.failed_msg()),
        };

        if n_tests_passed >= self.min_tests_passed {
            info!("(Passed {} SampleTest tests.)", n_tests_passed)
        } else {
            panic!(
                "(Unable to generate enough tests, {} not discarded.)",
                n_tests_passed
            )
        }
    }
}

/// Convenience function for running [SampleTest].
///
/// This is an alias for `SampleTest::new().sample_test(f)`.
pub fn sample_test<S, A>(s: S, f: A)
where
    A: Testable<S>,
    S: Sample,
    S::Output: Clone + Debug,
{
    SampleTest::new().sample_test(s, f)
}

/// Describes the status of a single instance of a test.
///
/// All testable things must be capable of producing a `TestResult`.
#[derive(Clone, Debug)]
pub struct TestResult {
    status: Status,
    arguments: String,
    err: Option<String>,
}

/// Whether a test has passed, failed or been discarded.
#[derive(Clone, Debug)]
enum Status {
    Pass,
    Fail,
    Discard,
}

impl TestResult {
    /// Produces a test result that indicates the current test has passed.
    pub fn passed() -> TestResult {
        TestResult::from_bool(true)
    }

    /// Produces a test result that indicates the current test has failed.
    pub fn failed() -> TestResult {
        TestResult::from_bool(false)
    }

    /// Produces a test result that indicates failure from a runtime error.
    pub fn error<S: Into<String>>(msg: S) -> TestResult {
        let mut r = TestResult::from_bool(false);
        r.err = Some(msg.into());
        r
    }

    /// Produces a test result that instructs `sample_test` to ignore it.
    /// This is useful for restricting the domain of your properties.
    /// When a test is discarded, `sample_test` will replace it with a
    /// fresh one (up to a certain limit).
    pub fn discard() -> TestResult {
        TestResult {
            status: Discard,
            arguments: String::from(""),
            err: None,
        }
    }

    /// Converts a `bool` to a `TestResult`. A `true` value indicates that
    /// the test has passed and a `false` value indicates that the test
    /// has failed.
    pub fn from_bool(b: bool) -> TestResult {
        TestResult {
            status: if b { Pass } else { Fail },
            arguments: String::from(""),
            err: None,
        }
    }

    /// Tests if a "procedure" fails when executed. The test passes only if
    /// `f` generates a task failure during its execution.
    pub fn must_fail<T, F>(f: F) -> TestResult
    where
        F: FnOnce() -> T,
        F: 'static,
        T: 'static,
    {
        let f = panic::AssertUnwindSafe(f);
        TestResult::from_bool(panic::catch_unwind(f).is_err())
    }

    /// Returns `true` if and only if this test result describes a successful
    /// test.
    pub fn is_success(&self) -> bool {
        match self.status {
            Pass => true,
            Fail | Discard => false,
        }
    }

    /// Returns `true` if and only if this test result describes a failing
    /// test.
    pub fn is_failure(&self) -> bool {
        match self.status {
            Fail => true,
            Pass | Discard => false,
        }
    }

    /// Returns `true` if and only if this test result describes a failing
    /// test as a result of a run time error.
    pub fn is_error(&self) -> bool {
        self.is_failure() && self.err.is_some()
    }

    pub fn arguments(&self) -> &str {
        &self.arguments
    }

    fn failed_msg(&self) -> String {
        match self.err {
            None => format!("[sample_test] TEST FAILED. Arguments: ({})", self.arguments),
            Some(ref err) => format!(
                "[sample_test] TEST FAILED (runtime error). \
                 Arguments: ({})\nError: {}",
                self.arguments, err
            ),
        }
    }
}

/// `Testable` describes types (e.g., a function) whose values can be
/// tested.
///
/// Anything that can be tested must be capable of producing a [TestResult]
/// from the output of an instance of [Sample].
///
/// It's unlikely that you'll have to implement this trait yourself.
pub trait Testable<S>: 'static
where
    S: Sample,
{
    /// Report a [`TestResult`] from a given value.
    fn result(&self, v: S::Output) -> TestResult;

    /// Convenience function for running this [`Testable`] once on a random
    /// value, and shrinking any failures.
    fn test_once(&self, s: &S, rng: &mut Random) -> TestResult
    where
        S::Output: Clone + Debug,
    {
        let v = Sample::generate(s, rng);
        let r = self.result(v.clone());
        match r.status {
            Pass | Discard => r,
            Fail => {
                error!("{:?}", r);
                self.shrink(s, r, v)
            }
        }
    }

    /// Iteratively shrink the given test result until the iteration limit is
    /// reached or no further shrinkage is possible.
    fn shrink(&self, s: &S, r: TestResult, v: S::Output) -> TestResult
    where
        S::Output: Clone + Debug,
    {
        trace!("shrinking {:?}", v);
        let mut result = r;
        let mut it = s.shrink(v);
        let iterations = 10_000_000;

        for _ in 0..iterations {
            let sv = it.next();
            if let Some(sv) = sv {
                let r_new = self.result(sv.clone());
                if r_new.is_failure() {
                    trace!("shrinking {:?}", sv);
                    result = r_new;
                    it = s.shrink(sv);
                }
            } else {
                return result;
            }
        }

        trace!(
            "halting shrinkage after {} iterations with: {:?}",
            iterations,
            result
        );

        result
    }
}

impl From<bool> for TestResult {
    fn from(value: bool) -> TestResult {
        TestResult::from_bool(value)
    }
}

impl From<()> for TestResult {
    fn from(_: ()) -> TestResult {
        TestResult::passed()
    }
}

impl<A, E> From<Result<A, E>> for TestResult
where
    TestResult: From<A>,
    E: Debug + 'static,
{
    fn from(value: Result<A, E>) -> TestResult {
        match value {
            Ok(r) => r.into(),
            Err(err) => TestResult::error(format!("{:?}", err)),
        }
    }
}

macro_rules! testable_fn {
    ($($name: ident),*) => {

impl<T: 'static, S, $($name),*> Testable<S> for fn($($name),*) -> T
where
    TestResult: From<T>,
    S: Sample<Output=($($name),*,)>,
    ($($name),*,): Clone,
    $($name: Debug + 'static),*
{
    #[allow(non_snake_case)]
    fn result(&self, v: S::Output) -> TestResult {
        let ( $($name,)* ) = v.clone();
        let f: fn($($name),*) -> T = *self;
        let mut r = <TestResult as From<Result<T, String>>>::from(safe(move || {f($($name),*)}));

        {
            let ( $(ref $name,)* ) = v;
            r.arguments = format!("{:?}", &($($name),*));
        }
        r
    }
}}}

testable_fn!(A);
testable_fn!(A, B);
testable_fn!(A, B, C);
testable_fn!(A, B, C, D);
testable_fn!(A, B, C, D, E);
testable_fn!(A, B, C, D, E, F);
testable_fn!(A, B, C, D, E, F, G);
testable_fn!(A, B, C, D, E, F, G, H);

fn safe<T, F>(fun: F) -> Result<T, String>
where
    F: FnOnce() -> T,
    F: 'static,
    T: 'static,
{
    panic::catch_unwind(panic::AssertUnwindSafe(fun)).map_err(|any_err| {
        // Extract common types of panic payload:
        // panic and assert produce &str or String
        if let Some(&s) = any_err.downcast_ref::<&str>() {
            s.to_owned()
        } else if let Some(s) = any_err.downcast_ref::<String>() {
            s.to_owned()
        } else {
            "UNABLE TO SHOW RESULT OF PANIC.".to_owned()
        }
    })
}
