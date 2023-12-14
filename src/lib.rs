//! Testing utilities for use with samplers defined via [`sample_std`].
//!
//! The easiest way to use these facilities is with the exported [`sample_test`]
//! macro:
//!
//! ```
//! use sample_test::{sample_test, Sample};
//!
//! #[sample_test]
//! fn test_order(#[sample(0..10)] a: usize, #[sample(20..30)] b: usize) -> bool {
//!     a < b
//! }
//! ```
//!
//! You may also use [`tester::SampleTest`] or [`tester::sample_test`] directly:
//!
//! ```
//! use sample_test::{Random, tester::sample_test};
//! fn test(a: usize, b: usize) -> bool {
//!     let sum = a + b;
//!     sum >= a && sum >= b
//! }
//!
//! let mut r = Random::new();
//! let mut s = (0..10, 0..10);
//!
//! sample_test(s, test as fn(usize, usize) -> bool);
//! ```
pub use sample_test_macros::sample_test;

pub use quickcheck::{Arbitrary, Gen};
pub use sample_std::{Random, Sample};

pub mod tester;

pub use tester::{SampleTest, TestResult, Testable};

#[cfg(feature = "use_logging")]
pub fn env_logger_init() -> Result<(), log::SetLoggerError> {
    env_logger::try_init()
}
#[cfg(feature = "use_logging")]
macro_rules! error {
    ($($tt:tt)*) => {
        log::error!($($tt)*)
    };
}
#[cfg(feature = "use_logging")]
macro_rules! info {
    ($($tt:tt)*) => {
        log::info!($($tt)*)
    };
}
#[cfg(feature = "use_logging")]
macro_rules! trace {
    ($($tt:tt)*) => {
        log::trace!($($tt)*)
    };
}

#[cfg(not(feature = "use_logging"))]
pub fn env_logger_init() {}
#[cfg(not(feature = "use_logging"))]
macro_rules! error {
    ($($_ignore:tt)*) => {
        ()
    };
}
#[cfg(not(feature = "use_logging"))]
macro_rules! info {
    ($($_ignore:tt)*) => {
        ()
    };
}
#[cfg(not(feature = "use_logging"))]
macro_rules! trace {
    ($($_ignore:tt)*) => {
        ()
    };
}

pub(crate) use error;
pub(crate) use info;
pub(crate) use trace;
