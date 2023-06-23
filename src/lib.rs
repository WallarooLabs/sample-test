pub use sample_test_macros::sample_test;

pub use quickcheck::{Arbitrary, Gen};

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
