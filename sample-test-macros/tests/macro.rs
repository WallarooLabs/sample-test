use sample_test::TestResult;
use sample_test::{lazy_static, sample_test};
use std::ops::Range;

lazy_static! {
    static ref RANGE: Range<usize> = 10..20;
}

#[sample_test]
fn min(#[sample(RANGE)] x: usize, #[sample(RANGE)] y: usize) -> TestResult {
    if x < y {
        TestResult::discard()
    } else {
        TestResult::from_bool(::std::cmp::min(x, y) == y)
    }
}
