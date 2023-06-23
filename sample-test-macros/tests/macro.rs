use lazy_static::lazy_static;
use once_cell::sync::Lazy;
use sample_test::sample_test;
use sample_test::TestResult;
use std::ops::Range;

lazy_static! {
    static ref RANGE: Range<usize> = 10..20;
}

static RANGE2: Lazy<Range<usize>> = Lazy::new(|| 10..20);

fn range() -> Range<usize> {
    10..20
}

#[sample_test]
fn min(#[sample(10..20)] x: usize, #[sample(range())] y: usize) -> TestResult {
    if x < y {
        TestResult::discard()
    } else {
        TestResult::from_bool(::std::cmp::min(x, y) == y)
    }
}

#[sample_test]
fn min2(#[sample(RANGE.clone())] x: usize, #[sample(RANGE2.clone())] y: usize) -> TestResult {
    if x < y {
        TestResult::discard()
    } else {
        TestResult::from_bool(::std::cmp::min(x, y) == y)
    }
}
