use sample_std::VecSampler;
use sample_test::{sample_test, TestResult};

#[sample_test]
fn age_range(#[sample(VecSampler { el: 1..25, length: 0..50 })] ages: Vec<u8>) -> TestResult {
    if ages.iter().all(|a| *a < 5) {
        TestResult::discard()
    } else {
        TestResult::from_bool(ages.iter().all(|a| *a < 25))
    }
}
