use sample_std::Random;
use sample_test::{
    env_logger_init,
    tester::{sample_test, TestResult, Testable},
};

#[test]
fn test_testable() {
    fn test(a: usize, b: usize) -> bool {
        let sum = a + b;
        sum >= a && sum >= b
    }

    let mut r = Random::new(1000);
    let s = (0..10, 0..10);
    assert!(Testable::test_once(&(test as fn(usize, usize) -> bool), &s, &mut r).is_success());

    sample_test(s, test as fn(usize, usize) -> bool);
}

#[test]
fn test_shrink() {
    let _ = crate::env_logger_init();

    fn test(a: usize, b: usize) -> bool {
        a <= 5 && b <= 5
    }

    let s = (0..10, 0..10);
    assert_eq!(
        Testable::shrink(
            &(test as fn(usize, usize) -> bool),
            &s,
            TestResult::passed(),
            (9, 9)
        )
        .arguments(),
        // ranges attempt to shrink to their start first
        // (0, 9) fails
        // (0, 0) passes
        // we then reverse-recurse down to the "smallest" failing value, (0, 6)
        "(0, 6)"
    );
}
