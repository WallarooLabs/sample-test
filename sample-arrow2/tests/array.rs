use arrow2::array::Array;
use sample_arrow2::{
    array::{ArbitrarySampler, ArraySampler},
    ArrowSampler,
};
use sample_std::{Chance, Regex};
use sample_test::{lazy_static, sample_test};
use std::boxed::Box;

fn deep_array(depth: usize) -> ArraySampler {
    Box::new(ArbitrarySampler {
        data_type_depth: depth,
        names: Regex::new("[a-z]{4,8}"),
        nullable: Chance(0.5),

        branch: 0..10,
        len: 10..11,
        null: Chance(0.1),
        is_nullable: true,
    })
}

lazy_static! {
    static ref DEEP_LIST: ArrowSampler = deep_array(3);
}

#[sample_test]
fn list_equality(#[sample(DEEP_LIST)] list: Box<dyn Array>) {
    let mut list = list.clone();
    assert_eq!(list.len(), 10);
    assert_eq!(list, list);
    if list.len() > 2 {
        let before = list.clone();
        list.slice(0, list.len() / 2);
        assert_ne!(before, list);
    }
}
