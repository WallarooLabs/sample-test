use arrow_array::{Array, ArrayRef};
use arrow_schema::DataType;
use sample_arrow_rs::{
    array::{ArbitraryArray, ChainedArraySampler},
    datatypes::{sample_flat, ArbitraryDataType},
};
use sample_std::{Chained, Chance, Regex};
use sample_test::sample_test;
use std::boxed::Box;

fn deep_array(depth: usize) -> ChainedArraySampler {
    let names = Regex::new("[a-z]{4,8}");
    let dt = ArbitraryDataType {
        struct_branch: 1..3,
        names: names.clone(),
        nullable: Chance(0.5),
        flat: sample_flat,
    }
    .sample_depth(depth);

    Box::new(
        ArbitraryArray {
            names,
            branch: 0..10,
            len: 10..11,
            null: Chance(0.1),
            is_nullable: true,
        }
        .arbitrary_array(dt),
    )
}

#[sample_test]
fn list_equality(#[sample(deep_array(3))] list: Chained<DataType, ArrayRef>) {
    let list = &list.value;
    assert_eq!(list.len(), 10);

    // In arrow-rs, we need to create a new slice rather than modifying in-place
    if list.len() > 2 {
        let before = list.clone();
        let sliced = list.slice(0, list.len() / 2);
        assert_ne!(before.len(), sliced.len());
    }
}
