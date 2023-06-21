use arrow2::{array::Array, datatypes::DataType};
use sample_arrow2::{
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
fn list_equality(#[sample(deep_array(3))] list: Chained<DataType, Box<dyn Array>>) {
    let mut list = list.value.clone();
    assert_eq!(list.len(), 10);
    assert_eq!(list, list);
    if list.len() > 2 {
        let before = list.clone();
        list.slice(0, list.len() / 2);
        assert_ne!(before, list);
    }
}
