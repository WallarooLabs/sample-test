use arrow_array::{Array, ArrayRef};
use arrow_schema::{DataType, Field};
use sample_arrow_rs::{array::FromDataType, AlwaysValid, FixedLenSampler};
use sample_std::Sample;
use sample_test::sample_test;
use std::boxed::Box;
use std::sync::Arc;

pub type ArraySampler = Box<dyn Sample<Output = ArrayRef>>;

fn fixed(len: usize, count: usize) -> ArraySampler {
    let data_type = DataType::FixedSizeList(
        Arc::new(Field::new("inner", DataType::UInt8, false)),
        count as i32,
    );

    let any = FromDataType {
        validity: AlwaysValid,
        branch: 0..10,
    };

    Box::new(FixedLenSampler {
        len,
        array: any.from_data_type(&data_type),
    })
}

#[sample_test]
fn fixed_size_list_equality(#[sample(fixed(10, 30))] array: ArrayRef) {
    assert_eq!(array.len(), 10);

    // In arrow-rs, we need to create a new slice rather than modifying in-place
    if array.len() > 2 {
        let before = array.clone();
        let sliced = array.slice(0, array.len() / 2);
        assert_ne!(before.len(), sliced.len());
    }
}
