use arrow2::{
    array::Array,
    datatypes::{DataType, Field},
};
use sample_arrow2::{array::FromDataType, AlwaysValid, FixedLenSampler};
use sample_std::Sample;
use sample_test::sample_test;
use std::boxed::Box;

pub type ArraySampler = Box<dyn Sample<Output = Box<dyn Array>>>;

fn fixed(len: usize, count: usize) -> ArraySampler {
    let data_type = DataType::FixedSizeList(
        Box::new(Field::new("inner".to_string(), DataType::UInt8, false)),
        count,
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
fn fixed_size_list_equality(#[sample(fixed(10, 30))] mut array: Box<dyn Array>) {
    assert_eq!(array.len(), 10);
    assert_eq!(array, array);
    if array.len() > 2 {
        let before = array.clone();
        array.slice(0, array.len() / 2);
        assert_ne!(before, array);
    }
}
