//! Chained samplers for generating arbitrary `Box<dyn Array>` arrow arrays.

use std::ops::Range;

use arrow2::{
    array::{Array, FixedSizeListArray, ListArray},
    bitmap::Bitmap,
    datatypes::DataType,
};
use sample_std::{valid_f32, valid_f64, Always, Chained, Sample};

use crate::{
    datatypes::DataTypeSampler,
    fixed_size_list::FixedSizeListWithLen,
    list::{ListSampler, ListWithLen},
    primitive::{
        arbitrary_boxed_primitive, arbitrary_len_sampler, boxed_primitive, primitive_len_sampler,
    },
    struct_::StructSampler,
    AlwaysValid, ArrowLenSampler, SetLen,
};

pub fn sampler_from_example(array: &dyn Array) -> ArrowLenSampler {
    match array.data_type() {
        DataType::Float32 => primitive_len_sampler(valid_f32(), AlwaysValid),
        DataType::Float64 => primitive_len_sampler(valid_f64(), AlwaysValid),
        DataType::Int8 => arbitrary_len_sampler::<i8, _>(AlwaysValid),
        DataType::Int16 => arbitrary_len_sampler::<i16, _>(AlwaysValid),
        DataType::Int32 => arbitrary_len_sampler::<i32, _>(AlwaysValid),
        DataType::Int64 => arbitrary_len_sampler::<i64, _>(AlwaysValid),
        DataType::UInt8 => arbitrary_len_sampler::<u8, _>(AlwaysValid),
        DataType::UInt16 => arbitrary_len_sampler::<u16, _>(AlwaysValid),
        DataType::UInt32 => arbitrary_len_sampler::<u32, _>(AlwaysValid),
        DataType::UInt64 => arbitrary_len_sampler::<u64, _>(AlwaysValid),
        DataType::List(f) => {
            let list = array.as_any().downcast_ref::<ListArray<i32>>().unwrap();
            let min = list.offsets().lengths().min().unwrap_or(0) as i32;
            let max = list.offsets().lengths().max().unwrap_or(0) as i32 + 1;
            Box::new(ListWithLen {
                len: array.len(),
                validity: AlwaysValid,
                count: min..max,
                inner_name: Always(f.name.clone()),
                inner: sampler_from_example(list.values().as_ref()),
            })
        }
        DataType::FixedSizeList(f, count) => Box::new(FixedSizeListWithLen {
            len: array.len(),
            validity: AlwaysValid,
            count: Always(*count),
            inner_name: Always(f.name.clone()),
            inner: sampler_from_example(
                array
                    .as_any()
                    .downcast_ref::<FixedSizeListArray>()
                    .unwrap()
                    .values()
                    .as_ref(),
            ),
        }),
        dt => panic!("not implemented: {:?}", dt),
    }
}

pub struct FromDataType<V, B> {
    pub validity: V,

    pub branch: B,
}

impl<V, B> FromDataType<V, B>
where
    V: Sample<Output = Option<Bitmap>> + SetLen + Clone + Send + Sync + 'static,
    B: Sample<Output = i32> + Clone + Send + Sync + 'static,
{
    pub fn from_data_type(&self, data_type: &DataType) -> ArrowLenSampler {
        match data_type {
            DataType::Float32 => primitive_len_sampler(valid_f32(), self.validity.clone()),
            DataType::Float64 => primitive_len_sampler(valid_f64(), self.validity.clone()),
            DataType::Int8 => arbitrary_len_sampler::<i8, _>(self.validity.clone()),
            DataType::Int16 => arbitrary_len_sampler::<i16, _>(self.validity.clone()),
            DataType::Int32 => arbitrary_len_sampler::<i32, _>(self.validity.clone()),
            DataType::Int64 => arbitrary_len_sampler::<i64, _>(self.validity.clone()),
            DataType::UInt8 => arbitrary_len_sampler::<u8, _>(self.validity.clone()),
            DataType::UInt16 => arbitrary_len_sampler::<u16, _>(self.validity.clone()),
            DataType::UInt32 => arbitrary_len_sampler::<u32, _>(self.validity.clone()),
            DataType::UInt64 => arbitrary_len_sampler::<u64, _>(self.validity.clone()),
            DataType::List(f) => Box::new(ListWithLen {
                len: 0,
                validity: self.validity.clone(),
                count: self.branch.clone(),
                inner_name: Always(f.name.clone()),
                inner: self.from_data_type(f.data_type()),
            }),
            DataType::FixedSizeList(f, count) => Box::new(FixedSizeListWithLen {
                len: 0,
                validity: self.validity.clone(),
                count: Always(*count),
                inner_name: Always(f.name.clone()),
                inner: self.from_data_type(f.data_type()),
            }),
            dt => panic!("not implemented: {:?}", dt),
        }
    }
}

pub type ArraySampler = Box<dyn Sample<Output = Box<dyn Array>> + Send + Sync>;

pub type ChainedArraySampler =
    Box<dyn Sample<Output = Chained<DataType, Box<dyn Array>>> + Send + Sync>;

#[derive(Clone, Debug)]
pub struct ArbitraryArray<N, V> {
    pub names: N,
    pub branch: Range<usize>,
    pub len: Range<usize>,
    pub null: V,
    pub is_nullable: bool,
}

impl<N, V> ArbitraryArray<N, V>
where
    N: Sample<Output = String> + Send + Sync + Clone + 'static,
    V: Sample<Output = bool> + Send + Sync + Clone + 'static,
{
    pub fn with_len(&self, len: usize) -> Self {
        Self {
            len: len..(len + 1),
            ..self.clone()
        }
    }

    pub fn arbitrary_array(self, data_type_sampler: DataTypeSampler) -> ChainedArraySampler {
        Box::new(data_type_sampler.chain_resample(
            move |data_type| self.sampler_from_data_type(&data_type),
            100,
        ))
    }

    pub fn sampler_from_data_type(&self, data_type: &DataType) -> ArraySampler {
        let current_null = if self.is_nullable {
            Some(self.null.clone())
        } else {
            None
        };
        let len = self.len.clone();

        match data_type {
            DataType::Float32 => boxed_primitive(valid_f32(), len, current_null),
            DataType::Float64 => boxed_primitive(valid_f64(), len, current_null),
            DataType::Int8 => arbitrary_boxed_primitive::<i8, _>(len, current_null),
            DataType::Int16 => arbitrary_boxed_primitive::<i16, _>(len, current_null),
            DataType::Int32 => arbitrary_boxed_primitive::<i32, _>(len, current_null),
            DataType::Int64 => arbitrary_boxed_primitive::<i64, _>(len, current_null),
            DataType::UInt8 => arbitrary_boxed_primitive::<u8, _>(len, current_null),
            DataType::UInt16 => arbitrary_boxed_primitive::<u16, _>(len, current_null),
            DataType::UInt32 => arbitrary_boxed_primitive::<u32, _>(len, current_null),
            DataType::UInt64 => arbitrary_boxed_primitive::<u64, _>(len, current_null),
            DataType::Struct(fields) => Box::new(StructSampler {
                data_type: data_type.clone(),
                null: current_null,
                values: fields
                    .iter()
                    .map(|f| {
                        ArbitraryArray {
                            len: (len.end.saturating_sub(1))..len.end,
                            is_nullable: f.is_nullable,
                            ..self.clone()
                        }
                        .sampler_from_data_type(f.data_type())
                    })
                    .collect(),
            }),
            DataType::List(field) => Box::new(ListSampler {
                data_type: data_type.clone(),
                len: len.clone(),
                null: current_null,
                inner: ArbitraryArray {
                    branch: (self.branch.start * self.len.start)..(self.branch.end * self.len.end),
                    is_nullable: field.is_nullable,
                    ..self.clone()
                }
                .sampler_from_data_type(field.data_type()),
            }),
            dt => panic!("not implemented: {:?}", dt),
        }
    }
}
