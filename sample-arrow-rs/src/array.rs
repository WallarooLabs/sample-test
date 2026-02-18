//! Chained samplers for generating arbitrary `Arc<dyn Array>` arrow arrays.

use std::ops::Range;

use arrow_array::{Array, ArrayRef, FixedSizeListArray, ListArray};
use arrow_schema::DataType;
use sample_std::{Always, Chained, Sample};

use crate::{
    datatypes::DataTypeSampler,
    fixed_size_list::FixedSizeListWithLen,
    list::{ListSampler, ListWithLen},
    primitive::{
        f32_array, f32_sampler, f64_array, f64_sampler, i16_array, i16_sampler, i32_array,
        i32_sampler, i64_array, i64_sampler, i8_array, i8_sampler, u16_array, u16_sampler,
        u32_array, u32_sampler, u64_array, u64_sampler, u8_array, u8_sampler,
    },
    struct_::StructSampler,
    AlwaysValid, ArrowLenSampler, SetLen,
};

pub fn sampler_from_example(array: &dyn Array) -> ArrowLenSampler {
    match array.data_type() {
        DataType::Float32 => f32_sampler(AlwaysValid),
        DataType::Float64 => f64_sampler(AlwaysValid),
        DataType::Int8 => i8_sampler(AlwaysValid),
        DataType::Int16 => i16_sampler(AlwaysValid),
        DataType::Int32 => i32_sampler(AlwaysValid),
        DataType::Int64 => i64_sampler(AlwaysValid),
        DataType::UInt8 => u8_sampler(AlwaysValid),
        DataType::UInt16 => u16_sampler(AlwaysValid),
        DataType::UInt32 => u32_sampler(AlwaysValid),
        DataType::UInt64 => u64_sampler(AlwaysValid),
        DataType::List(_) => {
            let list = array.as_any().downcast_ref::<ListArray>().unwrap();
            // In arrow-rs we access offsets differently
            let _offsets_buffer = list.offsets();
            // Get the raw values as a slice - arrow-rs uses different methods
            // than arrow2, we'll try to access offsets directly
            let offsets = list.value_offsets();
            let lengths: Vec<_> = offsets.windows(2).map(|w| w[1] - w[0]).collect();
            let min = lengths.iter().min().copied().unwrap_or(0) as i32;
            let max = lengths.iter().max().copied().unwrap_or(0) as i32 + 1;

            // In arrow-rs, field information comes from the data_type
            let field_name = if let DataType::List(field) = array.data_type() {
                field.name().clone()
            } else {
                "item".to_string() // Fallback
            };

            Box::new(ListWithLen {
                len: array.len(),
                validity: AlwaysValid,
                count: min..max,
                inner_name: Always(field_name),
                inner: sampler_from_example(list.values()),
            })
        }
        DataType::FixedSizeList(field, size) => {
            let list = array.as_any().downcast_ref::<FixedSizeListArray>().unwrap();
            Box::new(FixedSizeListWithLen {
                len: array.len(),
                validity: AlwaysValid,
                count: Always(*size as i64),
                inner_name: Always(field.name().clone()),
                inner: sampler_from_example(list.values()),
            })
        }
        dt => panic!("not implemented: {:?}", dt),
    }
}

pub struct FromDataType<V, B> {
    pub validity: V,
    pub branch: B,
}

impl<V, B> FromDataType<V, B>
where
    V: Sample<Output = Option<crate::Bitmap>> + SetLen + Clone + Send + Sync + 'static,
    B: Sample<Output = i32> + Clone + Send + Sync + 'static,
{
    pub fn from_data_type(&self, data_type: &DataType) -> ArrowLenSampler {
        match data_type {
            DataType::Float32 => f32_sampler(self.validity.clone()),
            DataType::Float64 => f64_sampler(self.validity.clone()),
            DataType::Int8 => i8_sampler(self.validity.clone()),
            DataType::Int16 => i16_sampler(self.validity.clone()),
            DataType::Int32 => i32_sampler(self.validity.clone()),
            DataType::Int64 => i64_sampler(self.validity.clone()),
            DataType::UInt8 => u8_sampler(self.validity.clone()),
            DataType::UInt16 => u16_sampler(self.validity.clone()),
            DataType::UInt32 => u32_sampler(self.validity.clone()),
            DataType::UInt64 => u64_sampler(self.validity.clone()),
            DataType::List(field) => Box::new(ListWithLen {
                len: 0,
                validity: self.validity.clone(),
                count: self.branch.clone(),
                inner_name: Always(field.name().clone()),
                inner: self.from_data_type(field.data_type()),
            }),
            DataType::FixedSizeList(field, size) => Box::new(FixedSizeListWithLen {
                len: 0,
                validity: self.validity.clone(),
                count: Always(*size as i64),
                inner_name: Always(field.name().clone()),
                inner: self.from_data_type(field.data_type()),
            }),
            dt => panic!("not implemented: {:?}", dt),
        }
    }
}

pub type ArraySampler = Box<dyn Sample<Output = ArrayRef> + Send + Sync>;

pub type ChainedArraySampler = Box<dyn Sample<Output = Chained<DataType, ArrayRef>> + Send + Sync>;

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
            DataType::Float32 => f32_array(len.clone(), current_null),
            DataType::Float64 => f64_array(len.clone(), current_null),
            DataType::Int8 => i8_array(len.clone(), current_null),
            DataType::Int16 => i16_array(len.clone(), current_null),
            DataType::Int32 => i32_array(len.clone(), current_null),
            DataType::Int64 => i64_array(len.clone(), current_null),
            DataType::UInt8 => u8_array(len.clone(), current_null),
            DataType::UInt16 => u16_array(len.clone(), current_null),
            DataType::UInt32 => u32_array(len.clone(), current_null),
            DataType::UInt64 => u64_array(len.clone(), current_null),
            DataType::Struct(fields) => Box::new(StructSampler {
                data_type: data_type.clone(),
                null: current_null,
                values: fields
                    .iter()
                    .map(|f| {
                        ArbitraryArray {
                            len: (len.end.saturating_sub(1))..len.end,
                            is_nullable: f.is_nullable(),
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
                    is_nullable: field.is_nullable(),
                    ..self.clone()
                }
                .sampler_from_data_type(field.data_type()),
            }),
            dt => panic!("not implemented: {:?}", dt),
        }
    }
}
