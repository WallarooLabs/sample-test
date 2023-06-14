use std::ops::Range;

use arrow2::{array::Array, datatypes::DataType};
use sample_std::{valid_f32, valid_f64, Chained, Sample};

use crate::{
    datatypes::DataTypeSampler,
    list::ListSampler,
    primitive::{arbitrary_boxed_primitive, boxed_primitive},
    struct_::StructSampler,
};

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
                            len: (len.end - 1)..len.end,
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
