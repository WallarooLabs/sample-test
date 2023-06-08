use std::ops::Range;

use arrow2::{array::Array, datatypes::DataType};
use sample_std::{valid_f32, valid_f64, Random, Sample};

use crate::{
    datatypes::sample_data_type,
    list::ListSampler,
    primitive::{arbitrary_boxed_primitive, boxed_primitive},
    struct_::StructSampler,
};

pub type ArraySampler = Box<dyn Sample<Output = Box<dyn Array>> + Send + Sync>;

#[derive(Clone, Debug)]
pub struct ArbitrarySampler<N, AN, V> {
    pub data_type_depth: usize,
    pub names: N,
    pub nullable: AN,

    pub branch: Range<usize>,
    pub len: Range<usize>,
    pub null: V,
    pub is_nullable: bool,
}

impl<N, AN, V> Sample for ArbitrarySampler<N, AN, V>
where
    N: Sample<Output = String> + Send + Sync + Clone + 'static,
    AN: Sample<Output = bool> + Send + Sync + Clone + 'static,
    V: Sample<Output = bool> + Send + Sync + Clone + 'static,
{
    type Output = Box<dyn Array>;

    fn generate(&self, g: &mut Random) -> Self::Output {
        let dt = sample_data_type(
            self.data_type_depth,
            self.names.clone(),
            self.nullable.clone(),
        )
        .generate(g);

        sampler_from_data_type(
            &dt,
            self.branch.clone(),
            self.len.clone(),
            self.null.clone(),
            self.is_nullable,
        )
        .generate(g)
    }
}

pub fn sampler_from_data_type<V>(
    data_type: &DataType,
    branch: Range<usize>,
    len: Range<usize>,
    null: V,
    is_nullable: bool,
) -> ArraySampler
where
    V: Sample<Output = bool> + Send + Sync + Clone + 'static,
{
    let current_null = if is_nullable {
        Some(null.clone())
    } else {
        None
    };

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
                    sampler_from_data_type(
                        f.data_type(),
                        branch.clone(),
                        (len.end - 1)..len.end,
                        null.clone(),
                        f.is_nullable,
                    )
                })
                .collect(),
        }),
        DataType::List(field) => Box::new(ListSampler {
            data_type: data_type.clone(),
            len: len.clone(),
            null: current_null,
            inner: sampler_from_data_type(
                field.data_type(),
                branch.clone(),
                (branch.start * len.start)..(branch.end * len.end),
                null.clone(),
                field.is_nullable,
            ),
        }),
        dt => panic!("not implemented: {:?}", dt),
    }
}
