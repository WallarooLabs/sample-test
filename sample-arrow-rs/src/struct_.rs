//! Samplers for generating an arrow [`StructArray`].

use arrow_array::{Array, ArrayRef, StructArray};
use arrow_schema::DataType;
use sample_std::{Random, Sample, Shrunk};
use std::sync::Arc;

use crate::{generate_validity, ArrowSampler};

pub struct StructSampler<V> {
    pub data_type: DataType,
    pub null: Option<V>,
    pub values: Vec<ArrowSampler>,
}

impl<V> Sample for StructSampler<V>
where
    V: Sample<Output = bool>,
{
    type Output = ArrayRef;

    fn generate(&mut self, g: &mut Random) -> Self::Output {
        let arrays: Vec<ArrayRef> = self.values.iter_mut().map(|sa| sa.generate(g)).collect();
        let validity = generate_validity(&mut self.null, g, arrays[0].len());

        // Extract fields from the data_type
        let fields = if let DataType::Struct(fields) = &self.data_type {
            fields.clone()
        } else {
            panic!("Expected Struct data type")
        };

        // Create the struct array as ArrayRef
        Arc::new(StructArray::new(fields, arrays, validity))
    }

    fn shrink(&self, _v: Self::Output) -> Shrunk<Self::Output> {
        Box::new(std::iter::empty())
    }
}
