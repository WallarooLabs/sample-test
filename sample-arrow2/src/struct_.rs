//! Samplers for generating an arrow [`StructArray`].

use crate::{array::ArraySampler, generate_validity};
use arrow2::{
    array::{Array, StructArray},
    datatypes::DataType,
};
use sample_std::{Random, Sample};

pub struct StructSampler<V> {
    pub data_type: DataType,
    pub null: Option<V>,
    pub values: Vec<ArraySampler>,
}

impl<V> Sample for StructSampler<V>
where
    V: Sample<Output = bool>,
{
    type Output = Box<dyn Array>;

    fn generate(&mut self, g: &mut Random) -> Self::Output {
        let values: Vec<_> = self.values.iter_mut().map(|sa| sa.generate(g)).collect();
        let validity = generate_validity(&mut self.null, g, values[0].len());

        StructArray::new(self.data_type.clone(), values, validity).boxed()
    }
}
