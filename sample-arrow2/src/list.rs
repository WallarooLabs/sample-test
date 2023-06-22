use std::ops::Range;

use crate::{array::ArraySampler, validity::generate_validity};
use arrow2::{
    array::{Array, ListArray},
    datatypes::DataType,
    offset::OffsetsBuffer,
};
use sample_std::{Random, Sample};

pub struct ListSampler<V> {
    pub data_type: DataType,
    pub null: Option<V>,
    pub len: Range<usize>,
    pub inner: ArraySampler,
}

impl<V> Sample for ListSampler<V>
where
    V: Sample<Output = bool> + Send + Sync + 'static,
{
    type Output = Box<dyn Array>;

    fn generate(&self, g: &mut Random) -> Self::Output {
        let values = self.inner.generate(g);
        let len = self.len.generate(g);
        let mut ix = 0;
        let mut offsets = vec![0];

        for outer_ix in 0..len {
            if outer_ix + 1 != len {
                let remaining = values.len() - ix;
                let fair = std::cmp::max(2, remaining / (len - outer_ix));
                let upper = std::cmp::min(values.len() - ix, fair);
                let count = g.gen_range(0..=upper);
                ix += count;
                offsets.push(ix as i32);
            } else {
                offsets.push(values.len() as i32);
            }
        }

        let validity = generate_validity(&self.null, g, len);

        ListArray::new(
            self.data_type.clone(),
            OffsetsBuffer::try_from(offsets).unwrap(),
            values,
            validity,
        )
        .boxed()
    }
}
