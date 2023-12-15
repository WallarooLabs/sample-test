//! Samplers for generating an arrow [`ListArray`].

use std::ops::Range;

use crate::{array::ArraySampler, generate_validity, SampleLen, SetLen};
use arrow2::{
    array::{Array, ListArray},
    bitmap::Bitmap,
    datatypes::{DataType, Field},
    offset::OffsetsBuffer,
};
use sample_std::{Random, Sample};

pub struct ListWithLen<V, C, A, N> {
    pub len: usize,
    pub validity: V,
    pub count: C,

    pub inner: A,
    pub inner_name: N,
}

impl<V: SetLen, C, A, N> SetLen for ListWithLen<V, C, A, N> {
    fn set_len(&mut self, len: usize) {
        self.len = len;
        self.validity.set_len(len);
    }
}

impl<V, C, A, N> Sample for ListWithLen<V, C, A, N>
where
    V: Sample<Output = Option<Bitmap>> + SetLen,
    C: Sample<Output = i32>,
    A: Sample<Output = Box<dyn Array>> + SetLen,
    N: Sample<Output = String>,
{
    type Output = Box<dyn Array>;

    fn generate(&mut self, g: &mut Random) -> Self::Output {
        let mut offsets = vec![0];
        let mut inner_len: i32 = 0;
        for _ in 0..self.len {
            let count = self.count.generate(g);
            assert!(count >= 0);
            inner_len += count;
            offsets.push(inner_len);
        }

        self.inner.set_len(inner_len as usize);
        let values = self.inner.generate(g);
        let is_nullable = values.validity().is_some();
        let inner_name = self.inner_name.generate(g);
        let field = Field::new(inner_name, values.data_type().clone(), is_nullable);
        let data_type = DataType::List(Box::new(field));

        // SAFETY: see loop above. starts at zero, asserts all increments are positive.
        let offsets = OffsetsBuffer::try_from(offsets).unwrap();
        let validity = self.validity.generate(g);
        ListArray::new(data_type, offsets, values, validity).boxed()
    }

    fn shrink(&self, _: Self::Output) -> Box<dyn Iterator<Item = Self::Output>> {
        Box::new(std::iter::empty())
    }
}

impl<V, C, A, N> SampleLen for ListWithLen<V, C, A, N>
where
    V: Sample<Output = Option<Bitmap>> + SetLen,
    C: Sample<Output = i32>,
    A: Sample<Output = Box<dyn Array>> + SetLen,
    N: Sample<Output = String>,
{
}

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

    fn generate(&mut self, g: &mut Random) -> Self::Output {
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

        let validity = generate_validity(&mut self.null, g, len);

        ListArray::new(
            self.data_type.clone(),
            OffsetsBuffer::try_from(offsets).unwrap(),
            values,
            validity,
        )
        .boxed()
    }
}
