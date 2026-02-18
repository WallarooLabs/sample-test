//! Samplers for generating an arrow [`ListArray`].

use std::ops::Range;
use std::sync::Arc;

use arrow_array::{Array, ArrayRef, ListArray};
use arrow_buffer::OffsetBuffer;
use arrow_schema::{DataType, Field};
use sample_std::{Random, Sample, Shrunk};

use crate::{generate_validity, ArrowSampler, Bitmap, SampleLen, SetLen};

pub struct ListSampler<V> {
    pub data_type: DataType,
    pub len: Range<usize>,
    pub null: Option<V>,
    pub inner: ArrowSampler,
}

impl<V> Sample for ListSampler<V>
where
    V: Sample<Output = bool> + Send + Sync + 'static,
{
    type Output = ArrayRef;

    fn generate(&mut self, g: &mut Random) -> Self::Output {
        let values = self.inner.generate(g);
        let len = g.gen_range(self.len.clone());
        let mut ix = 0;
        let mut offsets = vec![0i32];

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
        // Convert our custom Bitmap type to arrow's NullBuffer
        let null_buffer = validity.map(|bitmap| bitmap);

        // Convert offsets to a ScalarBuffer first, then to an OffsetBuffer
        let scalar_buffer = arrow_buffer::ScalarBuffer::from(offsets);
        let offsets_buffer = OffsetBuffer::new(scalar_buffer);

        let field = if let DataType::List(field) = &self.data_type {
            field.clone()
        } else {
            panic!("Expected List data type")
        };

        // Create the ListArray and return as Arc<dyn Array>
        Arc::new(ListArray::new(field, offsets_buffer, values, null_buffer))
    }

    fn shrink(&self, _v: Self::Output) -> Shrunk<Self::Output> {
        Box::new(std::iter::empty())
    }
}

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
    A: Sample<Output = ArrayRef> + SetLen,
    N: Sample<Output = String>,
{
    type Output = ArrayRef;

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
        let is_nullable = values.nulls().is_some();
        let inner_name = self.inner_name.generate(g);
        let field = Arc::new(Field::new(
            inner_name,
            values.data_type().clone(),
            is_nullable,
        ));

        // Convert offsets to a ScalarBuffer first, then to an OffsetBuffer
        let scalar_buffer = arrow_buffer::ScalarBuffer::from(offsets);
        let offsets_buffer = OffsetBuffer::new(scalar_buffer);

        // Convert the validity bitmap to a NullBuffer if present
        let null_buffer = self.validity.generate(g).map(|bitmap| bitmap);

        // Create the ListArray
        Arc::new(ListArray::new(field, offsets_buffer, values, null_buffer))
    }

    fn shrink(&self, _: Self::Output) -> Shrunk<Self::Output> {
        Box::new(std::iter::empty())
    }
}

impl<V, C, A, N> SampleLen for ListWithLen<V, C, A, N>
where
    V: Sample<Output = Option<Bitmap>> + SetLen,
    C: Sample<Output = i32>,
    A: Sample<Output = ArrayRef> + SetLen,
    N: Sample<Output = String>,
{
}
