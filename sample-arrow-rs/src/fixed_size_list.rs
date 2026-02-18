//! Samplers for generating an arrow [`FixedSizeListArray`].

use std::sync::Arc;

use arrow_array::{Array, ArrayRef, FixedSizeListArray};
use arrow_schema::Field;
use sample_std::{Random, Sample, Shrunk};

use crate::{SampleLen, SetLen};

pub struct FixedSizeListWithLen<V, C, A, N> {
    pub len: usize,
    pub validity: V,
    pub count: C,
    pub inner: A,
    pub inner_name: N,
}

impl<V: SetLen, C, A, N> SetLen for FixedSizeListWithLen<V, C, A, N> {
    fn set_len(&mut self, len: usize) {
        self.len = len;
        self.validity.set_len(len);
    }
}

impl<V, C, A, N> Sample for FixedSizeListWithLen<V, C, A, N>
where
    V: Sample<Output = Option<crate::Bitmap>> + SetLen,
    C: Sample<Output = i64>, // Using i64 for size in arrow-rs
    A: Sample<Output = ArrayRef> + SetLen,
    N: Sample<Output = String>,
{
    type Output = ArrayRef;

    fn generate(&mut self, g: &mut Random) -> Self::Output {
        let count = self.count.generate(g) as i32; // Convert to i32 for arrow-rs
        self.inner.set_len(count as usize * self.len);
        let values = self.inner.generate(g);
        let is_nullable = values.nulls().is_some();
        let inner_name = self.inner_name.generate(g);
        let field = Arc::new(Field::new(
            inner_name,
            values.data_type().clone(),
            is_nullable,
        ));

        // Convert the validity bitmap to a NullBuffer if present
        let null_buffer = self.validity.generate(g).map(|bitmap| bitmap);

        // Create a FixedSizeListArray
        Arc::new(FixedSizeListArray::new(field, count, values, null_buffer))
    }

    fn shrink(&self, _: Self::Output) -> Shrunk<Self::Output> {
        Box::new(std::iter::empty())
    }
}

impl<V, C, A, N> SampleLen for FixedSizeListWithLen<V, C, A, N>
where
    V: Sample<Output = Option<crate::Bitmap>> + SetLen,
    C: Sample<Output = i64>,
    A: Sample<Output = ArrayRef> + SetLen,
    N: Sample<Output = String>,
{
}
