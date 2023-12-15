//! Samplers for generating an arrow [`FixedSizeListArray`].

use crate::{SampleLen, SetLen};
use arrow2::{
    array::{Array, FixedSizeListArray},
    bitmap::Bitmap,
    datatypes::{DataType, Field},
};
use sample_std::{Random, Sample};

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
    V: Sample<Output = Option<Bitmap>> + SetLen,
    C: Sample<Output = usize>,
    A: Sample<Output = Box<dyn Array>> + SetLen,
    N: Sample<Output = String>,
{
    type Output = Box<dyn Array>;

    fn generate(&mut self, g: &mut Random) -> Self::Output {
        let count = self.count.generate(g);
        self.inner.set_len(count * self.len);
        let values = self.inner.generate(g);
        let is_nullable = values.validity().is_some();
        let inner_name = self.inner_name.generate(g);
        let field = Field::new(inner_name, values.data_type().clone(), is_nullable);
        let data_type = DataType::FixedSizeList(Box::new(field), count);
        let validity = self.validity.generate(g);
        FixedSizeListArray::new(data_type, values, validity).boxed()
    }

    fn shrink(&self, _: Self::Output) -> Box<dyn Iterator<Item = Self::Output>> {
        Box::new(std::iter::empty())
    }
}

impl<V, C, A, N> SampleLen for FixedSizeListWithLen<V, C, A, N>
where
    V: Sample<Output = Option<Bitmap>> + SetLen,
    C: Sample<Output = usize>,
    A: Sample<Output = Box<dyn Array>> + SetLen,
    N: Sample<Output = String>,
{
}
