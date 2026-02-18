//! Samplers for generating an arrow [`PrimitiveArray`].

use std::marker::PhantomData;
use std::ops::Range;
use std::sync::Arc;

use arrow_array::{ArrayRef, ArrowPrimitiveType, PrimitiveArray};
use arrow_buffer::ScalarBuffer;
use sample_std::{
    arbitrary, sampler_choice, valid_f32, valid_f64, Random, Sample, Shrunk, VecSampler,
};

use crate::{ArrowLenSampler, ArrowSampler, Bitmap, SampleLen, SetLen};

#[derive(Debug, Clone)]
pub struct PrimitiveArraySampler<PT, V, T> {
    len: usize,
    inner: PT,
    validity: V,
    _phantom: PhantomData<T>,
}

impl<PT, V, T> SetLen for PrimitiveArraySampler<PT, V, T>
where
    V: SetLen,
{
    fn set_len(&mut self, len: usize) {
        self.len = len;
        self.validity.set_len(len);
    }
}

impl<PT, V, T> SampleLen for PrimitiveArraySampler<PT, V, T>
where
    PT: Sample + 'static,
    T: ArrowPrimitiveType + 'static,
    T::Native: From<PT::Output>,
    V: Sample<Output = Option<Bitmap>> + SetLen + 'static,
{
}

impl<PT, V, T> Sample for PrimitiveArraySampler<PT, V, T>
where
    PT: Sample,
    T: ArrowPrimitiveType,
    T::Native: From<PT::Output>,
    V: Sample<Output = Option<Bitmap>> + SetLen,
{
    type Output = ArrayRef;

    fn generate(&mut self, g: &mut Random) -> Self::Output {
        // Generate values and convert to T::Native
        let values: Vec<T::Native> = (0..self.len)
            .map(|_| T::Native::from(self.inner.generate(g)))
            .collect();

        // Convert to ScalarBuffer
        let values_buffer = ScalarBuffer::from(values);

        // Generate validity and convert to NullBuffer if present
        let null_buffer = self.validity.generate(g).map(|bitmap| bitmap);

        // Create the primitive array and return as Arc<dyn Array>
        Arc::new(PrimitiveArray::<T>::new(values_buffer, null_buffer))
    }

    fn shrink(&self, _: Self::Output) -> Shrunk<Self::Output> {
        Box::new(std::iter::empty())
    }
}

pub fn primitive_len_sampler<PT, V, T>(inner: PT, validity: V) -> ArrowLenSampler
where
    PT: Sample + 'static,
    T: ArrowPrimitiveType + 'static,
    T::Native: From<PT::Output>,
    V: Sample<Output = Option<Bitmap>> + SetLen + 'static,
{
    Box::new(PrimitiveArraySampler::<PT, V, T> {
        len: 0,
        inner,
        validity,
        _phantom: PhantomData,
    })
}

// Helper to implement samplers for int types (i8, i16, i32, etc.)
macro_rules! primitive_samplers {
    ($type:ty, $arrow_type:ty, $fn_name:ident) => {
        pub fn $fn_name(
            validity: impl Sample<Output = Option<Bitmap>> + SetLen + Clone + 'static,
        ) -> ArrowLenSampler {
            primitive_len_sampler::<_, _, $arrow_type>(arbitrary::<$type>(), validity)
        }
    };
}

// Implement primitive samplers for different arrow types
primitive_samplers!(i8, arrow_array::types::Int8Type, i8_sampler);
primitive_samplers!(i16, arrow_array::types::Int16Type, i16_sampler);
primitive_samplers!(i32, arrow_array::types::Int32Type, i32_sampler);
primitive_samplers!(i64, arrow_array::types::Int64Type, i64_sampler);
primitive_samplers!(u8, arrow_array::types::UInt8Type, u8_sampler);
primitive_samplers!(u16, arrow_array::types::UInt16Type, u16_sampler);
primitive_samplers!(u32, arrow_array::types::UInt32Type, u32_sampler);
primitive_samplers!(u64, arrow_array::types::UInt64Type, u64_sampler);
primitive_samplers!(f32, arrow_array::types::Float32Type, f32_sampler);
primitive_samplers!(f64, arrow_array::types::Float64Type, f64_sampler);

pub fn valid_float_len_sampler<V>(valid: V) -> ArrowLenSampler
where
    V: Sample<Output = Option<Bitmap>> + SetLen + Clone + 'static,
{
    Box::new(sampler_choice([
        primitive_len_sampler::<_, _, arrow_array::types::Float32Type>(valid_f32(), valid.clone()),
        primitive_len_sampler::<_, _, arrow_array::types::Float64Type>(valid_f64(), valid),
    ]))
}

pub fn arbitrary_int_len_sampler<V>(valid: V) -> ArrowLenSampler
where
    V: Sample<Output = Option<Bitmap>> + SetLen + Clone + 'static,
{
    Box::new(sampler_choice([
        i8_sampler(valid.clone()),
        i16_sampler(valid.clone()),
        i32_sampler(valid.clone()),
        i64_sampler(valid.clone()),
    ]))
}

pub fn arbitrary_uint_len_sampler<V>(valid: V) -> ArrowLenSampler
where
    V: Sample<Output = Option<Bitmap>> + SetLen + Clone + 'static,
{
    Box::new(sampler_choice([
        u8_sampler(valid.clone()),
        u16_sampler(valid.clone()),
        u32_sampler(valid.clone()),
        u64_sampler(valid.clone()),
    ]))
}

pub fn valid_primitive_len<V>(valid: V) -> ArrowLenSampler
where
    V: Sample<Output = Option<Bitmap>> + SetLen + Clone + 'static,
{
    Box::new(sampler_choice([
        valid_float_len_sampler(valid.clone()),
        arbitrary_int_len_sampler(valid.clone()),
        arbitrary_uint_len_sampler(valid.clone()),
    ]))
}

pub fn arbitrary_primitive_len<V>(valid: V) -> ArrowLenSampler
where
    V: Sample<Output = Option<Bitmap>> + SetLen + Clone + 'static,
{
    valid_primitive_len(valid)
}

#[derive(Debug, Clone)]
pub struct ProtoNullablePrimitiveArray<PT, T> {
    inner: VecSampler<Range<usize>, PT>,
    _phantom: PhantomData<T>,
}

impl<PT, N, T> Sample for ProtoNullablePrimitiveArray<PT, T>
where
    PT: Sample<Output = Option<N>> + Clone + 'static,
    N: Clone + 'static,
    T: ArrowPrimitiveType + 'static,
    T::Native: From<N>,
{
    type Output = ArrayRef;

    fn generate(&mut self, g: &mut Random) -> Self::Output {
        // Generate optional values
        let values: Vec<Option<T::Native>> = self
            .inner
            .generate(g)
            .into_iter()
            .map(|opt| opt.map(T::Native::from))
            .collect();

        // Use from_iter to create a PrimitiveArray with nulls
        Arc::new(PrimitiveArray::<T>::from_iter(values))
    }

    fn shrink(&self, _: Self::Output) -> Shrunk<Self::Output> {
        Box::new(std::iter::empty())
    }
}

#[derive(Debug, Clone)]
pub struct ProtoPrimitiveArray<PT, T> {
    inner: VecSampler<Range<usize>, PT>,
    _phantom: PhantomData<T>,
}

impl<PT, N, T> Sample for ProtoPrimitiveArray<PT, T>
where
    PT: Sample<Output = N> + Clone + 'static,
    N: Clone + 'static,
    T: ArrowPrimitiveType + 'static,
    T::Native: From<N>,
{
    type Output = ArrayRef;

    fn generate(&mut self, g: &mut Random) -> Self::Output {
        // Generate values
        let values: Vec<T::Native> = self
            .inner
            .generate(g)
            .into_iter()
            .map(T::Native::from)
            .collect();

        // Create a ScalarBuffer
        let buffer = ScalarBuffer::from(values);

        // Create PrimitiveArray and return as Arc<dyn Array>
        Arc::new(PrimitiveArray::<T>::new(buffer, None))
    }

    fn shrink(&self, _: Self::Output) -> Shrunk<Self::Output> {
        Box::new(std::iter::empty())
    }
}

#[derive(Clone)]
pub struct ProtoBoxedNullablePrimitiveArray<PT, T> {
    inner: ProtoNullablePrimitiveArray<PT, T>,
}

impl<PT, N, T> Sample for ProtoBoxedNullablePrimitiveArray<PT, T>
where
    PT: Sample<Output = Option<N>> + Clone + 'static,
    N: Clone + 'static,
    T: ArrowPrimitiveType + 'static,
    T::Native: From<N>,
{
    type Output = ArrayRef;

    fn generate(&mut self, g: &mut Random) -> Self::Output {
        self.inner.generate(g)
    }

    fn shrink(&self, _: Self::Output) -> Shrunk<Self::Output> {
        Box::new(std::iter::empty())
    }
}

pub fn boxed_nullable<GT, N, T>(len: Range<usize>, el: GT) -> ArrowSampler
where
    GT: Sample<Output = Option<N>> + Send + Sync + Clone + 'static,
    N: Clone + Send + Sync + 'static,
    T: ArrowPrimitiveType + Send + Sync + 'static,
    T::Native: From<N>,
{
    Box::new(ProtoBoxedNullablePrimitiveArray {
        inner: ProtoNullablePrimitiveArray {
            inner: VecSampler { length: len, el },
            _phantom: PhantomData::<T>,
        },
    })
}

pub fn boxed<GT, N, T>(len: Range<usize>, el: GT) -> ArrowSampler
where
    GT: Sample<Output = N> + Send + Sync + Clone + 'static,
    N: Clone + Send + Sync + 'static,
    T: ArrowPrimitiveType + Send + Sync + 'static,
    T::Native: From<N>,
{
    Box::new(ProtoPrimitiveArray {
        inner: VecSampler { length: len, el },
        _phantom: PhantomData::<T>,
    })
}

#[derive(Clone)]
struct Nullable<SI, V> {
    inner: SI,
    null: V,
}

impl<SI, V> Sample for Nullable<SI, V>
where
    SI: Sample,
    V: Sample<Output = bool>,
{
    type Output = Option<SI::Output>;
    fn generate(&mut self, g: &mut Random) -> Self::Output {
        if self.null.generate(g) {
            None
        } else {
            Some(self.inner.generate(g))
        }
    }

    fn shrink(&self, v: Self::Output) -> Shrunk<Self::Output> {
        if let Some(v) = v {
            Box::new(std::iter::once(None).chain(self.inner.shrink(v).map(Some)))
        } else {
            Box::new(std::iter::empty())
        }
    }
}

// Helper macro to create boxed_primitive functions
macro_rules! boxed_primitives {
    ($type:ty, $arrow_type:ty, $fn_name:ident) => {
        pub fn $fn_name<V>(len: Range<usize>, null: Option<V>) -> ArrowSampler
        where
            V: Sample<Output = bool> + Clone + Send + Sync + 'static,
        {
            match null {
                Some(null) => boxed_nullable::<_, $type, $arrow_type>(
                    len.clone(),
                    Nullable {
                        inner: arbitrary::<$type>(),
                        null,
                    },
                ),
                None => boxed::<_, $type, $arrow_type>(len.clone(), arbitrary::<$type>()),
            }
        }
    };
}

// Implement boxed_primitive functions
boxed_primitives!(i8, arrow_array::types::Int8Type, i8_array);
boxed_primitives!(i16, arrow_array::types::Int16Type, i16_array);
boxed_primitives!(i32, arrow_array::types::Int32Type, i32_array);
boxed_primitives!(i64, arrow_array::types::Int64Type, i64_array);
boxed_primitives!(u8, arrow_array::types::UInt8Type, u8_array);
boxed_primitives!(u16, arrow_array::types::UInt16Type, u16_array);
boxed_primitives!(u32, arrow_array::types::UInt32Type, u32_array);
boxed_primitives!(u64, arrow_array::types::UInt64Type, u64_array);
boxed_primitives!(f32, arrow_array::types::Float32Type, f32_array);
boxed_primitives!(f64, arrow_array::types::Float64Type, f64_array);

pub fn valid_float_array<V>(len: Range<usize>, null: Option<V>) -> ArrowSampler
where
    V: Sample<Output = bool> + Clone + Send + Sync + 'static,
{
    Box::new(sampler_choice([
        f32_array(len.clone(), null.clone()),
        f64_array(len, null),
    ]))
}

pub fn arbitrary_float_array<V>(len: Range<usize>, null: Option<V>) -> ArrowSampler
where
    V: Sample<Output = bool> + Clone + Send + Sync + 'static,
{
    valid_float_array(len, null)
}

pub fn arbitrary_int_array<V>(len: Range<usize>, null: Option<V>) -> ArrowSampler
where
    V: Sample<Output = bool> + Clone + Send + Sync + 'static,
{
    Box::new(sampler_choice([
        i8_array(len.clone(), null.clone()),
        i16_array(len.clone(), null.clone()),
        i32_array(len.clone(), null.clone()),
        i64_array(len.clone(), null.clone()),
    ]))
}

pub fn arbitrary_uint_array<V>(len: Range<usize>, null: Option<V>) -> ArrowSampler
where
    V: Sample<Output = bool> + Clone + Send + Sync + 'static,
{
    Box::new(sampler_choice([
        u8_array(len.clone(), null.clone()),
        u16_array(len.clone(), null.clone()),
        u32_array(len.clone(), null.clone()),
        u64_array(len.clone(), null.clone()),
    ]))
}

pub fn valid_primitive<V>(len: Range<usize>, null: Option<V>) -> ArrowSampler
where
    V: Sample<Output = bool> + Clone + Send + Sync + 'static,
{
    Box::new(sampler_choice([
        valid_float_array(len.clone(), null.clone()),
        arbitrary_int_array(len.clone(), null.clone()),
        arbitrary_uint_array(len.clone(), null.clone()),
    ]))
}

pub fn arbitrary_primitive<V>(len: Range<usize>, null: Option<V>) -> ArrowSampler
where
    V: Sample<Output = bool> + Clone + Send + Sync + 'static,
{
    valid_primitive(len, null)
}

#[cfg(test)]
mod tests {
    use sample_std::Chance;

    use super::*;

    #[test]
    fn gen_float() {
        let mut gen = valid_float_array(50..51, Some(Chance(0.5)));
        let mut r = Random::new();
        let arr = gen.generate(&mut r);
        assert_eq!(arr.len(), 50);
    }
}
