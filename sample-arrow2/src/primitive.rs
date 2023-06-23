//! Samplers for generating an arrow [`PrimitiveArray`].

use std::ops::Range;

use arrow2::{
    array::{Array, PrimitiveArray},
    types::NativeType,
};
use sample_std::{
    arbitrary, sampler_choice, valid_f32, valid_f64, Arbitrary, ArbitrarySampler, Random, Sample,
    Shrunk, VecSampler,
};

use crate::ArrowSampler;

#[derive(Debug, Clone)]
pub struct ProtoNullablePrimitiveArray<PT> {
    inner: VecSampler<Range<usize>, PT>,
}

fn to_primitive<T>(vec: Vec<Option<T>>) -> PrimitiveArray<T>
where
    T: NativeType + Arbitrary,
{
    PrimitiveArray::from_trusted_len_iter(vec.into_iter())
}

impl<PT, T> Sample for ProtoNullablePrimitiveArray<PT>
where
    PT: Sample<Output = Option<T>> + Clone + 'static,
    T: NativeType + Arbitrary,
{
    type Output = PrimitiveArray<T>;

    fn generate(&self, g: &mut Random) -> Self::Output {
        to_primitive(self.inner.generate(g))
    }

    fn shrink(&self, v: Self::Output) -> Shrunk<Self::Output> {
        let vec = v.iter().map(|el| el.cloned()).collect();
        Box::new(self.inner.shrink(vec).map(to_primitive))
    }
}

#[derive(Debug, Clone)]
pub struct ProtoPrimitiveArray<PT> {
    inner: VecSampler<Range<usize>, PT>,
}

impl<PT, T> Sample for ProtoPrimitiveArray<PT>
where
    PT: Sample<Output = T> + Clone + 'static,
    T: NativeType,
{
    type Output = PrimitiveArray<T>;

    fn generate(&self, g: &mut Random) -> Self::Output {
        PrimitiveArray::from_trusted_len_values_iter(self.inner.generate(g).into_iter())
    }

    fn shrink(&self, v: Self::Output) -> Shrunk<Self::Output> {
        let vec = v.values_iter().cloned().collect();
        Box::new(
            self.inner
                .shrink(vec)
                .map(IntoIterator::into_iter)
                .map(PrimitiveArray::from_trusted_len_values_iter),
        )
    }
}

pub fn boxed_nullable<GT, T>(len: Range<usize>, el: GT) -> ArrowSampler
where
    GT: Sample<Output = Option<T>> + Send + Sync + Clone + 'static,
    T: NativeType + Arbitrary,
{
    Box::new(ProtoBoxedNullablePrimitiveArray {
        inner: ProtoNullablePrimitiveArray {
            inner: VecSampler { length: len, el },
        },
    })
}

pub fn boxed<GT, T>(len: Range<usize>, el: GT) -> ArrowSampler
where
    GT: Sample<Output = T> + Send + Sync + Clone + 'static,
    T: NativeType + Arbitrary,
{
    Box::new(
        ProtoPrimitiveArray {
            inner: VecSampler { length: len, el },
        }
        .try_convert(PrimitiveArray::boxed, |boxed| {
            if boxed.validity().is_none() {
                boxed.as_any().downcast_ref::<PrimitiveArray<T>>().cloned()
            } else {
                None
            }
        }),
    )
}

#[derive(Clone)]
pub struct ProtoBoxedNullablePrimitiveArray<PT> {
    inner: ProtoNullablePrimitiveArray<PT>,
}

impl<GT, T> Sample for ProtoBoxedNullablePrimitiveArray<GT>
where
    GT: Sample<Output = Option<T>> + Clone + 'static,
    T: NativeType + Arbitrary,
{
    type Output = Box<dyn Array>;

    fn generate(&self, g: &mut Random) -> Self::Output {
        self.inner.generate(g).boxed()
    }

    fn shrink(&self, v: Self::Output) -> Shrunk<Self::Output> {
        Box::new(
            v.as_any()
                .downcast_ref::<PrimitiveArray<T>>()
                .cloned()
                .into_iter()
                .flat_map(move |arr| self.inner.shrink(arr.clone()).map(|arr| arr.boxed())),
        )
    }
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
    fn generate(&self, g: &mut Random) -> Self::Output {
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

pub fn boxed_primitive<T, V>(
    el: ArbitrarySampler<T>,
    len: Range<usize>,
    null: Option<V>,
) -> ArrowSampler
where
    T: Arbitrary + NativeType,
    V: Sample<Output = bool> + Clone + Send + Sync + 'static,
{
    match null {
        Some(null) => boxed_nullable(len.clone(), Nullable { inner: el, null }),
        None => boxed(len.clone(), el),
    }
}

pub fn arbitrary_boxed_primitive<T, V>(len: Range<usize>, null: Option<V>) -> ArrowSampler
where
    V: Sample<Output = bool> + Clone + Send + Sync + 'static,
    T: NativeType + Arbitrary,
{
    boxed_primitive(arbitrary::<T>(), len, null)
}

// todo: arbitrary_daysms_array

pub fn valid_float_array<V>(len: Range<usize>, null: Option<V>) -> ArrowSampler
where
    V: Sample<Output = bool> + Clone + Send + Sync + 'static,
{
    Box::new(sampler_choice(vec![
        // todo: f16
        boxed_primitive(valid_f32(), len.clone(), null.clone()),
        boxed_primitive(valid_f64(), len, null),
    ]))
}

pub fn arbitrary_float_array<V>(len: Range<usize>, null: Option<V>) -> ArrowSampler
where
    V: Sample<Output = bool> + Clone + Send + Sync + 'static,
{
    Box::new(sampler_choice(vec![
        // todo: f16
        arbitrary_boxed_primitive::<f32, _>(len.clone(), null.clone()),
        arbitrary_boxed_primitive::<f32, _>(len, null),
    ]))
}

pub fn arbitrary_int_array<V>(len: Range<usize>, null: Option<V>) -> ArrowSampler
where
    V: Sample<Output = bool> + Clone + Send + Sync + 'static,
{
    Box::new(sampler_choice(vec![
        arbitrary_boxed_primitive::<i8, _>(len.clone(), null.clone()),
        arbitrary_boxed_primitive::<i16, _>(len.clone(), null.clone()),
        arbitrary_boxed_primitive::<i32, _>(len.clone(), null.clone()),
        arbitrary_boxed_primitive::<i64, _>(len.clone(), null.clone()),
    ]))
}

// todo: arbitrary_monthsdaysnano_array

pub fn arbitrary_uint_array<V>(len: Range<usize>, null: Option<V>) -> ArrowSampler
where
    V: Sample<Output = bool> + Clone + Send + Sync + 'static,
{
    Box::new(sampler_choice(vec![
        arbitrary_boxed_primitive::<u8, _>(len.clone(), null.clone()),
        arbitrary_boxed_primitive::<u16, _>(len.clone(), null.clone()),
        arbitrary_boxed_primitive::<u32, _>(len.clone(), null.clone()),
        arbitrary_boxed_primitive::<u64, _>(len.clone(), null.clone()),
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
    Box::new(sampler_choice([
        arbitrary_float_array(len.clone(), null.clone()),
        arbitrary_int_array(len.clone(), null.clone()),
        arbitrary_uint_array(len.clone(), null.clone()),
    ]))
}

#[cfg(test)]
mod tests {
    use sample_std::Chance;

    use super::*;

    #[test]
    fn gen_float() {
        let gen = valid_float_array(50..51, Some(Chance(0.5)));
        let mut r = Random::new();
        let arr = gen.generate(&mut r);
        assert_eq!(arr, arr);
    }
}
