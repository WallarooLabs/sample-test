use arrow2::{array::Array, bitmap::Bitmap};
use sample_std::{Random, Sample, SampleAll, Shrunk};

pub mod array;
pub mod chunk;
pub mod datatypes;
pub mod fixed_size_list;
pub mod list;
pub mod primitive;
pub mod struct_;

pub type ArrowSampler = Box<dyn Sample<Output = Box<dyn Array>> + Send + Sync>;

pub(crate) fn generate_validity<V>(
    null: &mut Option<V>,
    g: &mut Random,
    len: usize,
) -> Option<Bitmap>
where
    V: Sample<Output = bool>,
{
    null.as_mut().map(|null| {
        Bitmap::from_trusted_len_iter(std::iter::repeat(()).take(len).map(|_| !null.generate(g)))
    })
}

pub trait SetLen {
    fn set_len(&mut self, len: usize);
}

#[derive(Debug, Clone)]
pub struct AlwaysValid;

impl Sample for AlwaysValid {
    type Output = Option<Bitmap>;

    fn generate(&mut self, _: &mut sample_std::Random) -> Self::Output {
        None
    }

    fn shrink(&self, _: Self::Output) -> Shrunk<Self::Output> {
        Box::new(std::iter::empty())
    }
}

impl SetLen for AlwaysValid {
    fn set_len(&mut self, _: usize) {}
}

impl<C, I, O> SetLen for sample_std::TryConvert<C, I, O>
where
    C: SetLen,
{
    fn set_len(&mut self, len: usize) {
        self.inner.set_len(len)
    }
}

impl<S, F, I> SampleLen for sample_std::TryConvert<S, F, I>
where
    S: Sample + SetLen,
    F: Fn(S::Output) -> Box<dyn Array>,
    I: Fn(Box<dyn Array>) -> Option<S::Output>,
{
}

impl<C> SetLen for sample_std::SamplerChoice<C>
where
    C: SetLen,
{
    fn set_len(&mut self, len: usize) {
        for choice in &mut self.choices {
            choice.set_len(len);
        }
    }
}

impl SampleLen for sample_std::SamplerChoice<ArrowLenSampler> {}

impl<S: SetLen> SetLen for SampleAll<S> {
    fn set_len(&mut self, len: usize) {
        for sampler in &mut self.samplers {
            sampler.set_len(len);
        }
    }
}

impl Sample for Box<dyn SampleLen> {
    type Output = Box<dyn Array>;

    fn generate(&mut self, g: &mut Random) -> Self::Output {
        self.as_mut().generate(g)
    }

    fn shrink(&self, v: Self::Output) -> Shrunk<'_, Self::Output> {
        self.as_ref().shrink(v)
    }
}

pub trait SampleLen: Sample<Output = Box<dyn Array>> + SetLen {}

pub type ArrowLenSampler = Box<dyn SampleLen>;

impl SetLen for ArrowLenSampler {
    fn set_len(&mut self, len: usize) {
        self.as_mut().set_len(len)
    }
}

impl SampleLen for ArrowLenSampler {}

pub struct FixedLenSampler<A> {
    pub len: usize,
    pub array: A,
}

impl<A> Sample for FixedLenSampler<A>
where
    A: Sample + SetLen,
    A::Output: Clone,
{
    type Output = A::Output;

    fn generate(&mut self, g: &mut Random) -> Self::Output {
        self.array.set_len(self.len);
        self.array.generate(g)
    }

    fn shrink(&self, array: Self::Output) -> Shrunk<Self::Output> {
        self.array.shrink(array)
    }
}
