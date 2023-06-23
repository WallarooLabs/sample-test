//! Core sampling strategies, along with useful implementations for samplers on
//! types from [`std`] and the [`quickcheck`] crate.
//!
//! The core of this library is the [`Sample`] trait, which uses [`Random`] to
//! generate arbitrary values from a sampled `Output` type with a custom
//! "sampling strategy". It also defines a procedure for "shrinking" generated
//! values, which can be used to generate simple counterexamples against
//! expected properties.
//!
//! This library is generally intended for usage alongside the [`sample_test`][1]
//! crate. See that crate for macros and examples for using samplers within unit
//! tests.
//!
//! # Sampling Strategies
//!
//! The simplest [`Sample`] implementation is for [`Range`]. It is a sampler
//! that generates values uniformly from the given range, and attempts to shrink
//! down to the start of the range:
//!
//! ```
//! use sample_std::{Sample, Random};
//!
//! let s = 10..100;
//! let v = s.generate(&mut Random::new());
//! assert!(s.contains(&v));
//! let mut shrunk = s.shrink(v);
//! assert_eq!(shrunk.next(), Some(s.start));
//! if v > s.start {
//!     assert!(shrunk.next().unwrap() < v)
//! }
//! ```
//!
//! Samplers are defined for tuples of samplers up to size 8, which can be
//! used in concert with [`Sample::try_convert`] to combine samplers into a
//! sampler for a larger type:
//!
//! ```
//! use sample_std::{Chance, Sample, VecSampler, choice};
//!
//! struct Large {
//!     values: Vec<usize>,
//!     is_blue: bool,
//!     name: String,
//! }
//!
//! let sampler = (
//!     VecSampler { length: 0..10, el: 5..20 },
//!     Chance(0.5),
//!     choice(["cora".to_string(), "james".to_string()])
//! ).try_convert(
//!     |(values, is_blue, name)| Large { values, is_blue, name },
//!     |large| Some((large.values, large.is_blue, large.name))
//! );
//! ```
//!
//! For an example of sampling an `enum`, see [`sampler_choice`].
//!
//! # Prior Work
//!
//! This crate is heavily inspired by [`quickcheck`]. It builds upon it, in
//! particular by defining samplers for [`Arbitrary`] (see [`arbitrary`]). Many
//! methods and structs in here were derived from their [`quickcheck`]
//! counterparts.
//!
//! It attempts to iterate and improve on the [`quickcheck`] core idea:
//!
//! - Allow definition of multiple sampling strategies for the same type.
//! - No need to define newtypes for custom sampling strategies.
//!
//! There is still some cruft and weirdness from this early attempt to combine
//! these worldviews:
//!
//! - The concept of `size` isn't really necessary in a world with sampling
//!   strategies.
//! - The [`Random`] struct could probably just become a type definition around
//!   the underlying `rng`.
//!
//! The core idea for sampling "strategies" comes from [`proptest`][2], which
//! uses macros instead of combinators for composition, and has more complex
//! shrinking functionality.
//!
//! [1]: https://docs.rs/sample_test/latest/sample_test/
//! [2]: https://docs.rs/proptest/latest/proptest/
pub use quickcheck::{Arbitrary, Gen, TestResult, Testable};
use std::{marker::PhantomData, ops::Range};

pub use rand;
use rand::{
    distributions::uniform::SampleUniform, prelude::Distribution, seq::SliceRandom, Rng,
    SeedableRng,
};

pub mod recursive;

/// [`Random`] represents a PRNG.
///
/// It is a reimplementation of [`quickcheck::Gen`], which does not export the
/// methods we need to properly generate random values.
///
/// It is unspecified whether this is a secure RNG or not. Therefore, callers
/// should assume it is insecure.
pub struct Random {
    pub rng: rand::rngs::SmallRng,
}

impl Random {
    pub fn arbitrary<T: Arbitrary>(&self) -> T {
        let mut qcg = Gen::new(100);

        Arbitrary::arbitrary(&mut qcg)
    }

    /// Returns a new [Random] instance.
    pub fn new() -> Self {
        Random {
            rng: rand::rngs::SmallRng::from_entropy(),
        }
    }

    pub fn from_seed(seed: u64) -> Self {
        let seed: Vec<u8> = seed
            .to_be_bytes()
            .into_iter()
            .chain(std::iter::repeat(0))
            .take(32)
            .collect();
        Random {
            rng: rand::rngs::SmallRng::from_seed(seed[0..32].try_into().unwrap()),
        }
    }

    /// Choose among the possible alternatives in the slice given. If the slice
    /// is empty, then `None` is returned. Otherwise, a non-`None` value is
    /// guaranteed to be returned.
    pub fn choose<'a, T>(&mut self, slice: &'a [T]) -> Option<&'a T> {
        slice.choose(&mut self.rng)
    }

    pub fn gen<T>(&mut self) -> T
    where
        rand::distributions::Standard: rand::distributions::Distribution<T>,
    {
        self.rng.gen()
    }

    pub fn gen_range<T, R>(&mut self, range: R) -> T
    where
        T: rand::distributions::uniform::SampleUniform,
        R: rand::distributions::uniform::SampleRange<T>,
    {
        self.rng.gen_range(range)
    }
}

/// An [`Iterator`] of "smaller" values derived from a given value.
pub type Shrunk<'a, T> = Box<dyn Iterator<Item = T> + 'a>;

/// User-defined strategies for generating and shrinking an `Output` type.
pub trait Sample {
    type Output;

    /// Randomly generate the requested type.
    fn generate(&self, g: &mut Random) -> Self::Output;

    /// Shrink the given value into a "smaller" value. Defaults to an empty
    /// iterator (which represents that the value cannot be shrunk).
    fn shrink(&self, _: Self::Output) -> Shrunk<'_, Self::Output> {
        Box::new(std::iter::empty())
    }

    /// Convert this sampler into a new sampler with `from` and `try_into`
    /// functions:
    ///
    /// ```
    /// use sample_std::{Sample, VecSampler};
    ///
    /// struct Wrapper {
    ///     vec: Vec<usize>
    /// }
    ///
    /// impl Wrapper {
    ///     fn new(vec: Vec<usize>) -> Self {
    ///         Self { vec }
    ///     }
    /// }
    ///
    /// let sampler = VecSampler { length: 10..20, el: 1..5 }.try_convert(
    ///     Wrapper::new,
    ///     |w| Some(w.vec)
    /// );
    /// ```
    ///
    /// [`Sample::generate`] will use `from` to convert the inner sampled value
    /// to the desired type.
    ///
    /// [`Sample::shrink`] will use `try_into` to convert the desired type back
    /// to the inner sampled type, if possible. The inner `shrink` method will
    /// be called on that type, and all values will be converted back to the
    /// target type again with `into`.
    fn try_convert<T, I, F>(self, from: F, try_into: I) -> TryConvert<Self, F, I>
    where
        Self: Sized,
        F: Fn(Self::Output) -> T + Copy,
        I: Fn(T) -> Option<Self::Output>,
    {
        TryConvert {
            inner: self,
            from,
            try_into,
        }
    }

    /// "Zip" two samplers together. Functionally equivalent to `(self, other)`.
    fn zip<OS>(self, other: OS) -> Zip<Self, OS>
    where
        Self: Sized,
        OS: Sample,
    {
        Zip { t: (self, other) }
    }

    /// "Resampling" method for chaining samplers.
    ///
    /// For sampling, use this sampler as a  "supersampler" that creates a
    /// "seed" value. The provided function then converts this seed into an
    /// inner sampler that is used to generate a final value.
    ///
    /// This value is returned within a [Chained] wrapper that also captures the
    /// seed. This allows us to use the "supersampler" in the shrinking process.
    /// This then shrinks the seed, and then "resamples" (generates new samples)
    /// with the shrunk inner sampler.
    ///
    /// Note that the resulting sampler will only perform a very shallow search
    /// (`subsamples`) of the shrunk inner sampler space.
    fn chain_resample<F, RS>(self, transform: F, subsamples: usize) -> ChainResample<Self, F>
    where
        Self: Sized,
        F: Fn(Self::Output) -> RS,
        RS: Sample,
    {
        ChainResample {
            supersampler: self,
            transform,
            subsamples,
        }
    }
}

/// See [`Sample::try_convert`].
#[derive(Clone)]
pub struct TryConvert<P, F, I> {
    inner: P,
    from: F,
    try_into: I,
}

impl<P, F, I, T> Sample for TryConvert<P, F, I>
where
    P: Sample,
    F: Fn(P::Output) -> T + Copy,
    I: Fn(T) -> Option<P::Output>,
{
    type Output = T;

    fn generate(&self, g: &mut Random) -> Self::Output {
        (self.from)(P::generate(&self.inner, g))
    }

    fn shrink(&self, v: Self::Output) -> Shrunk<Self::Output> {
        Box::new(
            (self.try_into)(v)
                .into_iter()
                .flat_map(|v| P::shrink(&self.inner, v))
                .map(self.from),
        )
    }
}

/// See [`Sample::zip`].
#[derive(Clone)]
pub struct Zip<A, B> {
    t: (A, B),
}

impl<A, B> Sample for Zip<A, B>
where
    A: Sample,
    B: Sample,
    A::Output: Clone,
    B::Output: Clone,
{
    type Output = (A::Output, B::Output);

    fn generate(&self, g: &mut Random) -> Self::Output {
        self.t.generate(g)
    }

    fn shrink(&self, v: Self::Output) -> Shrunk<Self::Output> {
        self.t.shrink(v)
    }
}

macro_rules! replace_expr {
    ($_t:tt $sub:expr) => {
        $sub
    };
}

macro_rules! none_pad {
    (($cur:ident) $($post: ident)*) => {
        (Some($cur), $(replace_expr!($post  None)),*)
    };
    ($($pre:ident)+ ($cur:ident) $($post: ident)*) => {
        ($(replace_expr!($pre  None)),*, Some($cur), $(replace_expr!($post  None)),*)
    };
}

macro_rules! shrink_tuple {
    ($v:ident () $($sample: ident)*) => {
        let ($(casey::lower!($sample)),*) = $v.clone();
        let r = std::iter::empty();
        shrink_tuple!(r $v () ($($sample)*));
    };
    ($r:ident $v:ident () ($cur:ident $($sample: ident)*)) => {
        let (casey::lower!(ref $cur), $(casey::lower!(ref $sample)),*) = $v;
        let r = $r.chain(
            $cur.shrink(casey::lower!($cur).clone()).map(move |$cur| {
                none_pad!(($cur) $($sample)*)
            })
        );
        shrink_tuple!(r $v ($cur) ($($sample)*));
    };
    ($r:ident $v:ident ($($pre:ident)+) ($cur:ident $($sample: ident)*)) => {
        let ($(casey::lower!(ref $pre)),*, casey::lower!(ref $cur), $(casey::lower!(ref $sample)),*) = $v;
        let r = $r.chain(
            $cur.shrink(casey::lower!($cur).clone()).map(move |$cur| {
                none_pad!($($pre)* ($cur) $($sample)*)
            })
        );
        shrink_tuple!(r $v ($($pre)* $cur) ($($sample)*));
    };
    ($r:ident $v:ident ($($pre:ident)*) ()) => {
        let ($(casey::lower!($pre)),*) = $v;
        return Box::new($r.map(move |($($pre),*)| {
            ($($pre.unwrap_or(casey::lower!($pre).clone())),*)
        }))
    };
}

macro_rules! sample_tuple {
    ($($name: ident),*) => {

impl<$($name),*> Sample for ($($name),*,)
where
    $($name: Sample),*,
    $($name::Output: Clone),*,
{
    type Output = ($($name::Output),*,);

    #[allow(non_snake_case)]
    fn generate(&self, r: &mut Random) -> Self::Output {
        let ($(casey::lower!($name)),*,) = &self;
        ($(casey::lower!($name).generate(r)),*,)
    }

    #[allow(non_snake_case)]
    fn shrink(&self, v: Self::Output) -> Shrunk<Self::Output> {
        let ($($name),*,) = self;
        shrink_tuple!(v () $($name)*);
    }
}

    }
}

impl<A> Sample for (A,)
where
    A: Sample,
{
    type Output = (A::Output,);

    fn generate(&self, g: &mut Random) -> Self::Output {
        (self.0.generate(g),)
    }

    fn shrink(&self, v: Self::Output) -> Shrunk<Self::Output> {
        Box::new(self.0.shrink(v.0).map(|v| (v,)))
    }
}

sample_tuple!(A, B);
sample_tuple!(A, B, C);
sample_tuple!(A, B, C, D);
sample_tuple!(A, B, C, D, E);
sample_tuple!(A, B, C, D, E, F);
sample_tuple!(A, B, C, D, E, F, G);
sample_tuple!(A, B, C, D, E, F, G, H);

/// See [`Sample::chain_resample`].
#[derive(Clone, Debug)]
pub struct ChainResample<S, F> {
    supersampler: S,
    transform: F,
    subsamples: usize,
}

/// Capture the `seed` used to generate the given `value`.
#[derive(Clone, Debug)]
pub struct Chained<S, V> {
    seed: S,
    pub value: V,
}

impl<S, F, SS> Sample for ChainResample<S, F>
where
    S: Sample,
    S::Output: Clone,
    SS: Sample + 'static,
    F: Fn(S::Output) -> SS,
{
    type Output = Chained<S::Output, SS::Output>;

    fn generate(&self, g: &mut Random) -> Self::Output {
        let seed = self.supersampler.generate(g);
        let value = (self.transform)(seed.clone()).generate(g);

        Chained { seed, value }
    }

    fn shrink(&self, v: Self::Output) -> Shrunk<Self::Output> {
        Box::new(self.supersampler.shrink(v.seed).flat_map(|shrunk_seed| {
            let mut g = Random::new();
            let sampler = (self.transform)(shrunk_seed.clone());
            (0..self.subsamples).map(move |_| Chained {
                seed: shrunk_seed.clone(),
                value: sampler.generate(&mut g),
            })
        }))
    }
}

impl<T> Sample for Box<dyn Sample<Output = T>> {
    type Output = T;

    fn generate(&self, g: &mut Random) -> Self::Output {
        self.as_ref().generate(g)
    }

    fn shrink(&self, v: Self::Output) -> Shrunk<'_, Self::Output> {
        self.as_ref().shrink(v)
    }
}

impl<T> Sample for Box<dyn Sample<Output = T> + Send + Sync> {
    type Output = T;

    fn generate(&self, g: &mut Random) -> Self::Output {
        self.as_ref().generate(g)
    }

    fn shrink(&self, v: Self::Output) -> Shrunk<'_, Self::Output> {
        self.as_ref().shrink(v)
    }
}

/// Generate a boolean value with the specified probability (in the range
/// `0..=1`).
#[derive(Debug, Clone)]
pub struct Chance(pub f32);

impl Sample for Chance {
    type Output = bool;

    fn generate(&self, g: &mut Random) -> Self::Output {
        g.gen_range(0.0..1.0) < self.0
    }
}

/// Bridge for creating a [`Sample`] from an [`Arbitrary`] type.
#[derive(Clone)]
pub struct ArbitrarySampler<T> {
    size: Option<usize>,
    phantom: PhantomData<T>,
    validate: fn(&T) -> bool,
}

impl<T: Arbitrary> Sample for ArbitrarySampler<T> {
    type Output = T;

    fn generate(&self, _: &mut Random) -> Self::Output {
        let mut g = Gen::new(self.size.unwrap_or(100));
        for _ in 0..1000 {
            let value = Arbitrary::arbitrary(&mut g);
            if (self.validate)(&value) {
                return value;
            }
        }
        panic!("could not find valid value after 1000 iterations")
    }

    fn shrink(&self, v: Self::Output) -> Shrunk<'_, Self::Output> {
        Arbitrary::shrink(&v)
    }
}

/// Sampler for any type implementing [`Arbitrary`].
pub fn arbitrary<T: Arbitrary>() -> ArbitrarySampler<T> {
    ArbitrarySampler {
        size: None,
        phantom: PhantomData,
        validate: |_| true,
    }
}

/// Sampler for non-NaN [f32]
pub fn valid_f32() -> ArbitrarySampler<f32> {
    ArbitrarySampler {
        size: None,
        phantom: PhantomData,
        validate: |f| !f.is_nan(),
    }
}

/// Sampler for non-NaN [f64]
pub fn valid_f64() -> ArbitrarySampler<f64> {
    ArbitrarySampler {
        size: None,
        phantom: PhantomData,
        validate: |f| !f.is_nan(),
    }
}

/// Sampler that always generates a fixed value.
#[derive(Debug, Clone)]
pub struct Always<T>(pub T);

impl<T: Clone> Sample for Always<T> {
    type Output = T;

    fn generate(&self, _: &mut Random) -> Self::Output {
        self.0.clone()
    }
}

/// Sample from a list of `choice` values.
///
/// [`Sample::shrink`] will attempt to shrink down to the first element in the
/// [`Choice`]:
///
/// ```
/// use sample_std::{Random, Sample, choice};
///
/// let sampler = choice(["cora", "coraline"]);
/// let name = sampler.generate(&mut Random::new());
/// assert!(name.starts_with("cora"));
///
/// assert_eq!(sampler.shrink("coraline").next(), Some("cora"));
/// ```
pub fn choice<T, II>(choices: II) -> Choice<T>
where
    T: Clone + PartialEq,
    II: IntoIterator<Item = T>,
{
    Choice {
        choices: choices.into_iter().collect(),
    }
}

/// See [choice].
#[derive(Clone, Debug)]
pub struct Choice<T> {
    pub choices: Vec<T>,
}

impl<T> Sample for Choice<T>
where
    T: Clone + PartialEq,
{
    type Output = T;

    fn generate(&self, g: &mut Random) -> Self::Output {
        g.choose(&self.choices).unwrap().clone()
    }

    fn shrink(&self, v: Self::Output) -> Shrunk<'_, Self::Output> {
        let ix = self.choices.iter().position(|el| el == &v).unwrap_or(0);
        Box::new((0..ix).map(|shrunk_ix| self.choices[shrunk_ix].clone()))
    }
}

/// Sample values from a sampler randomly drawn from a list of `choices`.
///
/// `shrink` attempts to run the [`Sample::shrink`] method from each specified
/// sampler in order. This allows [`sampler_choice`] to work with choices that
/// generate `enum` variants (e.g. via [`Sample::try_convert`]):
///
/// ```
/// use std::boxed::Box;
/// use sample_std::{Sample, sampler_choice};
///
/// #[derive(Clone)]
/// enum Widget {
///     Bib(usize),
///     Bob(usize)
/// }
///
/// type WidgetSampler = Box<dyn Sample<Output = Widget>>;
///
/// let bibs: WidgetSampler = Box::new((0..100).try_convert(Widget::Bib, |v| match v {
///     Widget::Bib(u) => Some(u),
///     _ => None,
/// }));
///
/// let bobs: WidgetSampler = Box::new((100..200).try_convert(Widget::Bob, |v| match v {
///     Widget::Bob(u) => Some(u),
///     _ => None,
/// }));
///
/// let widgets = sampler_choice([bibs, bobs]);
/// ```
///
/// This may lead to unexpected shrinking behavior if every sampler in the
/// [`SamplerChoice`] can shrink a given value.
pub fn sampler_choice<C, II>(choices: II) -> SamplerChoice<C>
where
    II: IntoIterator<Item = C>,
    C: Sample,
    <C as Sample>::Output: Clone,
{
    SamplerChoice {
        choices: choices.into_iter().collect(),
    }
}

/// See [`sampler_choice`].
#[derive(Clone, Debug)]
pub struct SamplerChoice<C> {
    pub choices: Vec<C>,
}

impl<C> SamplerChoice<C> {
    pub fn or(self, other: Self) -> Self {
        Self {
            choices: self
                .choices
                .into_iter()
                .chain(other.choices.into_iter())
                .collect(),
        }
    }
}

impl<C, T> Sample for SamplerChoice<C>
where
    C: Sample<Output = T>,
    T: Clone + 'static,
{
    type Output = T;

    fn generate(&self, g: &mut Random) -> Self::Output {
        g.choose(&self.choices).unwrap().generate(g)
    }

    fn shrink(&self, v: Self::Output) -> Shrunk<'_, Self::Output> {
        Box::new(self.choices.iter().flat_map(move |c| c.shrink(v.clone())))
    }
}

impl<T, I> Sample for Range<T>
where
    T: SampleUniform + Clone + PartialOrd + 'static,
    Range<T>: IntoIterator<IntoIter = I>,
    I: DoubleEndedIterator<Item = T> + 'static,
{
    type Output = T;

    fn generate(&self, g: &mut Random) -> Self::Output {
        g.gen_range(self.clone())
    }

    fn shrink(&self, v: Self::Output) -> Shrunk<'_, Self::Output> {
        if self.start != v {
            Box::new(
                std::iter::once(self.start.clone())
                    .chain((self.start.clone()..v.clone()).into_iter().rev().take(1)),
            )
        } else {
            Box::new(std::iter::empty())
        }
    }
}

/// Sample strings given a "valid" regular expression.
///
/// Shrinking is done by shortening the string and testing if the expression
/// still matches.
#[derive(Clone, Debug)]
pub struct Regex {
    pub dist: rand_regex::Regex,
    pub re: regex::Regex,
}

impl Regex {
    /// Create a new [Regex] sampler with the given `pattern` string.
    pub fn new(pattern: &str) -> Self {
        Regex {
            dist: rand_regex::Regex::compile(pattern, 100).unwrap(),
            re: regex::Regex::new(pattern).unwrap(),
        }
    }
}

impl Sample for Regex {
    type Output = String;

    fn generate(&self, g: &mut Random) -> Self::Output {
        self.dist.sample(&mut g.rng)
    }

    fn shrink(&self, v: Self::Output) -> Shrunk<'_, Self::Output> {
        let re = self.re.clone();
        // this obviously could be improved via deeper integration with the
        // underlying regex, but the generation library does not appear to
        // expose interfaces to do so
        Box::new(Iterator::flat_map(0..v.len(), move |ix| {
            let mut shrunk: String = String::with_capacity(v.len());
            shrunk.push_str(&v[0..ix]);
            shrunk.push_str(&v[ix..]);

            if re.is_match(&shrunk) {
                Some(shrunk)
            } else {
                None
            }
        }))
    }
}

/// Sample a [`Vec`] with a length drawn from a `usize` [`Sample`] and elements
/// drawn from the `el` sampler.
///
/// Shrinking attempts to first shrink the length, and then shrink each element
/// within the [`Vec`].
#[derive(Debug, Clone)]
pub struct VecSampler<S, I> {
    pub length: S,
    pub el: I,
}

impl<S, I, T> Sample for VecSampler<S, I>
where
    S: Sample<Output = usize>,
    I: Sample<Output = T>,
    T: Clone + 'static,
{
    type Output = Vec<T>;

    fn generate(&self, g: &mut Random) -> Self::Output {
        Iterator::map(0..self.length.generate(g), |_| self.el.generate(g)).collect()
    }

    fn shrink(&self, v: Self::Output) -> Shrunk<'_, Self::Output> {
        Box::new(self.length.shrink(v.len()).flat_map(move |new_len| {
            assert!(new_len < v.len());
            let gap = v.len() - new_len;

            let iv = v.clone();
            let iv2 = v.clone();

            Iterator::map(0..new_len, move |cut| {
                iv[0..cut]
                    .iter()
                    .chain(iv[cut + gap..].iter())
                    .cloned()
                    .collect()
            })
            .chain(Iterator::flat_map(0..v.len(), move |ix| {
                let vref = iv2.clone();
                let el = vref[ix].clone();
                self.el.shrink(el).map(move |shrunk| {
                    let mut copy: Vec<T> = vref.clone();
                    copy[ix] = shrunk.clone();
                    copy.clone()
                })
            }))
        }))
    }
}

/// Use a [`Vec`] of `samplers` to generate a new [`Vec`] of equal length, where
/// each sampler in the [`Vec`] is used to sample the value at its corresponding
/// position.
///
/// Shrinking proceeds per-element within the generated [`Vec`] in index order.
pub fn sample_all<S>(samplers: Vec<S>) -> SampleAll<S>
where
    S: Sample,
{
    SampleAll { samplers }
}

/// See [`sample_all`].
pub struct SampleAll<S> {
    samplers: Vec<S>,
}

impl<S> Sample for SampleAll<S>
where
    S: Sample,
    S::Output: Clone,
{
    type Output = Vec<S::Output>;

    fn generate(&self, g: &mut Random) -> Self::Output {
        self.samplers.iter().map(|s| s.generate(g)).collect()
    }

    fn shrink(&self, v: Self::Output) -> Shrunk<'_, Self::Output> {
        Box::new((0..self.samplers.len()).flat_map(move |ix| {
            let mut updated = v.clone();

            self.samplers[ix]
                .shrink(updated[ix].clone())
                .map(move |sv| {
                    updated[ix] = sv;
                    updated.clone()
                })
        }))
    }
}

#[cfg(test)]
mod tests {}
