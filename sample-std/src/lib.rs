pub use quickcheck::{Arbitrary, Gen, TestResult, Testable};
use std::{marker::PhantomData, ops::Range};

pub use rand;
use rand::{
    distributions::uniform::SampleUniform, prelude::Distribution, seq::SliceRandom, Rng,
    SeedableRng,
};

pub mod recursive;

/// [Random] represents a PRNG.
///
/// It is a reimplementation of [quickcheck::Gen], which does not export the
/// methods we need to properly generate random values.
///
/// It is unspecified whether this is a secure RNG or not. Therefore, callers
/// should assume it is insecure.
pub struct Random {
    pub rng: rand::rngs::SmallRng,
    size: usize,
}

impl Random {
    pub fn arbitrary<T: Arbitrary>(&self) -> T {
        let mut qcg = Gen::new(self.size);

        Arbitrary::arbitrary(&mut qcg)
    }

    /// Returns a `Gen` with the given size configuration.
    ///
    /// The `size` parameter controls the size of random values generated.
    /// For example, it specifies the maximum length of a randomly generated
    /// vector, but is and should not be used to control the range of a
    /// randomly generated number. (Unless that number is used to control the
    /// size of a data structure.)
    pub fn new(size: usize) -> Self {
        Random {
            rng: rand::rngs::SmallRng::from_entropy(),
            size: size,
        }
    }

    pub fn from_seed(seed: u64, size: usize) -> Self {
        let seed: Vec<u8> = seed
            .to_be_bytes()
            .into_iter()
            .chain(std::iter::repeat(0))
            .take(32)
            .collect();
        Random {
            rng: rand::rngs::SmallRng::from_seed(seed[0..32].try_into().unwrap()),
            size: size,
        }
    }

    /// Returns the size configured with this generator.
    pub fn size(&self) -> usize {
        self.size
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

/// An [Iterator] of "smaller" values derived from a given value.
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

    fn wrap<T, I, II, F>(self, into: I, from: F) -> FlatMap<Self, I, F>
    where
        Self: Sized,
        I: Fn(T) -> II,
        II: IntoIterator<Item = Self::Output> + 'static,
        F: Fn(Self::Output) -> T + Copy,
    {
        FlatMap {
            inner: self,
            into,
            from,
        }
    }

    fn zip<OS>(self, other: OS) -> Zip<Self, OS>
    where
        Self: Sized,
        OS: Sample,
    {
        Zip { a: self, b: other }
    }

    /// "Resampling" method for chaining generators.
    ///
    /// For sampling, use this sampler as a  "supersampler" that creates a
    /// "seed" value. The provided function then converts this seed into an
    /// inner sampler that is used to generate a final value.
    ///
    /// This value is returned within a `Chained` wrapper that also captures the
    /// seed. This allows us to use the "supersampler" in the shrinking process.
    /// This then shrinks the seed, and then "resamples" (generates new samples)
    /// with the shrunk inner sampler.
    ///
    /// Note that this will can only perform a very shallow search of the shrunk
    /// inner sampler space.
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

#[derive(Clone)]
pub struct FlatMap<P, I, F> {
    inner: P,
    into: I,
    from: F,
}

impl<P, II, I, F, T> Sample for FlatMap<P, I, F>
where
    P: Sample,
    I: Fn(T) -> II,
    II: IntoIterator<Item = P::Output> + 'static,
    F: Fn(P::Output) -> T + Copy,
{
    type Output = T;

    fn generate(&self, g: &mut Random) -> Self::Output {
        (self.from)(P::generate(&self.inner, g))
    }

    fn shrink(&self, v: Self::Output) -> Shrunk<Self::Output> {
        Box::new(
            (self.into)(v)
                .into_iter()
                .flat_map(|v| P::shrink(&self.inner, v))
                .map(self.from),
        )
    }
}

#[derive(Clone)]
pub struct Zip<A, B> {
    a: A,
    b: B,
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
        (self.a.generate(g), self.b.generate(g))
    }

    fn shrink(&self, v: Self::Output) -> Shrunk<Self::Output> {
        let (a, b) = v;
        Box::new(self.a.shrink(a.clone()).flat_map(move |sa| {
            std::iter::once(b.clone())
                .chain(self.b.shrink(b.clone()))
                .map(move |sb| (sa.clone(), sb))
        }))
    }
}

#[derive(Clone, Debug)]
pub struct ChainResample<S, F> {
    supersampler: S,
    transform: F,
    subsamples: usize,
}

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
            let mut g = Random::new(1000);
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

#[derive(Debug, Clone)]
pub struct Chance(pub f32);

impl Sample for Chance {
    type Output = bool;

    fn generate(&self, g: &mut Random) -> Self::Output {
        g.gen_range(0.0..1.0) < self.0
    }
}

/// Bridge for creating a [Sample] from an [Arbitrary] type.
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

/// Generate an arbitrary type.
pub fn arbitrary<T: Arbitrary>() -> ArbitrarySampler<T> {
    ArbitrarySampler {
        size: None,
        phantom: PhantomData,
        validate: |_| true,
    }
}

/// Generate a non-NaN [f32]
pub fn valid_f32() -> ArbitrarySampler<f32> {
    ArbitrarySampler {
        size: None,
        phantom: PhantomData,
        validate: |f| !f.is_nan(),
    }
}

/// Generate a non-NaN [f64]
pub fn valid_f64() -> ArbitrarySampler<f64> {
    ArbitrarySampler {
        size: None,
        phantom: PhantomData,
        validate: |f| !f.is_nan(),
    }
}

#[derive(Debug, Clone)]
pub struct Always<T>(pub T);

impl<T: Clone> Sample for Always<T> {
    type Output = T;

    fn generate(&self, _: &mut Random) -> Self::Output {
        self.0.clone()
    }
}

pub fn choice<C, II>(choices: II) -> Choice<C>
where
    II: IntoIterator<Item = C>,
    C: Sample,
    <C as Sample>::Output: Clone,
{
    Choice {
        choices: choices.into_iter().collect(),
    }
}

#[derive(Clone, Debug)]
pub struct Choice<C> {
    pub choices: Vec<C>,
}

impl<C> Choice<C> {
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

impl<C, T> Sample for Choice<C>
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
            Box::new((self.start.clone()..v.clone()).into_iter().rev().take(1))
        } else {
            Box::new(std::iter::empty())
        }
    }
}

#[derive(Clone, Debug)]
pub struct Regex {
    pub dist: rand_regex::Regex,
    pub re: regex::Regex,
}

impl Regex {
    pub fn new(pat: &str) -> Self {
        Regex {
            dist: rand_regex::Regex::compile(pat, 100).unwrap(),
            re: regex::Regex::new(pat).unwrap(),
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

/// Use a [Vec] of `samplers` to generate a new [Vec] of equal length, where
/// each sampler in the [Vec] is used to sample the value at its corresponding
/// position.
pub fn sample_all<S>(samplers: Vec<S>) -> SampleAll<S>
where
    S: Sample,
{
    SampleAll { samplers }
}

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
