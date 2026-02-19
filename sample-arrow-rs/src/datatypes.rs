//! Samplers for generating an arrow [`DataType`].

use arrow_schema::{DataType, Field, Fields};
use sample_std::{sampler_choice, Always, Random, Sample, Shrunk};
use std::sync::Arc;

pub type DataTypeSampler = Box<dyn Sample<Output = DataType> + Send + Sync>;

struct FieldSampler<N, V> {
    names: N,
    nullable: V,
    inner: DataTypeSampler,
}

impl<N, V> Sample for FieldSampler<N, V>
where
    N: Sample<Output = String>,
    V: Sample<Output = bool>,
{
    type Output = Field;

    fn generate(&mut self, g: &mut Random) -> Self::Output {
        // In arrow-rs, Field::new takes name, data_type, is_nullable
        Field::new(
            self.names.generate(g),
            self.inner.generate(g),
            self.nullable.generate(g),
        )
    }
}

struct StructDataTypeSampler<S, F> {
    size: S,
    field: F,
}

impl<S, F> Sample for StructDataTypeSampler<S, F>
where
    S: Sample<Output = usize>,
    F: Sample<Output = Field>,
{
    type Output = DataType;

    fn generate(&mut self, g: &mut Random) -> Self::Output {
        let size = self.size.generate(g);
        let fields = (0..size)
            .map(|_| self.field.generate(g))
            .collect::<Fields>();
        // In arrow-rs, we use the struct constructor directly
        DataType::Struct(fields)
    }
}

pub fn sample_flat() -> DataTypeSampler {
    Box::new(sampler_choice([
        Always(DataType::Float32),
        Always(DataType::Float64),
        Always(DataType::Int8),
        Always(DataType::Int16),
        Always(DataType::Int32),
        Always(DataType::Int64),
        Always(DataType::UInt8),
        Always(DataType::UInt16),
        Always(DataType::UInt32),
        Always(DataType::UInt64),
    ]))
}

pub struct ArbitraryDataType<N, V, B, F> {
    pub names: N,
    pub nullable: V,
    pub struct_branch: B,
    pub flat: F,
}

impl<N, V, B, F> ArbitraryDataType<N, V, B, F>
where
    N: Sample<Output = String> + Clone + Send + Sync + 'static,
    V: Sample<Output = bool> + Clone + Send + Sync + 'static,
    B: Sample<Output = usize> + Clone + Send + Sync + 'static,
    F: Fn() -> DataTypeSampler,
{
    pub fn sample_nested<IF>(&self, inner: IF) -> DataTypeSampler
    where
        IF: Fn() -> DataTypeSampler + Clone,
    {
        let names_clone = self.names.clone();
        let nullable_clone = self.nullable.clone();
        let inner_clone = inner.clone();

        // Create a field constructor that can be used multiple times
        let field_constructor = move || FieldSampler {
            names: names_clone.clone(),
            nullable: nullable_clone.clone(),
            inner: inner_clone(),
        };

        // Create a list type sampler
        let list_field_sampler = {
            let names_clone = self.names.clone();
            let nullable_clone = self.nullable.clone();
            let inner_clone = inner.clone();

            move || {
                // Create a field sampler for the list element
                let field_sampler = FieldSampler {
                    names: names_clone.clone(),
                    nullable: nullable_clone.clone(),
                    inner: inner_clone(),
                };

                // Create a sampler that produces DataType::List with the generated field
                Box::new(ListFieldSampler { field_sampler }) as DataTypeSampler
            }
        };

        Box::new(sampler_choice([
            Box::new((self.flat)()) as DataTypeSampler,
            Box::new(StructDataTypeSampler {
                size: self.struct_branch.clone(),
                field: field_constructor(),
            }),
            list_field_sampler(),
        ]))
    }

    pub fn sample_depth(&self, depth: usize) -> DataTypeSampler {
        let flats = (self.flat)();
        if depth == 0 {
            flats
        } else {
            let inner = || self.sample_depth(depth - 1);
            Box::new(sampler_choice([self.sample_nested(inner), flats]))
        }
    }
}

struct ListFieldSampler<F> {
    field_sampler: F,
}

impl<F> Sample for ListFieldSampler<F>
where
    F: Sample<Output = Field>,
{
    type Output = DataType;

    fn generate(&mut self, g: &mut Random) -> Self::Output {
        let field = self.field_sampler.generate(g);
        DataType::List(Arc::new(field))
    }

    fn shrink(&self, _: Self::Output) -> Shrunk<Self::Output> {
        Box::new(std::iter::empty())
    }
}
