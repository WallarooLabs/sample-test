use arrow2::datatypes::{DataType, Field};
use sample_std::{choice, Always, Random, Sample, VecSampler};

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

    fn generate(&self, g: &mut Random) -> Self::Output {
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

    fn generate(&self, g: &mut Random) -> Self::Output {
        let size = self.size.generate(g);
        DataType::Struct((0..size).map(|_| self.field.generate(g)).collect())
    }
}

pub fn sample_flat() -> DataTypeSampler {
    Box::new(choice([
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
        IF: Fn() -> DataTypeSampler,
    {
        let field = || FieldSampler {
            names: self.names.clone(),
            nullable: self.nullable.clone(),
            inner: inner(),
        };

        Box::new(choice([
            Box::new((self.flat)()) as DataTypeSampler,
            Box::new(
                VecSampler {
                    length: self.struct_branch.clone(),
                    el: field(),
                }
                .wrap(|_| std::iter::empty(), DataType::Struct),
            ),
            Box::new(field().wrap(|_| std::iter::empty(), |f| DataType::List(Box::new(f)))),
        ]))
    }

    pub fn sample_depth(&self, depth: usize) -> DataTypeSampler {
        let flats = (self.flat)();
        if depth == 0 {
            flats
        } else {
            let inner = || self.sample_depth(depth - 1);
            Box::new(choice([self.sample_nested(inner), flats]))
        }
    }
}
