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

pub fn sample_nested<N, V, F>(names: N, nullable: V, inner: F) -> DataTypeSampler
where
    N: Sample<Output = String> + Clone + Send + Sync + 'static,
    V: Sample<Output = bool> + Clone + Send + Sync + 'static,
    F: Fn() -> DataTypeSampler,
{
    let field = || FieldSampler {
        names: names.clone(),
        nullable: nullable.clone(),
        inner: inner(),
    };
    Box::new(choice([
        Box::new(sample_flat()) as DataTypeSampler,
        Box::new(
            VecSampler {
                length: 1..4,
                el: field(),
            }
            .wrap(|_| std::iter::empty(), DataType::Struct),
        ),
        Box::new(field().wrap(|_| std::iter::empty(), |f| DataType::List(Box::new(f)))),
    ]))
}

pub fn sample_data_type<N, V>(depth: usize, names: N, nullable: V) -> DataTypeSampler
where
    N: Sample<Output = String> + Clone + Send + Sync + 'static,
    V: Sample<Output = bool> + Clone + Send + Sync + 'static,
{
    let flats = sample_flat();
    if depth == 0 {
        flats
    } else {
        let inner = || sample_data_type(depth - 1, names.clone(), nullable.clone());
        Box::new(choice([
            sample_nested(names.clone(), nullable.clone(), inner),
            flats,
        ]))
    }
}
