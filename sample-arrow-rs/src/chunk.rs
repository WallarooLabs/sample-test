//! Chained samplers for generating arbitrary `RecordBatch` arrow record batches.

use std::ops::Range;
use std::sync::Arc;

use arrow_array::{ArrayRef, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use sample_std::{sample_all, Chained, Random, Sample, Shrunk, VecSampler};

use crate::{array::ArbitraryArray, datatypes::DataTypeSampler};

// In arrow-rs, we use RecordBatch instead of Chunk
pub type ChainedChunk = Chained<(Vec<DataType>, usize), RecordBatch>;
pub type ChainedMultiChunk = Chained<(Vec<DataType>, Vec<usize>), Vec<RecordBatch>>;

pub struct ArbitraryChunk<N, V> {
    pub chunk_len: Range<usize>,
    pub array_count: Range<usize>,
    pub data_type: DataTypeSampler,
    pub array: ArbitraryArray<N, V>,
}

struct RecordBatchSampler {
    arrays_sampler: Box<dyn Sample<Output = Vec<ArrayRef>> + Send + Sync>,
    schema: Arc<Schema>,
}

impl Sample for RecordBatchSampler {
    type Output = RecordBatch;

    fn generate(&mut self, g: &mut Random) -> Self::Output {
        // Generate the arrays
        let arrays = self.arrays_sampler.generate(g);

        // Create the record batch
        RecordBatch::try_new(self.schema.clone(), arrays)
            .unwrap_or_else(|_| panic!("Failed to create record batch"))
    }

    fn shrink(&self, _: Self::Output) -> Shrunk<Self::Output> {
        Box::new(std::iter::empty())
    }
}

impl<N, V> ArbitraryChunk<N, V>
where
    N: Sample<Output = String> + Send + Sync + Clone + 'static,
    V: Sample<Output = bool> + Send + Sync + Clone + 'static,
{
    pub fn sample_one(self) -> Box<dyn Sample<Output = ChainedChunk> + Send + Sync> {
        Box::new(
            VecSampler {
                length: self.array_count,
                el: self.data_type,
            }
            .zip(self.chunk_len)
            .chain_resample(move |seed| Self::from_seed(&self.array, seed), 100),
        )
    }

    pub fn sample_many(
        self,
        chunk_count: Range<usize>,
    ) -> Box<dyn Sample<Output = ChainedMultiChunk> + Send + Sync> {
        Box::new(
            VecSampler {
                length: self.array_count,
                el: self.data_type,
            }
            .zip(VecSampler {
                length: chunk_count,
                el: self.chunk_len,
            })
            .chain_resample(
                move |(dts, lens)| {
                    sample_all(
                        lens.into_iter()
                            .map(|len| Self::from_seed(&self.array, (dts.clone(), len)))
                            .collect(),
                    )
                },
                100,
            ),
        )
    }

    pub fn from_seed(
        array: &ArbitraryArray<N, V>,
        seed: (Vec<DataType>, usize),
    ) -> Box<dyn Sample<Output = RecordBatch> + Send + Sync> {
        let (dts, len) = seed;

        // Create field names for the schema
        let field_names: Vec<String> = dts
            .iter()
            .enumerate()
            .map(|(i, _)| format!("field_{}", i))
            .collect();

        // Create fields for the schema
        let fields: Vec<Field> = dts
            .iter()
            .zip(field_names.iter())
            .map(|(dt, name)| Field::new(name, dt.clone(), true))
            .collect();

        // Create the schema
        let schema = Arc::new(Schema::new(fields));

        // Generate arrays from data types
        let arrays_sampler = sample_all(
            dts.into_iter()
                .map(|data_type| array.with_len(len).sampler_from_data_type(&data_type))
                .collect(),
        );

        // Create the record batch sample generator
        Box::new(RecordBatchSampler {
            arrays_sampler: Box::new(arrays_sampler),
            schema,
        })
    }
}
