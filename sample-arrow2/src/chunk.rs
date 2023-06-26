//! Chained samplers for generating arbitrary `Chunk<Box<dyn Array>>` arrow chunks.

use std::ops::Range;

use arrow2::{array::Array, chunk::Chunk, datatypes::DataType};
use sample_std::{sample_all, Chained, Sample, VecSampler};

use crate::{array::ArbitraryArray, datatypes::DataTypeSampler};

pub type ChainedChunk = Chained<(Vec<DataType>, usize), Chunk<Box<dyn Array>>>;
pub type ChunkSampler = Box<dyn Sample<Output = ChainedChunk> + Send + Sync>;

pub type ChainedMultiChunk = Chained<(Vec<DataType>, Vec<usize>), Vec<Chunk<Box<dyn Array>>>>;
pub type MultiChunkSampler = Box<dyn Sample<Output = ChainedMultiChunk> + Send + Sync>;

pub struct ArbitraryChunk<N, V> {
    pub chunk_len: Range<usize>,
    pub array_count: Range<usize>,
    pub data_type: DataTypeSampler,
    pub array: ArbitraryArray<N, V>,
}

impl<N, V> ArbitraryChunk<N, V>
where
    N: Sample<Output = String> + Send + Sync + Clone + 'static,
    V: Sample<Output = bool> + Send + Sync + Clone + 'static,
{
    pub fn sample_one(self) -> ChunkSampler {
        Box::new(
            VecSampler {
                length: self.array_count,
                el: self.data_type,
            }
            .zip(self.chunk_len)
            .chain_resample(move |seed| Self::from_seed(&self.array, seed), 100),
        )
    }

    pub fn sample_many(self, chunk_count: Range<usize>) -> MultiChunkSampler {
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
    ) -> Box<dyn Sample<Output = Chunk<Box<dyn Array>>> + Send + Sync> {
        let (dts, len) = seed;
        Box::new(
            sample_all(
                dts.into_iter()
                    .map(|data_type| array.with_len(len).sampler_from_data_type(&data_type))
                    .collect(),
            )
            .try_convert(Chunk::new, |chunk| Some(chunk.to_vec())),
        )
    }
}
