use std::ops::Range;

use arrow2::{array::Array, chunk::Chunk};
use sample_std::{Sample, VecSampler};

use crate::array::ArraySampler;

pub type ChunkSampler = Box<dyn Sample<Output = Chunk<Box<dyn Array>>> + Send + Sync>;

pub fn sample_chunk(count: Range<usize>, array: ArraySampler) -> ChunkSampler {
    Box::new(
        VecSampler {
            length: count,
            el: array,
        }
        .wrap(|chunk| std::iter::once(chunk.to_vec()), Chunk::new),
    )
}
