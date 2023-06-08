use arrow2::{array::Array, chunk::Chunk};
use sample_arrow2::{
    array::{ArbitrarySampler, ArraySampler},
    chunk::{sample_chunk, ChunkSampler},
};
use sample_std::{Chance, Regex};
use sample_test::{lazy_static, sample_test};
use std::boxed::Box;

fn deep_array(len: usize, depth: usize) -> ArraySampler {
    Box::new(ArbitrarySampler {
        data_type_depth: depth,
        names: Regex::new("[a-z]{4,8}"),
        nullable: Chance(0.5),

        branch: 0..10,
        len: len..(len + 1),
        null: Chance(0.1),
        is_nullable: true,
    })
}

lazy_static! {
    static ref DEEP_CHUNK: ChunkSampler = sample_chunk(4..6, deep_array(100, 3));
}

#[sample_test]
fn arbitrary_chunk(#[sample(DEEP_CHUNK)] chunk: Chunk<Box<dyn Array>>) {
    assert_eq!(chunk, chunk);
}
