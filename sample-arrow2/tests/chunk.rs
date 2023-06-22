use sample_arrow2::{
    array::ArbitraryArray,
    chunk::{ArbitraryChunk, ChainedChunk, ChainedMultiChunk, ChunkSampler, MultiChunkSampler},
    datatypes::{sample_flat, ArbitraryDataType},
};
use sample_std::{Chance, Regex};
use sample_test::{lazy_static, sample_test};
use std::boxed::Box;

fn deep_chunk(depth: usize, len: usize) -> ArbitraryChunk<Regex, Chance> {
    let names = Regex::new("[a-z]{4,8}");
    let data_type = ArbitraryDataType {
        struct_branch: 1..3,
        names: names.clone(),
        nullable: Chance(0.5),
        flat: sample_flat,
    }
    .sample_depth(depth);

    let array = ArbitraryArray {
        names,
        branch: 0..10,
        len: len..(len + 1),
        null: Chance(0.1),
        is_nullable: true,
    };

    ArbitraryChunk {
        chunk_len: 10..100,
        array_count: 4..6,
        data_type,
        array,
    }
}

lazy_static! {
    static ref DEEP_CHUNK: ChunkSampler = deep_chunk(3, 100).sample_one();
    static ref MANY_DEEP_CHUNK: MultiChunkSampler = deep_chunk(3, 100).sample_many(2..10);
}

#[sample_test]
fn arbitrary_chunk(#[sample(DEEP_CHUNK)] chunk: ChainedChunk) {
    let chunk = chunk.value;
    assert_eq!(chunk, chunk);
}

#[sample_test]
fn arbitrary_chunks(#[sample(MANY_DEEP_CHUNK)] chunk: ChainedMultiChunk) {
    let chunk = chunk.value;
    assert_eq!(chunk, chunk);
}
