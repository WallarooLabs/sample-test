use sample_arrow2::{
    array::ArbitraryArray,
    chunk::{ArbitraryChunk, ChainedChunk, ChainedMultiChunk},
    datatypes::{sample_flat, ArbitraryDataType},
};
use sample_std::{Chance, Regex};
use sample_test::sample_test;

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
        branch: 0..5,
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

#[sample_test]
fn arbitrary_chunk(#[sample(deep_chunk(3, 100).sample_one())] chunk: ChainedChunk) {
    let chunk = chunk.value;
    assert_eq!(chunk, chunk);
}

#[sample_test]
fn arbitrary_chunks(#[sample(deep_chunk(3, 100).sample_many(2..10))] chunk: ChainedMultiChunk) {
    let chunk = chunk.value;
    assert_eq!(chunk, chunk);
}
