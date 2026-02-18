# sample-arrow-rs

This crate provides samplers for [arrow-rs](https://github.com/apache/arrow-rs) for use with [sample-test](https://github.com/BlockScience/sample-test).

It allows you to generate random Arrow arrays for testing purposes.

## Usage

```rust
use sample_arrow_rs::array::primitive_array;
use sample_test::Sample;
use arrow::array::Int32Array;

let mut sampler = primitive_array::<Int32Array>();
let array = sampler.sample().unwrap();
```

## Features

- Samplers for various Arrow array types
- Integration with sample-test for property-based testing
