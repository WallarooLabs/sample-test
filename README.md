# sample-test: utilities for sample testing

[![](https://docs.rs/sample-test/badge.svg)](https://docs.rs/sample-test/)

Create tests that sample arbitrary data to produce counterexamples for a given
proposition.

# Prior Work

This library was heavily inspired by [`quickcheck`][1] and [`proptest`][2].

Here's where it differs:

- [`quickcheck`][1] works at the type level, and thus creating a new sampling
  strategy requires an entirely new type. This gets painfully verbose with
  complex types and many different sampling strategies.
- [`proptest`][2] uses macros to create strategies, and creates a tree of seed
  values to shrink data. This tree can get very large for recursive data, and
  macros can be a pain to work with.

This library attempts to split the difference. It allows user-defined `Sample`
strategies which are fed into tests. Shrinking, like with [`quickcheck`][1],
operates directly on generated values. This avoids the need to create and
maintain the seed tree used by [`proptest`][2], and allows this library to
scale up to larger generated data sizes.

Instead of macros, this library and any downstream users rely heavily on
`Sample` combinators. This is inspired by `Iterator` composition which
is very performant and concise, all without the need for any macros.

The tradeoffs are:

- `sample-test` is slightly more complicated than [`quickcheck`][1], with the
  benefit of easier definition of sampling strategies.
- `sample-test` is not as good at shrinking as [`proptest`][2] as it does not
  record the seed values that were used to generate a given output.

[1]: https://github.com/BurntSushi/quickcheck
[2]: https://github.com/proptest-rs/proptest
